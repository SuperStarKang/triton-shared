//===----------------------------------------------------------------------===//
//
// UPMEM PIM backend metadata extraction pass for Triton-Shared.
//
// This pass walks linalg.matmul ops and annotates the enclosing func.func
// with "upmem.*" attributes that the Python PIM runtime reads to configure
// a DPU kernel launch.  The pass is purely additive – no op is modified –
// so the standard CPU lowering pipeline continues to work unchanged.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/LinalgMatmulToUpmem/LinalgMatmulToUpmem.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "linalg-matmul-to-upmem"

using namespace mlir;
using namespace mlir::triton;

// Auto-generated pass base class.
#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/LinalgMatmulToUpmem/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: map an MLIR scalar type to the dtype string the PIM runtime uses.
// ---------------------------------------------------------------------------
static std::string getTypeName(Type t) {
  if (t.isInteger(8))  return "i8";
  if (t.isInteger(16)) return "i16";
  if (t.isInteger(32)) return "i32";
  if (t.isInteger(64)) return "i64";
  if (t.isF16())       return "f16";
  if (t.isBF16())      return "bf16";
  if (t.isF32())       return "f32";
  if (t.isF64())       return "f64";
  return "unknown";
}

// ---------------------------------------------------------------------------
// Helper: try to extract the NameLoc string from a block argument's location.
// Triton stores the original Python parameter name as a NameLoc on every
// function block argument (visible in IR text as  loc("a_ptr")).
// ---------------------------------------------------------------------------
static StringRef getArgName(BlockArgument arg) {
  Location loc = arg.getLoc();
  // Direct NameLoc.
  if (auto nl = dyn_cast<NameLoc>(loc))
    return nl.getName();
  // FusedLoc whose first child is a NameLoc.
  if (auto fl = dyn_cast<FusedLoc>(loc)) {
    for (Location inner : fl.getLocations())
      if (auto nl = dyn_cast<NameLoc>(inner))
        return nl.getName();
  }
  return {};
}

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------
class LinalgMatmulToUpmemPass
    : public LinalgMatmulToUpmemBase<LinalgMatmulToUpmemPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](func::FuncOp func) {
      // ----------------------------------------------------------------
      // 1. Find the first linalg.matmul in this function.
      //    (Triton GEMM kernels contain exactly one matmul per function.)
      // ----------------------------------------------------------------
      linalg::MatmulOp matmulOp;
      func.walk([&](linalg::MatmulOp op) {
        matmulOp = op;
        return WalkResult::interrupt();
      });

      if (!matmulOp)
        return; // Not a matmul kernel – skip.

      // ----------------------------------------------------------------
      // 2. Extract tile sizes and element types from the matmul operands.
      //
      //   linalg.matmul ins(%a: tensor<BM×BK×Ti>, %b: tensor<BK×BN×Ti>)
      //                 outs(%c: tensor<BM×BN×To>)
      //
      // Both tensor and memref (ShapedType) are handled uniformly.
      // ----------------------------------------------------------------
      auto aType = dyn_cast<ShapedType>(matmulOp.getInputs()[0].getType());
      auto bType = dyn_cast<ShapedType>(matmulOp.getInputs()[1].getType());
      auto cType = dyn_cast<ShapedType>(matmulOp.getOutputs()[0].getType());

      if (!aType || !bType || !cType) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[linalg-matmul-to-upmem] operand types are not "
                      "ShapedType – skipping function '"
                   << func.getName() << "'\n");
        return;
      }

      // Require static shapes (BLOCK_M/K/N are constexpr in Triton).
      if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
          !cType.hasStaticShape()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[linalg-matmul-to-upmem] dynamic shapes – skipping '"
                   << func.getName() << "'\n");
        return;
      }

      int64_t bm = aType.getShape()[0]; // rows of A  == tile rows
      int64_t bk = aType.getShape()[1]; // cols of A  == tile k-dim
      int64_t bn = bType.getShape()[1]; // cols of B  == tile cols

      std::string elemTypeName = getTypeName(aType.getElementType());
      std::string accTypeName  = getTypeName(cType.getElementType());


      // ----------------------------------------------------------------
      // 3. Identify function argument indices by NameLoc.
      //
      //   Triton attaches  loc("a_ptr") / loc("b_ptr") etc. to every
      //   block argument so we can resolve them reliably in C++.
      // ----------------------------------------------------------------
      int aIdx = -1, bIdx = -1, cIdx = -1;
      int mIdx = -1, nIdx = -1, kIdx = -1;

      for (auto [idx, arg] :
           llvm::enumerate(func.getBody().front().getArguments())) {
        StringRef name = getArgName(arg);
        if (name.empty())
          continue;
        if (name == "a_ptr")       aIdx = static_cast<int>(idx);
        else if (name == "b_ptr")  bIdx = static_cast<int>(idx);
        else if (name == "c_ptr")  cIdx = static_cast<int>(idx);
        else if (name == "m")      mIdx = static_cast<int>(idx);
        else if (name == "n")      nIdx = static_cast<int>(idx);
        else if (name == "k")      kIdx = static_cast<int>(idx);
      }

      if (aIdx < 0 || bIdx < 0 || cIdx < 0 ||
          mIdx < 0 || nIdx < 0 || kIdx < 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[linalg-matmul-to-upmem] could not resolve all "
                      "argument names – skipping '"
                   << func.getName() << "'\n");
        return;
      }

      // ----------------------------------------------------------------
      // 4. Attach upmem.* attributes to the function.
      //    These are read by the Python compiler to build pim_meta.
      // ----------------------------------------------------------------
      MLIRContext *ctx = module.getContext();

      // Tile sizes.
      func->setAttr("upmem.bm", IntegerAttr::get(IntegerType::get(ctx, 32),
                                                   bm));
      func->setAttr("upmem.bk", IntegerAttr::get(IntegerType::get(ctx, 32),
                                                   bk));
      func->setAttr("upmem.bn", IntegerAttr::get(IntegerType::get(ctx, 32),
                                                   bn));

      // Pointer argument indices.
      func->setAttr("upmem.a_ptr_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), aIdx));
      func->setAttr("upmem.b_ptr_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), bIdx));
      func->setAttr("upmem.c_ptr_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), cIdx));

      // M / N / K argument indices.
      func->setAttr("upmem.m_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), mIdx));
      func->setAttr("upmem.n_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), nIdx));
      func->setAttr("upmem.k_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), kIdx));


      // Type strings (used by the runtime to choose the DPU kernel variant).
      func->setAttr("upmem.elem_type", StringAttr::get(ctx, elemTypeName));
      func->setAttr("upmem.acc_type",  StringAttr::get(ctx, accTypeName));

      LLVM_DEBUG(llvm::dbgs()
                 << "[linalg-matmul-to-upmem] annotated '"
                 << func.getName()
                 << "' bm=" << bm << " bk=" << bk << " bn=" << bn
                 << " a=" << aIdx << " b=" << bIdx << " c=" << cIdx
                 << " m=" << mIdx << " n=" << nIdx << " k=" << kIdx
                 << " elem=" << elemTypeName
                 << " acc=" << accTypeName << "\n");
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLinalgMatmulToUpmemPass() {
  return std::make_unique<LinalgMatmulToUpmemPass>();
}
