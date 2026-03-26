//===----------------------------------------------------------------------===//
// PimToRuntimeCallsPass
//
// Stage 4 of the PIM lowering pipeline:
//   Replaces every pim.matmul with a call to the external C function
//   triton_pim_matmul() declared in triton-shared/runtime/pim_runtime.h.
//
// Generated call signature:
//   triton_pim_matmul(
//     i64 a, i64 b, i64 c,            -- aligned base pointers
//     i64 m, i64 n, i64 k,            -- problem dimensions
//     i64 tile_m, i64 tile_n, i64 tile_k,
//     i32 split_axis, i32 reuse_policy, i32 reduction,
//     i32 tasklets, i32 active_dpus,
//     i32 kernel_variant, i32 pack_format, i32 accum_type, i32 writeback_mode,
//     i32 alignment, i32 group_m, i32 batch_count
//   )
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/PimToRuntimeCalls/PimToRuntimeCalls.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pim-to-runtime-calls"

// Runtime function name that matches pim_runtime.h.
static constexpr llvm::StringLiteral kRuntimeFunc = "triton_pim_matmul";

using namespace mlir;
using namespace mlir::triton;

// Auto-generated pass base classes.
#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/PimToRuntimeCalls/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return the FuncType for `triton_pim_matmul`.
/// void(i64×3, i64×3, i64×3, i32×9)  →  21 arguments total.
static FunctionType buildRuntimeFuncType(MLIRContext *ctx) {
  auto i64 = IntegerType::get(ctx, 64);
  auto i32 = IntegerType::get(ctx, 32);
  SmallVector<Type> args = {
      i64, i64, i64,   // a, b, c  pointers
      i64, i64, i64,   // m, n, k
      i64, i64, i64,   // tile_m, tile_n, tile_k
      i32, i32, i32,   // split_axis, reuse_policy, reduction
      i32, i32,        // tasklets, active_dpus
      i32, i32, i32, i32, // kernel_variant, pack_format, accum_type, writeback_mode
      i32, i32, i32,   // alignment, group_m, batch_count
  };
  return FunctionType::get(ctx, args, /*results=*/{});
}

/// Find `triton_pim_matmul` in the module, or insert a private declaration.
static func::FuncOp getOrInsertRuntimeDecl(OpBuilder &b, ModuleOp mod) {
  if (auto existing = mod.lookupSymbol<func::FuncOp>(kRuntimeFunc))
    return existing;

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(mod.getBody());
  auto decl = b.create<func::FuncOp>(
      mod.getLoc(), kRuntimeFunc,
      buildRuntimeFuncType(mod.getContext()));
  decl.setPrivate();
  return decl;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct PimToRuntimeCallsPass
    : public PimToRuntimeCallsBase<PimToRuntimeCallsPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    // Collect all pim.matmul ops before modifying the IR.
    SmallVector<pim::MatmulOp> matmuls;
    mod.walk([&](pim::MatmulOp op) { matmuls.push_back(op); });

    if (matmuls.empty())
      return;

    // Ensure the runtime declaration exists.
    func::FuncOp decl = getOrInsertRuntimeDecl(builder, mod);

    auto i64 = IntegerType::get(mod.getContext(), 64);
    auto i32 = IntegerType::get(mod.getContext(), 32);

    for (pim::MatmulOp matmul : matmuls) {
      OpBuilder b(matmul);
      Location loc = matmul.getLoc();
      pim::ExecutionPlanAttr plan = matmul.getPlan();

      // ------------------------------------------------------------------
      // Extract aligned pointers from A, B, C memrefs → i64.
      // ------------------------------------------------------------------
      auto extractPtr = [&](Value memref) -> Value {
        Value idx = b.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, memref);
        return b.create<arith::IndexCastUIOp>(loc, i64, idx);
      };
      Value ptrA = extractPtr(matmul.getA());
      Value ptrB = extractPtr(matmul.getB());
      Value ptrC = extractPtr(matmul.getC());

      // ------------------------------------------------------------------
      // Cast index-typed m / n / k operands to i64.
      // ------------------------------------------------------------------
      auto toI64 = [&](Value v) -> Value {
        return b.create<arith::IndexCastUIOp>(loc, i64, v);
      };
      Value m = toI64(matmul.getMSize());
      Value n = toI64(matmul.getNSize());
      Value k = toI64(matmul.getKSize());

      // ------------------------------------------------------------------
      // Materialise plan fields as constants.
      // ------------------------------------------------------------------
      auto cI64 = [&](int64_t v) -> Value {
        return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(i64, v));
      };
      auto cI32 = [&](int32_t v) -> Value {
        return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(i32, v));
      };

      Value tileM = cI64(plan.getTileM());
      Value tileN = cI64(plan.getTileN());
      Value tileK = cI64(plan.getTileK());

      Value splitAxis  = cI32(static_cast<int32_t>(plan.getSplitAxis()));
      Value reusePol   = cI32(static_cast<int32_t>(plan.getReusePolicy()));
      Value reduction  = cI32(static_cast<int32_t>(plan.getReduction()));
      Value tasklets   = cI32(plan.getTasklets());
      Value activeDpus = cI32(plan.getActiveDpus());
      Value kernelVar  = cI32(static_cast<int32_t>(plan.getKernelVariant()));
      Value packFmt    = cI32(static_cast<int32_t>(plan.getPackFormat()));
      Value accumType  = cI32(static_cast<int32_t>(plan.getAccumType()));
      Value writeMode  = cI32(static_cast<int32_t>(plan.getWritebackMode()));
      Value alignment  = cI32(plan.getAlignment());
      Value groupM     = cI32(plan.getGroupM());
      Value batchCnt   = cI32(plan.getBatchCount());

      // ------------------------------------------------------------------
      // Emit the runtime call and erase the pim.matmul.
      // ------------------------------------------------------------------
      SmallVector<Value> args = {
          ptrA, ptrB, ptrC,
          m, n, k,
          tileM, tileN, tileK,
          splitAxis, reusePol, reduction,
          tasklets, activeDpus,
          kernelVar, packFmt, accumType, writeMode,
          alignment, groupM, batchCnt,
      };
      b.create<func::CallOp>(loc, decl, args);

      LLVM_DEBUG(llvm::dbgs()
                 << "[pim-to-runtime-calls] lowered pim.matmul → "
                 << kRuntimeFunc << "\n");

      matmul.erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPimToRuntimeCallsPass() {
  return std::make_unique<PimToRuntimeCallsPass>();
}
