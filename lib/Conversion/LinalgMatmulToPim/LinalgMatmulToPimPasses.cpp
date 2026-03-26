//===----------------------------------------------------------------------===//
// LinalgAnnotateTileMetaPass  &  LinalgMatmulToPimCandidatePass
//
// Stage 2 of the PIM lowering pipeline:
//   Pass 1 – annotate every Triton-origin linalg.matmul with static tile sizes
//   Pass 2 – replace the annotated matmul with a pim.matmul at function entry
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/LinalgMatmulToPim/LinalgMatmulToPim.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-matmul-to-pim"

using namespace mlir;
using namespace mlir::triton;

// Auto-generated pass base classes.
#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/LinalgMatmulToPim/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Extract the NameLoc string attached to a block argument by Triton.
/// Triton stores the original Python parameter name as loc("a_ptr"), etc.
static StringRef getArgName(BlockArgument arg) {
  Location loc = arg.getLoc();
  if (auto nl = dyn_cast<NameLoc>(loc))
    return nl.getName();
  if (auto fl = dyn_cast<FusedLoc>(loc))
    for (Location inner : fl.getLocations())
      if (auto nl = dyn_cast<NameLoc>(inner))
        return nl.getName();
  return {};
}

/// Trace a Value upward through subview / cast / reinterpret_cast /
/// bufferization.to_memref / tensor ops until we reach a BlockArgument.
/// Returns a null BlockArgument if the trace is ambiguous or fails.
static BlockArgument traceToFuncArg(Value val) {
  llvm::SmallDenseSet<Value> visited;
  SmallVector<Value> worklist{val};
  BlockArgument found;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (found && found != ba)
        return BlockArgument(); // ambiguous: multiple candidates
      found = ba;
      continue;
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;

    TypeSwitch<Operation *>(defOp)
        .Case<memref::SubViewOp>(
            [&](auto op) { worklist.push_back(op.getSource()); })
        .Case<memref::CastOp>(
            [&](auto op) { worklist.push_back(op.getSource()); })
        .Case<memref::ReinterpretCastOp>(
            [&](auto op) { worklist.push_back(op.getSource()); })
        .Case<bufferization::ToBufferOp>(
            [&](auto op) { worklist.push_back(op.getTensor()); })
        .Case<tensor::ExtractSliceOp>(
            [&](auto op) { worklist.push_back(op.getSource()); })
        .Case<tensor::CastOp>(
            [&](auto op) { worklist.push_back(op.getSource()); })
        .Default([&](Operation *op) {
          // Generic fall-through: trace all operands.
          for (Value operand : op->getOperands())
            worklist.push_back(operand);
        });
  }
  return found;
}

/// Find the output element type of a linalg.matmul (from its outs operand).
static Type getMatmulOutputElemType(linalg::MatmulOp matmul) {
  Type outType = matmul.getOutputs()[0].getType();
  if (auto mr = dyn_cast<MemRefType>(outType))
    return mr.getElementType();
  if (auto tt = dyn_cast<TensorType>(outType))
    return tt.getElementType();
  return {};
}

/// Find the first function argument that is a 2-D MemRef with the given
/// element type, excluding already-identified arguments.
static BlockArgument findMemRefArgByElemType(func::FuncOp func, Type elemType,
                                             ArrayRef<BlockArgument> excluded) {
  for (BlockArgument arg : func.getArguments()) {
    auto mr = dyn_cast<MemRefType>(arg.getType());
    if (!mr || mr.getRank() != 2 || mr.getElementType() != elemType)
      continue;
    if (llvm::is_contained(excluded, arg))
      continue;
    return arg;
  }
  return BlockArgument();
}

//===----------------------------------------------------------------------===//
// Pass 1: LinalgAnnotateTileMetaPass
//===----------------------------------------------------------------------===//

namespace {
struct LinalgAnnotateTileMetaPass
    : public LinalgAnnotateTileMetaBase<LinalgAnnotateTileMetaPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    func.walk([&](linalg::MatmulOp matmul) {
      // Only annotate Triton-origin matmuls.
      if (!matmul->hasAttr("triton.dot_origin"))
        return;

      // Read tile sizes from input operand types.
      // A operand: [BM x BK x elem_type]
      // B operand: [BK x BN x elem_type]
      auto getStaticSize = [](Value v, int dim) -> int64_t {
        Type t = v.getType();
        if (auto mr = dyn_cast<MemRefType>(t))
          return mr.getShape()[dim];
        if (auto tt = dyn_cast<TensorType>(t))
          return tt.getShape()[dim];
        return ShapedType::kDynamic;
      };

      int64_t bm = getStaticSize(matmul.getInputs()[0], 0);
      int64_t bk = getStaticSize(matmul.getInputs()[0], 1);
      int64_t bn = getStaticSize(matmul.getInputs()[1], 1);

      if (bm == ShapedType::kDynamic || bk == ShapedType::kDynamic ||
          bn == ShapedType::kDynamic) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[linalg-annotate-tile-meta] dynamic tile shapes, "
                      "skipping annotation\n");
        return;
      }

      auto tileMeta = builder.getDictionaryAttr({
          builder.getNamedAttr("bm", builder.getI64IntegerAttr(bm)),
          builder.getNamedAttr("bn", builder.getI64IntegerAttr(bn)),
          builder.getNamedAttr("bk", builder.getI64IntegerAttr(bk)),
      });
      matmul->setAttr("triton.tile_meta", tileMeta);

      LLVM_DEBUG(llvm::dbgs()
                 << "[linalg-annotate-tile-meta] annotated matmul with "
                    "bm=" << bm << " bn=" << bn << " bk=" << bk << "\n");
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createLinalgAnnotateTileMetaPass() {
  return std::make_unique<LinalgAnnotateTileMetaPass>();
}

//===----------------------------------------------------------------------===//
// Pass 2: LinalgMatmulToPimCandidatePass
//===----------------------------------------------------------------------===//

namespace {
struct LinalgMatmulToPimCandidatePass
    : public LinalgMatmulToPimCandidateBase<LinalgMatmulToPimCandidatePass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // ----------------------------------------------------------------
    // Step 1: collect all Triton-origin matmuls.
    // ----------------------------------------------------------------
    SmallVector<linalg::MatmulOp> pimMatmuls;
    func.walk([&](linalg::MatmulOp matmul) {
      if (matmul->hasAttr("triton.dot_origin"))
        pimMatmuls.push_back(matmul);
    });

    if (pimMatmuls.size() != 1) {
      LLVM_DEBUG({
        if (pimMatmuls.empty())
          llvm::dbgs() << "[linalg-matmul-to-pim-candidate] "
                       << func.getName() << ": no triton.dot_origin matmul, skipping\n";
        else
          llvm::dbgs() << "[linalg-matmul-to-pim-candidate] "
                       << func.getName() << ": " << pimMatmuls.size()
                       << " triton.dot_origin matmuls, skipping\n";
      });
      return;
    }

    linalg::MatmulOp matmul = pimMatmuls[0];

    // ----------------------------------------------------------------
    // Step 2: read tile sizes from triton.tile_meta (or fallback to
    //         static shapes of the matmul operands).
    // ----------------------------------------------------------------
    int64_t bm = 0, bn = 0, bk = 0;
    if (auto tileMetaAttr =
            matmul->getAttrOfType<DictionaryAttr>("triton.tile_meta")) {
      if (auto a = tileMetaAttr.get("bm"))
        bm = cast<IntegerAttr>(a).getInt();
      if (auto a = tileMetaAttr.get("bn"))
        bn = cast<IntegerAttr>(a).getInt();
      if (auto a = tileMetaAttr.get("bk"))
        bk = cast<IntegerAttr>(a).getInt();
    }

    // Fallback: read directly from operand types.
    if (bm == 0 || bn == 0 || bk == 0) {
      auto getStatic = [](Value v, int dim) -> int64_t {
        Type t = v.getType();
        if (auto mr = dyn_cast<MemRefType>(t))
          return mr.getShape()[dim];
        if (auto tt = dyn_cast<TensorType>(t))
          return tt.getShape()[dim];
        return 0;
      };
      bm = getStatic(matmul.getInputs()[0], 0);
      bk = getStatic(matmul.getInputs()[0], 1);
      bn = getStatic(matmul.getInputs()[1], 1);
    }

    // ----------------------------------------------------------------
    // Step 3: trace matmul inputs to full-matrix function arguments.
    // ----------------------------------------------------------------
    BlockArgument fullA = traceToFuncArg(matmul.getInputs()[0]);
    BlockArgument fullB = traceToFuncArg(matmul.getInputs()[1]);

    // ----------------------------------------------------------------
    // Step 4: find full C.
    // The matmul's outs operand may be a temporary alloc, not the real
    // output arg.  Try (a) trace, (b) NameLoc, (c) elem-type match.
    // ----------------------------------------------------------------
    BlockArgument fullC = traceToFuncArg(matmul.getOutputs()[0]);

    if (!fullC) {
      // (b) NameLoc search.
      for (BlockArgument arg : func.getArguments()) {
        StringRef name = getArgName(arg);
        if (name == "c_ptr" || name == "c" || name == "C") {
          fullC = arg;
          break;
        }
      }
    }

    if (!fullC) {
      // (c) Element-type match: find the first 2-D memref with the output
      //     element type that is neither fullA nor fullB.
      Type outElemType = getMatmulOutputElemType(matmul);
      SmallVector<BlockArgument> excluded;
      if (fullA) excluded.push_back(fullA);
      if (fullB) excluded.push_back(fullB);
      fullC = findMemRefArgByElemType(func, outElemType, excluded);
    }

    if (!fullA || !fullB || !fullC) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[linalg-matmul-to-pim-candidate] " << func.getName()
                 << ": could not resolve A/B/C function args, skipping\n");
      return;
    }

    // Verify the resolved args are actually 2-D MemRefs.
    auto checkMemRef = [&](BlockArgument arg, StringRef label) -> bool {
      if (!isa<MemRefType>(arg.getType())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[linalg-matmul-to-pim-candidate] " << label
                   << " is not a memref type, skipping\n");
        return false;
      }
      return true;
    };
    if (!checkMemRef(fullA, "A") || !checkMemRef(fullB, "B") ||
        !checkMemRef(fullC, "C"))
      return;

    // ----------------------------------------------------------------
    // Step 5: insert pim.matmul at the function entry with a partial plan.
    // ----------------------------------------------------------------
    Location loc = matmul.getLoc();
    OpBuilder builder(&func.getBody().front().front());

    // Dynamic M, N, K from the full matrix dimensions.
    Value M = builder.create<memref::DimOp>(loc, fullC, 0);
    Value N = builder.create<memref::DimOp>(loc, fullC, 1);
    Value K = builder.create<memref::DimOp>(loc, fullA, 1);

    // Partial plan: tile sizes filled in, everything else UNKNOWN/0.
    auto plan = pim::ExecutionPlanAttr::get(
        func.getContext(),
        /*tile_m=*/bm, /*tile_n=*/bn, /*tile_k=*/bk,
        pim::SplitAxis::UNKNOWN, pim::ReusePolicy::UNKNOWN,
        pim::ReductionStrategy::UNKNOWN,
        /*tasklets=*/0, /*active_dpus=*/0,
        pim::KernelVariant::UNKNOWN, pim::PackFormat::UNKNOWN,
        pim::AccumType::UNKNOWN, pim::WritebackMode::UNKNOWN,
        /*alignment=*/0, /*group_m=*/0, /*batch_count=*/1);

    builder.create<pim::MatmulOp>(loc, fullA, fullB, fullC, M, N, K, plan);

    LLVM_DEBUG(llvm::dbgs()
               << "[linalg-matmul-to-pim-candidate] " << func.getName()
               << ": inserted pim.matmul (bm=" << bm << " bn=" << bn
               << " bk=" << bk << ")\n");

    // ----------------------------------------------------------------
    // Step 6: erase the linalg.matmul.  Dead subviews / loops are cleaned
    //         by a subsequent canonicalize pass.
    // ----------------------------------------------------------------
    matmul.erase();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createLinalgMatmulToPimCandidatePass() {
  return std::make_unique<LinalgMatmulToPimCandidatePass>();
}
