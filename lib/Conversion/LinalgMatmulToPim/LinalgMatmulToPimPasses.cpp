//===----------------------------------------------------------------------===//
// Stage 2 – LinalgAnnotateTileMetaPass & LinalgMatmulToPimCandidatePass
// Stage 3 – PimPlanMaterializePass    & PimLayoutVerifyPass
//
// Lowering pipeline for Triton-origin matmuls → pim.matmul:
//   Pass 1 – annotate every Triton-origin linalg.matmul with static tile sizes
//   Pass 2 – replace the annotated matmul with a pim.matmul (partial plan)
//   Pass 3 – fill in UNKNOWN plan fields from types / hardware defaults
//   Pass 4 – verify the completed plan is consistent
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/LinalgMatmulToPim/LinalgMatmulToPim.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
    // Step 2b: gate on i8 inputs — the UPMEM DPU runtime only supports
    // integer (int8) matrix multiply.  Skip f32 / f16 / bf16 kernels so
    // that they fall back to the normal CPU path without NaN corruption.
    // ----------------------------------------------------------------
    {
      auto getInputElemType = [](Value v) -> Type {
        Type t = v.getType();
        if (auto mr = dyn_cast<MemRefType>(t)) return mr.getElementType();
        if (auto tt = dyn_cast<TensorType>(t)) return tt.getElementType();
        return {};
      };
      Type aElem = getInputElemType(matmul.getInputs()[0]);
      if (!aElem || !aElem.isInteger(8)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[linalg-matmul-to-pim-candidate] " << func.getName()
                   << ": input element type is not i8 ("
                   << (aElem ? "" : "unknown") << "), skipping\n");
        return;
      }
    }

    // ----------------------------------------------------------------
    // Step 3: trace matmul inputs to full-matrix function arguments.
    // SSA tracing may fail when tile loading uses alloc+copy pattern
    // (to_tensor(%alloc) ← memref.copy ← reinterpret_cast(%arg0)).
    // Fall back to NameLoc search in that case.
    // ----------------------------------------------------------------
    BlockArgument fullA = traceToFuncArg(matmul.getInputs()[0]);
    BlockArgument fullB = traceToFuncArg(matmul.getInputs()[1]);

    if (!fullA) {
      for (BlockArgument arg : func.getArguments()) {
        StringRef n = getArgName(arg);
        if (n == "a_ptr" || n == "a" || n == "A") { fullA = arg; break; }
      }
    }
    if (!fullB) {
      for (BlockArgument arg : func.getArguments()) {
        StringRef n = getArgName(arg);
        if (n == "b_ptr" || n == "b" || n == "B") { fullB = arg; break; }
      }
    }

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

    // Verify the resolved args are MemRefs (ranked or unranked).
    auto checkMemRef = [&](BlockArgument arg, StringRef label) -> bool {
      if (!isa<MemRefType>(arg.getType()) &&
          !isa<UnrankedMemRefType>(arg.getType())) {
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

    // Prefer the Triton integer kernel arguments M, N, K (i32) over DimOp on
    // unranked memrefs, which can cause issues during LLVM lowering.
    auto findIntArg = [&](StringRef name) -> Value {
      for (BlockArgument arg : func.getArguments()) {
        if (getArgName(arg) == name && arg.getType().isInteger())
          return arg;
      }
      return Value();
    };
    Value dynM = findIntArg("M"), dynN = findIntArg("N"),
          dynK = findIntArg("K");

    auto idxType = builder.getIndexType();
    Value M, N, K;
    if (dynM && dynN && dynK) {
      M = builder.create<arith::IndexCastOp>(loc, idxType, dynM);
      N = builder.create<arith::IndexCastOp>(loc, idxType, dynN);
      K = builder.create<arith::IndexCastOp>(loc, idxType, dynK);
    } else {
      M = builder.create<memref::DimOp>(loc, fullC, 0);
      N = builder.create<memref::DimOp>(loc, fullC, 1);
      K = builder.create<memref::DimOp>(loc, fullA, 1);
    }

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
    // Step 6: intentionally keep the linalg.matmul in the IR.
    //
    // Erasing it while the linalg.generic accumulation op still holds a
    // reference to the matmul's tensor result causes a use-after-free that
    // manifests as SIGSEGV inside --canonicalize's verifier
    // (hasPureTensorSemantics iterates freed operand Values).
    //
    // With the matmul intact the execution model is:
    //   • pim.matmul  – computes the full M×N result into fullC (c_ptr)
    //   • original for-loop – runs for pid_m=0, pid_n=0 (grid overridden to
    //     (1,1,1)); computes the first BM×BN tile and tl.stores it to
    //     c_ptr[0:BM, 0:BN] — identical to the pim.matmul result there.
    // Final c_ptr is correct everywhere.
    // ----------------------------------------------------------------
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createLinalgMatmulToPimCandidatePass() {
  return std::make_unique<LinalgMatmulToPimCandidatePass>();
}

//===----------------------------------------------------------------------===//
// Pass 3: PimPlanMaterializePass
//
// Fills in every UNKNOWN / zero field of a pim.matmul execution plan.
// Fields that already carry concrete values are left untouched (idempotent).
//
// Heuristics (all overridable by pre-filling the plan before this pass):
//   split_axis     = N        (distribute output columns across DPUs)
//   reuse_policy   = REUSE_A  (A is reused; each DPU holds a tile of B)
//   reduction      = HOST_REDUCE
//   writeback_mode = DIRECT
//   kernel_variant = GROUPED if group_m > 0, else FLAT
//   pack_format    = INT8 for i8 inputs, NONE otherwise
//   accum_type     = INT32 for integer outputs, FLOAT32 otherwise
//   tasklets       = 16       (safe default for all UPMEM DPU generations)
//   active_dpus    = ceil(M/tile_m)*ceil(N/tile_n) capped at 2560, or 2560
//                   when the matrix shapes are dynamic
//   alignment      = 8        (minimum DPU transfer alignment in bytes)
//===----------------------------------------------------------------------===//

namespace {
struct PimPlanMaterializePass
    : public PimPlanMaterializeBase<PimPlanMaterializePass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect (op, newPlan) pairs first; mutate after the walk.
    SmallVector<std::pair<pim::MatmulOp, pim::ExecutionPlanAttr>> toUpdate;

    func.walk([&](pim::MatmulOp matmul) {
      pim::ExecutionPlanAttr plan = matmul.getPlan();

      // ----------------------------------------------------------------
      // Derive element-type-dependent fields.
      // Handles both ranked MemRefType and UnrankedMemRefType.
      // ----------------------------------------------------------------
      auto getElemType = [](Type t) -> Type {
        if (auto mr = dyn_cast<MemRefType>(t))
          return mr.getElementType();
        if (auto umr = dyn_cast<UnrankedMemRefType>(t))
          return umr.getElementType();
        return {};
      };
      Type aElem = getElemType(matmul.getA().getType());
      Type cElem = getElemType(matmul.getC().getType());

      // KernelVariant
      pim::KernelVariant kv = plan.getKernelVariant();
      if (kv == pim::KernelVariant::UNKNOWN)
        kv = (plan.getGroupM() > 0) ? pim::KernelVariant::GROUPED
                                    : pim::KernelVariant::FLAT;

      // PackFormat
      pim::PackFormat pf = plan.getPackFormat();
      if (pf == pim::PackFormat::UNKNOWN)
        pf = aElem.isInteger(8) ? pim::PackFormat::INT8
                                : pim::PackFormat::NONE;

      // AccumType
      pim::AccumType at = plan.getAccumType();
      if (at == pim::AccumType::UNKNOWN)
        at = (cElem.isInteger(32) || cElem.isInteger(8))
                 ? pim::AccumType::INT32
                 : pim::AccumType::FLOAT32;

      // ----------------------------------------------------------------
      // Derive layout / distribution fields.
      // ----------------------------------------------------------------
      pim::SplitAxis sa = plan.getSplitAxis();
      if (sa == pim::SplitAxis::UNKNOWN)
        sa = pim::SplitAxis::N;

      pim::ReusePolicy rp = plan.getReusePolicy();
      if (rp == pim::ReusePolicy::UNKNOWN)
        rp = (sa == pim::SplitAxis::N) ? pim::ReusePolicy::REUSE_A
                                       : pim::ReusePolicy::REUSE_B;

      // Reduction is only meaningful for K-split; non-K splits must leave it
      // UNKNOWN (the verifier enforces this as a consistency constraint).
      pim::ReductionStrategy rs = plan.getReduction();
      if (rs == pim::ReductionStrategy::UNKNOWN && sa == pim::SplitAxis::K)
        rs = pim::ReductionStrategy::HOST_REDUCE;

      pim::WritebackMode wm = plan.getWritebackMode();
      if (wm == pim::WritebackMode::UNKNOWN)
        wm = pim::WritebackMode::DIRECT;

      // ----------------------------------------------------------------
      // Compute active_dpus: prefer TRITON_PIM_NDPU env var, then static
      // shape, then kMaxDpus.  PimToRuntimeCallsPass also reads the env var
      // at opt-run time and embeds the exact value as a constant — this pass
      // only stores it in the plan attr for debugging / verification purposes.
      // ----------------------------------------------------------------
      const char* ndpuEnv = getenv("TRITON_PIM_NDPU");
      const int32_t kMaxDpus = (ndpuEnv && *ndpuEnv)
                                   ? (int32_t)atoi(ndpuEnv)
                                   : 2560;
      int32_t activeDpus = plan.getActiveDpus();
      if (activeDpus == 0) {
        // Static shape is only available for ranked memrefs.
        auto cRanked = dyn_cast<MemRefType>(matmul.getC().getType());
        int64_t staticM = (cRanked && cRanked.getRank() >= 1)
                              ? cRanked.getShape()[0]
                              : ShapedType::kDynamic;
        int64_t staticN = (cRanked && cRanked.getRank() >= 2)
                              ? cRanked.getShape()[1]
                              : ShapedType::kDynamic;
        int64_t tileM   = plan.getTileM();
        int64_t tileN   = plan.getTileN();
        if (staticM != ShapedType::kDynamic && staticN != ShapedType::kDynamic
            && tileM > 0 && tileN > 0) {
          int64_t dpuM   = (staticM + tileM - 1) / tileM;
          int64_t dpuN   = (staticN + tileN - 1) / tileN;
          int64_t needed = dpuM * dpuN;
          activeDpus = static_cast<int32_t>(needed < kMaxDpus ? needed : kMaxDpus);
        } else {
          activeDpus = kMaxDpus;
        }
      }

      // ----------------------------------------------------------------
      // Fixed hardware defaults.
      // ----------------------------------------------------------------
      const char* tlEnv  = getenv("TRITON_PIM_TASKLETS");
      int32_t tasklets   = plan.getTasklets() > 0 ? plan.getTasklets()
                           : (tlEnv && *tlEnv) ? (int32_t)atoi(tlEnv) : 16;
      int32_t alignment  = plan.getAlignment()  > 0 ? plan.getAlignment()  : 8;
      int32_t groupM     = plan.getGroupM();
      int32_t batchCount = plan.getBatchCount() > 0 ? plan.getBatchCount() : 1;

      pim::ExecutionPlanAttr newPlan = pim::ExecutionPlanAttr::get(
          func.getContext(),
          plan.getTileM(), plan.getTileN(), plan.getTileK(),
          sa, rp, rs,
          tasklets, activeDpus,
          kv, pf, at, wm,
          alignment, groupM, batchCount);

      toUpdate.emplace_back(matmul, newPlan);

      LLVM_DEBUG(llvm::dbgs()
                 << "[pim-plan-materialize] " << func.getName()
                 << ": materialized plan (split=" << (int)sa
                 << " dpus=" << activeDpus << ")\n");
    });

    // Replace each matmul with a new one carrying the materialized plan.
    for (auto &[matmul, newPlan] : toUpdate) {
      OpBuilder b(matmul);
      b.create<pim::MatmulOp>(matmul.getLoc(),
                               matmul.getA(), matmul.getB(), matmul.getC(),
                               matmul.getMSize(), matmul.getNSize(),
                               matmul.getKSize(), newPlan);
      matmul.erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createPimPlanMaterializePass() {
  return std::make_unique<PimPlanMaterializePass>();
}

//===----------------------------------------------------------------------===//
// Pass 4: PimLayoutVerifyPass
//
// Verifies that every pim.matmul execution plan is complete and consistent.
// Emits error diagnostics and signals pass failure for each violated constraint.
//
// Checked constraints:
//   1. All enum fields (split_axis, reuse_policy, reduction, kernel_variant,
//      pack_format, accum_type, writeback_mode) must be non-UNKNOWN.
//   2. Integer fields (tasklets, active_dpus, alignment, tile_m/n/k) must be > 0.
//   3. group_m > 0 → kernel_variant must be GROUPED.
//   4. split_axis == K → reduction must not be UNKNOWN.
//===----------------------------------------------------------------------===//

namespace {
struct PimLayoutVerifyPass
    : public PimLayoutVerifyBase<PimLayoutVerifyPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool failed = false;

    func.walk([&](pim::MatmulOp matmul) {
      pim::ExecutionPlanAttr p = matmul.getPlan();

      auto err = [&](StringRef msg) {
        matmul.emitError(msg);
        failed = true;
      };

      // ---- enum fields must be non-UNKNOWN --------------------------------
      // (reduction is exempt: it must be UNKNOWN for non-K splits — see below)
      if (p.getSplitAxis()    == pim::SplitAxis::UNKNOWN)
        err("pim.matmul plan has UNKNOWN split_axis");
      if (p.getReusePolicy()  == pim::ReusePolicy::UNKNOWN)
        err("pim.matmul plan has UNKNOWN reuse_policy");
      if (p.getKernelVariant()== pim::KernelVariant::UNKNOWN)
        err("pim.matmul plan has UNKNOWN kernel_variant");
      if (p.getPackFormat()   == pim::PackFormat::UNKNOWN)
        err("pim.matmul plan has UNKNOWN pack_format");
      if (p.getAccumType()    == pim::AccumType::UNKNOWN)
        err("pim.matmul plan has UNKNOWN accum_type");
      if (p.getWritebackMode()== pim::WritebackMode::UNKNOWN)
        err("pim.matmul plan has UNKNOWN writeback_mode");

      // ---- integer fields must be positive --------------------------------
      if (p.getTasklets()  <= 0) err("pim.matmul plan: tasklets must be > 0");
      if (p.getActiveDpus()<= 0) err("pim.matmul plan: active_dpus must be > 0");
      if (p.getAlignment() <= 0) err("pim.matmul plan: alignment must be > 0");
      if (p.getTileM() <= 0 || p.getTileN() <= 0 || p.getTileK() <= 0)
        err("pim.matmul plan: tile_m/n/k must all be > 0");

      // ---- consistency checks --------------------------------------------
      if (p.getGroupM() > 0 && p.getKernelVariant() != pim::KernelVariant::GROUPED)
        err("pim.matmul plan: group_m > 0 but kernel_variant is not GROUPED");

      // K-split requires a concrete reduction; non-K splits must leave it UNKNOWN.
      if (p.getSplitAxis() == pim::SplitAxis::K &&
          p.getReduction() == pim::ReductionStrategy::UNKNOWN)
        err("pim.matmul plan: split_axis=K requires a non-UNKNOWN reduction");
      if (p.getSplitAxis() != pim::SplitAxis::K &&
          p.getSplitAxis() != pim::SplitAxis::UNKNOWN &&
          p.getReduction() != pim::ReductionStrategy::UNKNOWN)
        err("pim.matmul plan: reduction must be UNKNOWN for non-K split axis");
    });

    if (failed)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::triton::createPimLayoutVerifyPass() {
  return std::make_unique<PimLayoutVerifyPass>();
}
