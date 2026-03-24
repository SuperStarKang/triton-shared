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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <regex>
#include <string>

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

static std::optional<std::string> getFilePathFromLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc.getFilename().str();
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return getFilePathFromLoc(nameLoc.getChildLoc());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location inner : fusedLoc.getLocations())
      if (auto path = getFilePathFromLoc(inner))
        return path;
  }
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    if (auto path = getFilePathFromLoc(callLoc.getCaller()))
      return path;
    return getFilePathFromLoc(callLoc.getCallee());
  }
  return std::nullopt;
}

static std::optional<int> parseStaticAssignment(StringRef source,
                                                StringRef symbol) {
  std::regex pattern("(^|\\n)\\s*" + symbol.str() +
                     "\\s*(?::\\s*tl\\.constexpr)?\\s*=\\s*(\\d+)\\s*($|\\n)");
  std::smatch match;
  std::string sourceStr = source.str();
  if (!std::regex_search(sourceStr, match, pattern) || match.size() < 3)
    return std::nullopt;
  return std::stoi(match[2].str());
}

struct StaticProblemSizes {
  std::optional<int> m;
  std::optional<int> n;
  std::optional<int> k;
};

enum class LaunchKind {
  Unknown,
  Grid2D,
  FlattenedGroupedMM,
};

struct SourceKernelMetadata {
  StaticProblemSizes problemSizes;
  std::optional<int> groupM;
  LaunchKind launchKind = LaunchKind::Unknown;
};

static std::optional<std::string> readSourceFromFuncLoc(func::FuncOp func) {
  auto path = getFilePathFromLoc(func.getLoc());
  if (!path)
    return std::nullopt;

  auto buffer = llvm::MemoryBuffer::getFile(*path);
  if (!buffer) {
    LLVM_DEBUG(llvm::dbgs()
               << "[linalg-matmul-to-upmem] could not read source file '"
               << *path << "'\n");
    return std::nullopt;
  }

  return buffer->get()->getBuffer().str();
}

static LaunchKind detectLaunchKind(StringRef source) {
  const bool isFlattenedGroupedMM =
      source.contains("pid = tl.program_id(0)") &&
      source.contains("width = GROUP_M * grid_n") &&
      source.contains("pid_m = group_id * GROUP_M + (pid % group_size)") &&
      source.contains("pid_n = (pid % width) // (group_size)");
  if (isFlattenedGroupedMM)
    return LaunchKind::FlattenedGroupedMM;

  const bool isGrid2D = source.contains("tl.program_id(axis=0)") &&
                        source.contains("tl.program_id(axis=1)");
  if (isGrid2D)
    return LaunchKind::Grid2D;

  return LaunchKind::Unknown;
}

static StringRef launchKindToAttrValue(LaunchKind launchKind) {
  switch (launchKind) {
  case LaunchKind::FlattenedGroupedMM:
    return "flattened_grouped_mm";
  case LaunchKind::Grid2D:
    return "grid2d";
  case LaunchKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

static SourceKernelMetadata parseSourceKernelMetadata(func::FuncOp func) {
  auto sourceOpt = readSourceFromFuncLoc(func);
  if (!sourceOpt)
    return {};

  StringRef source(*sourceOpt);
  return {
      {/*problemSizes=*/
       parseStaticAssignment(source, "M"),
       parseStaticAssignment(source, "N"),
       parseStaticAssignment(source, "K")},
      /*groupM=*/parseStaticAssignment(source, "GROUP_M"),
      /*launchKind=*/detectLaunchKind(source),
  };
}

static bool matchesAnyAlias(StringRef name,
                            std::initializer_list<StringRef> aliases) {
  return llvm::is_contained(aliases, name);
}

static std::optional<int> resolveNamedArgIndex(
    func::FuncOp func, std::initializer_list<StringRef> aliases) {
  for (auto [idx, arg] : llvm::enumerate(func.getBody().front().getArguments())) {
    StringRef name = getArgName(arg);
    if (!name.empty() && matchesAnyAlias(name, aliases))
      return static_cast<int>(idx);
  }
  return std::nullopt;
}

static void collectBlockArgIndicesFromValue(
    Value value, llvm::SmallDenseSet<int> &result,
    llvm::SmallDenseSet<Value> &visitedValues,
    llvm::SmallDenseSet<Value> &visitedMemrefs);

static void collectBlockArgIndicesFromMemrefCopies(
    Value memref, llvm::SmallDenseSet<int> &result,
    llvm::SmallDenseSet<Value> &visitedValues,
    llvm::SmallDenseSet<Value> &visitedMemrefs) {
  if (!visitedMemrefs.insert(memref).second)
    return;

  for (Operation *user : memref.getUsers()) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      if (copyOp.getTarget() == memref)
        collectBlockArgIndicesFromValue(copyOp.getSource(), result, visitedValues,
                                        visitedMemrefs);
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(user)) {
      collectBlockArgIndicesFromMemrefCopies(castOp.getResult(), result,
                                             visitedValues, visitedMemrefs);
      continue;
    }
    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(user)) {
      collectBlockArgIndicesFromMemrefCopies(reinterpretOp.getResult(), result,
                                             visitedValues, visitedMemrefs);
      continue;
    }
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
      collectBlockArgIndicesFromMemrefCopies(subviewOp.getResult(), result,
                                             visitedValues, visitedMemrefs);
      continue;
    }
  }
}

static void collectBlockArgIndicesFromValue(
    Value value, llvm::SmallDenseSet<int> &result,
    llvm::SmallDenseSet<Value> &visitedValues,
    llvm::SmallDenseSet<Value> &visitedMemrefs) {
  if (!visitedValues.insert(value).second)
    return;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    result.insert(blockArg.getArgNumber());
    return;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return;

  if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(defOp)) {
    Value memref = toTensorOp.getBuffer();
    collectBlockArgIndicesFromValue(memref, result, visitedValues, visitedMemrefs);
    collectBlockArgIndicesFromMemrefCopies(memref, result, visitedValues,
                                           visitedMemrefs);
    return;
  }
  if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
    collectBlockArgIndicesFromValue(castOp.getSource(), result, visitedValues,
                                    visitedMemrefs);
    return;
  }
  if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
    collectBlockArgIndicesFromValue(reinterpretOp.getSource(), result,
                                    visitedValues, visitedMemrefs);
    return;
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    collectBlockArgIndicesFromValue(subviewOp.getSource(), result, visitedValues,
                                    visitedMemrefs);
    return;
  }
  if (auto tensorCastOp = dyn_cast<tensor::CastOp>(defOp)) {
    collectBlockArgIndicesFromValue(tensorCastOp.getSource(), result,
                                    visitedValues, visitedMemrefs);
    return;
  }
  if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
    collectBlockArgIndicesFromValue(extractSliceOp.getSource(), result,
                                    visitedValues, visitedMemrefs);
    return;
  }
  if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
    collectBlockArgIndicesFromValue(expandShapeOp.getSrc(), result, visitedValues,
                                    visitedMemrefs);
    return;
  }
  if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(defOp)) {
    collectBlockArgIndicesFromValue(collapseShapeOp.getSrc(), result,
                                    visitedValues, visitedMemrefs);
    return;
  }

  for (Value operand : defOp->getOperands())
    collectBlockArgIndicesFromValue(operand, result, visitedValues, visitedMemrefs);
}

static std::optional<int> resolveOperandArgIndex(Value value) {
  llvm::SmallDenseSet<int> result;
  llvm::SmallDenseSet<Value> visitedValues;
  llvm::SmallDenseSet<Value> visitedMemrefs;
  collectBlockArgIndicesFromValue(value, result, visitedValues, visitedMemrefs);
  if (result.size() != 1)
    return std::nullopt;
  return *result.begin();
}

static bool valueDependsOnOp(Value value, Operation *target,
                             llvm::SmallDenseSet<Value> &visitedValues) {
  if (!visitedValues.insert(value).second)
    return false;

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  if (defOp == target)
    return true;

  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(value).getResultNumber();
    if (resultIdx < forOp.getRegionIterArgs().size()) {
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (valueDependsOnOp(yieldOp.getOperand(resultIdx), target, visitedValues))
        return true;
    }
  }

  for (Value operand : defOp->getOperands())
    if (valueDependsOnOp(operand, target, visitedValues))
      return true;
  return false;
}

static std::optional<int> resolveOutputArgIndex(func::FuncOp func,
                                                linalg::MatmulOp matmulOp) {
  std::optional<int> resolved;
  bool ambiguous = false;

  func.walk([&](bufferization::MaterializeInDestinationOp matOp) {
    llvm::SmallDenseSet<Value> visitedValues;
    if (!valueDependsOnOp(matOp.getSource(), matmulOp, visitedValues))
      return;

    if (auto idx = resolveOperandArgIndex(matOp.getDest())) {
      if (resolved && *resolved != *idx) {
        ambiguous = true;
        return;
      }
      resolved = idx;
    }
  });

  if (ambiguous)
    return std::nullopt;
  return resolved;
}

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------
class LinalgMatmulToUpmemPass
    : public LinalgMatmulToUpmemBase<LinalgMatmulToUpmemPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

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
      auto aIdx =
          resolveNamedArgIndex(func, {"a_ptr", "arg_A", "A"});
      if (!aIdx)
        aIdx = resolveOperandArgIndex(matmulOp.getInputs()[0]);

      auto bIdx =
          resolveNamedArgIndex(func, {"b_ptr", "arg_B", "B"});
      if (!bIdx)
        bIdx = resolveOperandArgIndex(matmulOp.getInputs()[1]);

      auto cIdx =
          resolveNamedArgIndex(func, {"c_ptr", "out_ptr0", "C"});
      if (!cIdx)
        cIdx = resolveOutputArgIndex(func, matmulOp);

      auto mIdx = resolveNamedArgIndex(func, {"m", "M"});
      auto nIdx = resolveNamedArgIndex(func, {"n", "N"});
      auto kIdx = resolveNamedArgIndex(func, {"k", "K"});
      auto sourceMetadata = parseSourceKernelMetadata(func);
      auto staticSizes = sourceMetadata.problemSizes;

      if (!aIdx || !bIdx || !cIdx ||
          (!mIdx && !staticSizes.m) ||
          (!nIdx && !staticSizes.n) ||
          (!kIdx && !staticSizes.k)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[linalg-matmul-to-upmem] could not resolve all "
                      "pointer/size metadata – skipping '"
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
                    IntegerAttr::get(IntegerType::get(ctx, 32), *aIdx));
      func->setAttr("upmem.b_ptr_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), *bIdx));
      func->setAttr("upmem.c_ptr_idx",
                    IntegerAttr::get(IntegerType::get(ctx, 32), *cIdx));

      // M / N / K argument indices.
      if (mIdx)
        func->setAttr("upmem.m_idx",
                      IntegerAttr::get(IntegerType::get(ctx, 32), *mIdx));
      else
        func->setAttr("upmem.m_val",
                      IntegerAttr::get(IntegerType::get(ctx, 32), *staticSizes.m));
      if (nIdx)
        func->setAttr("upmem.n_idx",
                      IntegerAttr::get(IntegerType::get(ctx, 32), *nIdx));
      else
        func->setAttr("upmem.n_val",
                      IntegerAttr::get(IntegerType::get(ctx, 32), *staticSizes.n));
      if (kIdx)
        func->setAttr("upmem.k_idx",
                      IntegerAttr::get(IntegerType::get(ctx, 32), *kIdx));
      else
        func->setAttr("upmem.k_val",
                      IntegerAttr::get(IntegerType::get(ctx, 32), *staticSizes.k));

      func->setAttr("upmem.launch_kind",
                    StringAttr::get(ctx, launchKindToAttrValue(sourceMetadata.launchKind)));
      if (sourceMetadata.groupM)
        func->setAttr("upmem.group_m",
                      IntegerAttr::get(IntegerType::get(ctx, 32),
                                       *sourceMetadata.groupM));

      // Type strings (used by the runtime to choose the DPU kernel variant).
      func->setAttr("upmem.elem_type", StringAttr::get(ctx, elemTypeName));
      func->setAttr("upmem.acc_type",  StringAttr::get(ctx, accTypeName));

      LLVM_DEBUG(llvm::dbgs()
                 << "[linalg-matmul-to-upmem] annotated '"
                 << func.getName()
                 << "' bm=" << bm << " bk=" << bk << " bn=" << bn
                 << " a=" << *aIdx << " b=" << *bIdx << " c=" << *cIdx;
      if (mIdx)
        llvm::dbgs() << " m_idx=" << *mIdx;
      else
        llvm::dbgs() << " m_val=" << *staticSizes.m;
      if (nIdx)
        llvm::dbgs() << " n_idx=" << *nIdx;
      else
        llvm::dbgs() << " n_val=" << *staticSizes.n;
      if (kIdx)
        llvm::dbgs() << " k_idx=" << *kIdx;
      else
        llvm::dbgs() << " k_val=" << *staticSizes.k;
      llvm::dbgs()
                 << " launch_kind=" << launchKindToAttrValue(sourceMetadata.launchKind);
      if (sourceMetadata.groupM)
        llvm::dbgs() << " group_m=" << *sourceMetadata.groupM;
      llvm::dbgs()
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
