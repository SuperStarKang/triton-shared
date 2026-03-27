#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Enum definitions (stringify / symbolize helpers).
#include "triton-shared/Dialect/PIM/IR/PIMEnums.cpp.inc"

using namespace mlir;
using namespace mlir::pim;

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void PIMDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton-shared/Dialect/PIM/IR/PIMAttrs.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/PIM/IR/PIMOps.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// ExecutionPlanAttr – custom assembly format
//
// Text form:
//   #pim.execution_plan<tile_m = 64, tile_n = 64, tile_k = 512,
//                       split_axis = M, reuse_policy = stream_B,
//                       reduction = unknown, tasklets = 16,
//                       active_dpus = 1024, kernel_variant = flat,
//                       pack_format = none, accum_type = int32,
//                       writeback_mode = direct, alignment = 8,
//                       group_m = 0, batch_count = 1>
//===----------------------------------------------------------------------===//

void ExecutionPlanAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "tile_m = " << getTileM();
  p << ", tile_n = " << getTileN();
  p << ", tile_k = " << getTileK();
  p << ", split_axis = " << stringifySplitAxis(getSplitAxis());
  p << ", reuse_policy = " << stringifyReusePolicy(getReusePolicy());
  p << ", reduction = " << stringifyReductionStrategy(getReduction());
  p << ", tasklets = " << getTasklets();
  p << ", active_dpus = " << getActiveDpus();
  p << ", kernel_variant = " << stringifyKernelVariant(getKernelVariant());
  p << ", pack_format = " << stringifyPackFormat(getPackFormat());
  p << ", accum_type = " << stringifyAccumType(getAccumType());
  p << ", writeback_mode = " << stringifyWritebackMode(getWritebackMode());
  p << ", alignment = " << getAlignment();
  p << ", group_m = " << getGroupM();
  p << ", batch_count = " << getBatchCount();
  p << ">";
}

Attribute ExecutionPlanAttr::parse(AsmParser &parser, Type /*odsType*/) {
  int64_t tile_m = 0, tile_n = 0, tile_k = 0;
  int32_t tasklets = 0, active_dpus = 0, alignment = 0, group_m = 0,
          batch_count = 0;
  SplitAxis split_axis = SplitAxis::UNKNOWN;
  ReusePolicy reuse_policy = ReusePolicy::UNKNOWN;
  ReductionStrategy reduction = ReductionStrategy::UNKNOWN;
  KernelVariant kernel_variant = KernelVariant::UNKNOWN;
  PackFormat pack_format = PackFormat::UNKNOWN;
  AccumType accum_type = AccumType::UNKNOWN;
  WritebackMode writeback_mode = WritebackMode::UNKNOWN;

  // parseKeyword requires StringRef* in this MLIR version.
  auto parseKeyValue = [&]() -> ParseResult {
    StringRef key;
    if (parser.parseKeyword(&key))
      return failure();
    if (parser.parseEqual())
      return failure();

    // Integer fields
    if (key == "tile_m")      return parser.parseInteger(tile_m);
    if (key == "tile_n")      return parser.parseInteger(tile_n);
    if (key == "tile_k")      return parser.parseInteger(tile_k);
    if (key == "tasklets")    return parser.parseInteger(tasklets);
    if (key == "active_dpus") return parser.parseInteger(active_dpus);
    if (key == "alignment")   return parser.parseInteger(alignment);
    if (key == "group_m")     return parser.parseInteger(group_m);
    if (key == "batch_count") return parser.parseInteger(batch_count);

    // Enum fields – value is a bare keyword.
    StringRef val;
    if (parser.parseKeyword(&val))
      return failure();

    if (key == "split_axis") {
      auto opt = symbolizeSplitAxis(val);
      if (!opt)
        return parser.emitError(parser.getNameLoc(),
                                "unknown split_axis value: ") << val;
      split_axis = *opt;
    } else if (key == "reuse_policy") {
      auto opt = symbolizeReusePolicy(val);
      if (!opt)
        return parser.emitError(parser.getNameLoc(),
                                "unknown reuse_policy value: ") << val;
      reuse_policy = *opt;
    } else if (key == "reduction") {
      auto opt = symbolizeReductionStrategy(val);
      if (!opt)
        return parser.emitError(parser.getNameLoc(),
                                "unknown reduction value: ") << val;
      reduction = *opt;
    } else if (key == "kernel_variant") {
      auto opt = symbolizeKernelVariant(val);
      if (!opt)
        return parser.emitError(parser.getNameLoc(),
                                "unknown kernel_variant value: ") << val;
      kernel_variant = *opt;
    } else if (key == "pack_format") {
      auto opt = symbolizePackFormat(val);
      if (!opt)
        return parser.emitError(parser.getNameLoc(),
                                "unknown pack_format value: ") << val;
      pack_format = *opt;
    } else if (key == "accum_type") {
      auto opt = symbolizeAccumType(val);
      if (!opt)
        return parser.emitError(parser.getNameLoc(),
                                "unknown accum_type value: ") << val;
      accum_type = *opt;
    } else if (key == "writeback_mode") {
      auto opt = symbolizeWritebackMode(val);
      if (!opt)
        return parser.emitError(parser.getNameLoc(),
                                "unknown writeback_mode value: ") << val;
      writeback_mode = *opt;
    } else {
      return parser.emitError(parser.getNameLoc(),
                              "unknown key in #pim.execution_plan: ") << key;
    }
    return success();
  };

  // Parse <key = value, ...> manually; Delimiter::Less not available here.
  if (parser.parseLess())
    return {};
  if (parser.parseCommaSeparatedList(parseKeyValue))
    return {};
  if (parser.parseGreater())
    return {};

  return ExecutionPlanAttr::get(
      parser.getContext(), tile_m, tile_n, tile_k, split_axis, reuse_policy,
      reduction, tasklets, active_dpus, kernel_variant, pack_format,
      accum_type, writeback_mode, alignment, group_m, batch_count);
}

//===----------------------------------------------------------------------===//
// MatmulOp – verifier
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  // Rank checks only apply to ranked memrefs; unranked memrefs are accepted
  // as-is (rank is dynamic and will be validated at runtime).
  auto aRanked = dyn_cast<MemRefType>(getA().getType());
  auto bRanked = dyn_cast<MemRefType>(getB().getType());
  auto cRanked = dyn_cast<MemRefType>(getC().getType());

  if (aRanked && aRanked.getRank() != 2)
    return emitOpError("operand 'a' must be a rank-2 memref");
  if (bRanked && bRanked.getRank() != 2)
    return emitOpError("operand 'b' must be a rank-2 memref");
  if (cRanked && cRanked.getRank() != 2)
    return emitOpError("operand 'c' must be a rank-2 memref");

  // Element type check: only possible when both sides are ranked.
  auto getElem = [](Type t) -> Type {
    if (auto mr = dyn_cast<MemRefType>(t)) return mr.getElementType();
    if (auto umr = dyn_cast<UnrankedMemRefType>(t)) return umr.getElementType();
    return {};
  };
  Type aElem = getElem(getA().getType());
  Type bElem = getElem(getB().getType());
  if (aElem && bElem && aElem != bElem)
    return emitOpError("operands 'a' and 'b' must have the same element type");

  auto plan = getPlan();

  // K-split requires a non-UNKNOWN reduction strategy.
  if (plan.getSplitAxis() == SplitAxis::K &&
      plan.getReduction() == ReductionStrategy::UNKNOWN)
    return emitOpError(
        "K-split requires a non-UNKNOWN reduction strategy in the plan");

  // Non-K-split with a concrete reduction strategy is suspicious.
  if (plan.getSplitAxis() != SplitAxis::K &&
      plan.getSplitAxis() != SplitAxis::UNKNOWN &&
      plan.getReduction() != ReductionStrategy::UNKNOWN)
    return emitOpError(
        "reduction strategy should be UNKNOWN for non-K split axis");

  // group_m > 0 implies GROUPED kernel variant (or UNKNOWN if not yet set).
  if (plan.getGroupM() > 0 &&
      plan.getKernelVariant() != KernelVariant::GROUPED &&
      plan.getKernelVariant() != KernelVariant::UNKNOWN)
    return emitOpError(
        "group_m > 0 requires kernel_variant = grouped (or unknown)");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMOps.cpp.inc"

#include "triton-shared/Dialect/PIM/IR/PIMDialect.cpp.inc"
