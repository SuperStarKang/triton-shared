#ifndef TRITON_CONVERSION_PIMTORUNTIMECALLS_PASSES_H
#define TRITON_CONVERSION_PIMTORUNTIMECALLS_PASSES_H

#include "triton-shared/Conversion/PimToRuntimeCalls/PimToRuntimeCalls.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/PimToRuntimeCalls/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_PIMTORUNTIMECALLS_PASSES_H
