#ifndef TRITON_CONVERSION_LINALGMATMULTOPIM_PASSES_H
#define TRITON_CONVERSION_LINALGMATMULTOPIM_PASSES_H

#include "triton-shared/Conversion/LinalgMatmulToPim/LinalgMatmulToPim.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/LinalgMatmulToPim/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_LINALGMATMULTOPIM_PASSES_H
