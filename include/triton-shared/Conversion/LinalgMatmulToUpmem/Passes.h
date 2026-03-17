//===----------------------------------------------------------------------===//
//
// UPMEM PIM backend metadata extraction pass for Triton-Shared.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_LINALGMATMULTOUPMEM_PASSES_H
#define TRITON_CONVERSION_LINALGMATMULTOUPMEM_PASSES_H

#include "triton-shared/Conversion/LinalgMatmulToUpmem/LinalgMatmulToUpmem.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/LinalgMatmulToUpmem/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_LINALGMATMULTOUPMEM_PASSES_H
