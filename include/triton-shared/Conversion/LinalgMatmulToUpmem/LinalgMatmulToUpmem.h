//===----------------------------------------------------------------------===//
//
// UPMEM PIM backend metadata extraction pass for Triton-Shared.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_LINALGMATMULTOUPMEM_H
#define TRITON_CONVERSION_LINALGMATMULTOUPMEM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

/// Creates the linalg-matmul-to-upmem pass.
///
/// The pass walks linalg.matmul ops inside func.func operations and attaches
/// "upmem.*" attributes to the enclosing function that encode the tile sizes
/// (bm, bk, bn), element types, and the indices of the pointer / size
/// arguments that the UPMEM PIM runtime needs to launch a DPU kernel.
std::unique_ptr<OperationPass<ModuleOp>> createLinalgMatmulToUpmemPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_LINALGMATMULTOUPMEM_H
