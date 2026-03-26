#ifndef TRITON_CONVERSION_LINALGMATMULTOPIM_H
#define TRITON_CONVERSION_LINALGMATMULTOPIM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

/// Attaches 'triton.tile_meta' {bm, bn, bk} to every linalg.matmul that
/// carries 'triton.dot_origin', reading the tile sizes from the operand types.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgAnnotateTileMetaPass();

/// Finds the unique Triton-origin linalg.matmul per function, traces its
/// operands to the full-matrix function arguments, and inserts a pim.matmul
/// with a partial execution plan at the function entry.  The linalg.matmul is
/// erased; remaining dead code is expected to be removed by canonicalize.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgMatmulToPimCandidatePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_LINALGMATMULTOPIM_H
