#ifndef TRITON_CONVERSION_PIMTORUNTIMECALLS_H
#define TRITON_CONVERSION_PIMTORUNTIMECALLS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

/// Lower every pim.matmul in the module to a call to the external C function
/// `triton_pim_matmul` (declared in triton-shared/runtime/pim_runtime.h).
/// The execution plan attribute is fully expanded into individual i64/i32
/// arguments; memref operands are converted to i64 pointer values via
/// memref.extract_aligned_pointer_as_index + arith.index_castui.
std::unique_ptr<OperationPass<ModuleOp>> createPimToRuntimeCallsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_PIMTORUNTIMECALLS_H
