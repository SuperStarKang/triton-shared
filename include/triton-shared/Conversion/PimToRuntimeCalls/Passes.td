//===----------------------------------------------------------------------===//
// Pass definition for PimToRuntimeCalls conversion.
//===----------------------------------------------------------------------===//

#ifndef PIM_TO_RUNTIME_CALLS_PASSES
#define PIM_TO_RUNTIME_CALLS_PASSES

include "mlir/Pass/PassBase.td"

def PimToRuntimeCalls : Pass<"pim-to-runtime-calls", "mlir::ModuleOp"> {
  let summary = "Lower pim.matmul to a triton_pim_matmul() runtime call";
  let description = [{
    Walks every pim.matmul op in the module and replaces it with a call to
    the external C function `triton_pim_matmul` declared in pim_runtime.h.

    The pass:
      1. Declares `triton_pim_matmul` as a private external func.func in the
         module (idempotent – skipped if already present).
      2. Extracts the aligned base pointer from each memref operand using
         memref.extract_aligned_pointer_as_index + arith.index_castui → i64.
      3. Casts the index-typed m/n/k size operands to i64.
      4. Materialises all plan fields (tile sizes, enum values, integers) as
         arith.constant i64 / i32 values.
      5. Emits a func.call @triton_pim_matmul(...).
      6. Erases the pim.matmul op.

    Prerequisites:
      - PimPlanMaterializePass must have run (no UNKNOWN fields).
      - PimLayoutVerifyPass should have confirmed plan consistency.
  }];
  let constructor = "mlir::triton::createPimToRuntimeCallsPass()";
  let dependentDialects = [
    "mlir::pim::PIMDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // PIM_TO_RUNTIME_CALLS_PASSES
