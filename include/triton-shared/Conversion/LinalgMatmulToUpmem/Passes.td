//===----------------------------------------------------------------------===//
//
// UPMEM PIM backend metadata extraction pass for Triton-Shared.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_MATMUL_TO_UPMEM_PASSES
#define LINALG_MATMUL_TO_UPMEM_PASSES

include "mlir/Pass/PassBase.td"

def LinalgMatmulToUpmem : Pass<"linalg-matmul-to-upmem", "mlir::ModuleOp"> {
  let summary = "Annotate linalg.matmul with UPMEM PIM execution metadata";
  let description = [{
    Walks every func.func in the module and, for each function that contains a
    linalg.matmul, extracts the following information and attaches it as
    function-level attributes under the "upmem.*" namespace:

      upmem.bm         – tile rows    (from A-tensor shape[0])
      upmem.bk         – tile k-dim   (from A-tensor shape[1])
      upmem.bn         – tile cols    (from B-tensor shape[1])
      upmem.a_ptr_idx  – index of the A-pointer function argument
      upmem.b_ptr_idx  – index of the B-pointer function argument
      upmem.c_ptr_idx  – index of the C-pointer function argument
      upmem.m_idx      – index of the M-size   function argument
      upmem.n_idx      – index of the N-size   function argument
      upmem.k_idx      – index of the K-size   function argument
      upmem.elem_type  – element type string  (e.g. "i8", "f32")
      upmem.acc_type   – accumulator type string (e.g. "i32", "f32")

    The pass is purely additive: it does NOT modify any op, so the standard
    CPU lowering pipeline continues to work unchanged.  The Python runtime
    reads the upmem.* attributes to configure the DPU launch.

    Argument identification relies on NameLoc debug information that Triton
    attaches to each function block argument (e.g. loc("a_ptr")).  If the
    names cannot be resolved the pass simply skips annotation for that function.
  }];
  let constructor = "mlir::triton::createLinalgMatmulToUpmemPass()";
}

#endif // LINALG_MATMUL_TO_UPMEM_PASSES
