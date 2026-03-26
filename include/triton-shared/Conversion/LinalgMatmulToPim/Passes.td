//===----------------------------------------------------------------------===//
// Passes for lowering Triton-origin linalg.matmul to pim.matmul.
//===----------------------------------------------------------------------===//

#ifndef LINALG_MATMUL_TO_PIM_PASSES
#define LINALG_MATMUL_TO_PIM_PASSES

include "mlir/Pass/PassBase.td"

def LinalgAnnotateTileMeta : Pass<"linalg-annotate-tile-meta", "mlir::func::FuncOp"> {
  let summary = "Attach triton.tile_meta to Triton-origin linalg.matmul ops";
  let description = [{
    Walks linalg.matmul ops inside a function and attaches a
    'triton.tile_meta' dictionary attribute to every matmul that carries
    'triton.dot_origin'.  The dictionary encodes the static tile dimensions
    read from the operand types:
      bm – rows of the A tile
      bn – columns of the B tile
      bk – shared (K) dimension of the tile
    Ops without 'triton.dot_origin' are skipped silently.
  }];
  let constructor = "mlir::triton::createLinalgAnnotateTileMetaPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def LinalgMatmulToPimCandidate : Pass<"linalg-matmul-to-pim-candidate",
                                      "mlir::func::FuncOp"> {
  let summary = "Replace a Triton-origin linalg.matmul with a partial pim.matmul";
  let description = [{
    Finds the unique linalg.matmul carrying 'triton.dot_origin' in a function,
    traces its operands back to the full-matrix function arguments, and inserts
    a pim.matmul at the function entry block with a partial execution plan.
    The plan fields tile_m, tile_n, tile_k are filled from 'triton.tile_meta';
    all other plan fields are UNKNOWN and must be populated by
    PimPlanMaterializePass before PimLayoutVerifyPass is run.

    The original linalg.matmul is erased; dead code (subviews, loops) is
    cleaned up by a subsequent canonicalization pass.

    Functions that contain a number of Triton-origin matmuls other than 1 are
    skipped without modification.
  }];
  let constructor = "mlir::triton::createLinalgMatmulToPimCandidatePass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::pim::PIMDialect"
  ];
}

def PimPlanMaterialize : Pass<"pim-plan-materialize", "mlir::func::FuncOp"> {
  let summary = "Fill in UNKNOWN fields of pim.matmul execution plans";
  let description = [{
    Walks every pim.matmul in a function and replaces any field that is still
    UNKNOWN (or zero for integer fields) with a concrete value derived from:
      - element types of the A / C memref operands (→ kernel_variant, accum_type,
        pack_format)
      - tile sizes already stored in the plan (tile_m, tile_n, tile_k)
      - static shapes of the C memref operand (→ active_dpus)
      - fixed UPMEM hardware defaults (tasklets = 16, alignment = 8,
        MAX_DPUS = 2560)

    Fields that already carry non-UNKNOWN / non-zero values are left unchanged,
    so the pass is idempotent and can be used to override individual fields
    before running it on a partially-filled plan.

    Default decisions:
      split_axis     = N   (distribute output columns across DPUs)
      reuse_policy   = REUSE_A (when split along N, A is reused)
      reduction      = HOST_REDUCE
      writeback_mode = DIRECT
      kernel_variant = GROUPED if group_m > 0, otherwise FLAT
      pack_format    = INT8 for i8 element types, NONE otherwise
      accum_type     = INT32 for integer outputs, FLOAT32 otherwise
  }];
  let constructor = "mlir::triton::createPimPlanMaterializePass()";
  let dependentDialects = [
    "mlir::pim::PIMDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def PimLayoutVerify : Pass<"pim-layout-verify", "mlir::func::FuncOp"> {
  let summary = "Verify pim.matmul execution plans are complete and consistent";
  let description = [{
    Walks every pim.matmul in a function and checks that:
      - no enum field is UNKNOWN
      - integer fields (tasklets, active_dpus, alignment, tile sizes) are > 0
      - group_m > 0 implies kernel_variant == GROUPED
      - split_axis == K implies a non-UNKNOWN reduction strategy

    Emits an error diagnostic and signals pass failure for every violated
    constraint.  Run this pass after PimPlanMaterialize to catch incomplete
    or inconsistent plans before lowering to runtime calls.
  }];
  let constructor = "mlir::triton::createPimLayoutVerifyPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

#endif // LINALG_MATMUL_TO_PIM_PASSES
