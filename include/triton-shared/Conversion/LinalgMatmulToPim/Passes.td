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

#endif // LINALG_MATMUL_TO_PIM_PASSES
