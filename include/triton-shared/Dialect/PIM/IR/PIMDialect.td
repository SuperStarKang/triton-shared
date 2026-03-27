#ifndef PIM_DIALECT
#define PIM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PIM Dialect
//===----------------------------------------------------------------------===//

def PIM_Dialect : Dialect {
  let name = "pim";
  let cppNamespace = "::mlir::pim";
  let summary = "Dialect for PIM (Processing-In-Memory) operations";
  let description = [{
    The PIM dialect represents matrix operations that are offloaded to UPMEM
    DPU hardware. It carries the complete execution plan (tile sizes, DPU
    parallelism, data layout) as an attribute so that lowering passes have
    all the information needed to emit a single runtime dispatch call.
  }];
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Enums
// genSpecializedAttr = 0 → plain C++ enum + stringify/symbolize helpers only;
// no attribute wrapper is generated. The enum types are used directly as
// parameter types inside PIMExecutionPlanAttr.
//===----------------------------------------------------------------------===//

def PIM_SplitAxis : I32EnumAttr<"SplitAxis",
    "Dimension used to split work across DPUs", [
  I32EnumAttrCase<"UNKNOWN", 0, "unknown">,
  I32EnumAttrCase<"M",       1, "M">,
  I32EnumAttrCase<"N",       2, "N">,
  I32EnumAttrCase<"K",       3, "K">
]> {
  let cppNamespace = "::mlir::pim";
  let genSpecializedAttr = 0;
}

def PIM_ReusePolicy : I32EnumAttr<"ReusePolicy",
    "Data reuse policy for DPU-side operands", [
  I32EnumAttrCase<"UNKNOWN",  0, "unknown">,
  I32EnumAttrCase<"STREAM_A", 1, "stream_A">,
  I32EnumAttrCase<"STREAM_B", 2, "stream_B">,
  I32EnumAttrCase<"REUSE_A",  3, "reuse_A">,
  I32EnumAttrCase<"REUSE_B",  4, "reuse_B">
]> {
  let cppNamespace = "::mlir::pim";
  let genSpecializedAttr = 0;
}

def PIM_ReductionStrategy : I32EnumAttr<"ReductionStrategy",
    "How partial K-split results are reduced to produce the final output", [
  I32EnumAttrCase<"UNKNOWN",       0, "unknown">,
  I32EnumAttrCase<"HOST_REDUCE",   1, "host_reduce">,
  I32EnumAttrCase<"DPU_REDUCE",    2, "dpu_reduce">,
  I32EnumAttrCase<"ATOMIC_REDUCE", 3, "atomic_reduce">
]> {
  let cppNamespace = "::mlir::pim";
  let genSpecializedAttr = 0;
}

def PIM_KernelVariant : I32EnumAttr<"KernelVariant",
    "DPU binary variant to dispatch", [
  I32EnumAttrCase<"UNKNOWN", 0, "unknown">,
  I32EnumAttrCase<"FLAT",    1, "flat">,
  I32EnumAttrCase<"GROUPED", 2, "grouped">,
  I32EnumAttrCase<"TILED",   3, "tiled">
]> {
  let cppNamespace = "::mlir::pim";
  let genSpecializedAttr = 0;
}

def PIM_PackFormat : I32EnumAttr<"PackFormat",
    "Data packing format used during DPU transfer", [
  I32EnumAttrCase<"UNKNOWN", 0, "unknown">,
  I32EnumAttrCase<"NONE",    1, "none">,
  I32EnumAttrCase<"INT8",    2, "int8">,
  I32EnumAttrCase<"INT4",    3, "int4">
]> {
  let cppNamespace = "::mlir::pim";
  let genSpecializedAttr = 0;
}

def PIM_AccumType : I32EnumAttr<"AccumType",
    "Accumulator data type used on the DPU side", [
  I32EnumAttrCase<"UNKNOWN", 0, "unknown">,
  I32EnumAttrCase<"INT32",   1, "int32">,
  I32EnumAttrCase<"FLOAT32", 2, "float32">
]> {
  let cppNamespace = "::mlir::pim";
  let genSpecializedAttr = 0;
}

def PIM_WritebackMode : I32EnumAttr<"WritebackMode",
    "How DPU results are written back to host memory", [
  I32EnumAttrCase<"UNKNOWN",   0, "unknown">,
  I32EnumAttrCase<"DIRECT",    1, "direct">,
  I32EnumAttrCase<"SCATTERED", 2, "scattered">
]> {
  let cppNamespace = "::mlir::pim";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Attribute Base
//===----------------------------------------------------------------------===//

class PIM_Attr<string name, string _mnemonic, list<Trait> traits = []>
    : AttrDef<PIM_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// PIMExecutionPlanAttr
//===----------------------------------------------------------------------===//

def PIM_ExecutionPlanAttr : PIM_Attr<"ExecutionPlan", "execution_plan"> {
  let summary = "Complete execution plan for a PIM matmul dispatch";
  let description = [{
    Encodes all decisions needed to lower a `pim.matmul` to a single call to
    `triton_pim_kernel_launch`: tile sizes, DPU count, kernel binary variant,
    data packing, accumulator type, and memory layout.

    Fields set to 0 (UNKNOWN for enum fields, 0 for integer fields) are
    placeholders produced by LinalgMatmulToPimCandidatePass.  They must be
    filled in by PimPlanMaterializePass before PimLayoutVerifyPass runs.

    Text format:
      #pim.execution_plan<tile_m = 64, tile_n = 64, tile_k = 512,
                          split_axis = M, reuse_policy = stream_B,
                          reduction = unknown, tasklets = 16,
                          active_dpus = 1024, kernel_variant = flat,
                          pack_format = none, accum_type = int32,
                          writeback_mode = direct, alignment = 8,
                          group_m = 0, batch_count = 1>
  }];

  let parameters = (ins
    "int64_t":$tile_m,
    "int64_t":$tile_n,
    "int64_t":$tile_k,
    "::mlir::pim::SplitAxis":$split_axis,
    "::mlir::pim::ReusePolicy":$reuse_policy,
    "::mlir::pim::ReductionStrategy":$reduction,
    "int32_t":$tasklets,
    "int32_t":$active_dpus,
    "::mlir::pim::KernelVariant":$kernel_variant,
    "::mlir::pim::PackFormat":$pack_format,
    "::mlir::pim::AccumType":$accum_type,
    "::mlir::pim::WritebackMode":$writeback_mode,
    "int32_t":$alignment,
    "int32_t":$group_m,
    "int32_t":$batch_count
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Op Base
//===----------------------------------------------------------------------===//

class PIM_Op<string mnemonic, list<Trait> traits = []>
    : Op<PIM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// PIM_MatmulOp
//===----------------------------------------------------------------------===//

def PIM_MatmulOp : PIM_Op<"matmul", [
    MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
  ]> {
  let summary = "Full-problem matrix multiplication targeting UPMEM DPU hardware";
  let description = [{
    Represents a complete M×K · K×N → M×N matrix multiplication to be
    dispatched to the UPMEM DPU runtime.  The operands `a`, `b`, `c` are
    the *full* matrices (not tiles); `m_size`, `n_size`, `k_size` carry the
    dynamic problem dimensions as SSA values.  The `plan` attribute encodes
    every dispatch decision.

    The DPU runtime tiles and distributes work internally, so this op always
    lowers to a *single* call to `triton_pim_kernel_launch`.

    Example:
    ```mlir
    pim.matmul %a, %b, %c, %m, %n, %k
        {plan = #pim.execution_plan<tile_m = 64, tile_n = 64, tile_k = 512,
                                    split_axis = M, reuse_policy = stream_B,
                                    reduction = unknown, tasklets = 16,
                                    active_dpus = 1024, kernel_variant = flat,
                                    pack_format = none, accum_type = int32,
                                    writeback_mode = direct, alignment = 8,
                                    group_m = 0, batch_count = 1>}
        : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi32>, index, index, index
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyUnrankedMemRef]>:$a,
    AnyTypeOf<[AnyMemRef, AnyUnrankedMemRef]>:$b,
    AnyTypeOf<[AnyMemRef, AnyUnrankedMemRef]>:$c,
    Index:$m_size,
    Index:$n_size,
    Index:$k_size,
    PIM_ExecutionPlanAttr:$plan
  );

  let results = (outs);

  let assemblyFormat = [{
    $a `,` $b `,` $c `,` $m_size `,` $n_size `,` $k_size
    attr-dict `:` type($a) `,` type($b) `,` type($c) `,`
    type($m_size) `,` type($n_size) `,` type($k_size)
  }];

  let hasVerifier = 1;
}

#endif // PIM_DIALECT
