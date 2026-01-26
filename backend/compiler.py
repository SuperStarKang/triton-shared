from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
import triton
from pathlib import Path

def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))

def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")
    
    return sanitizer_type

def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        _dump_ir_if_needed([src_path])
        triton_shared_opt_path = _get_triton_shared_opt_path()

        subprocess_args = [triton_shared_opt_path, src_path, "--triton-to-linalg-experimental", "--mlir-print-debuginfo", "-o", dst_path]

        if _get_sanitizer_type() != "":
            print("Building with sanitizer support...")

            # has to run before the other passes as operates on the tt dialect
            subprocess_args.insert(2, "--add-llvm-debug-info")

        subprocess.check_call(subprocess_args)
        return Path(dst_path).read_text()


def _use_pim_ir() -> bool:
    return os.getenv("TRITON_USE_PIM", "").lower() in ("1", "true", "on", "yes", "y")


# Match linalg.matmul (and triton_gpu.dot if it appears in the future) and capture
# ins/outs/return tensor payload so we can preserve operands.
_MATMUL_PATTERN = re.compile(
    r"(linalg\.matmul)\s+ins\((?P<ins>[^)]*)\)\s+outs\((?P<outs>[^)]*)\)\s*->\s*(?P<ret>tensor<[^>]+>)",
    re.DOTALL,
)

_PIM_ATTR_PATTERN = re.compile(
    r"pim\.matmul.*?attributes\s*\{(?P<attrs>[^}]+)\}",
    re.DOTALL,
)


def _extract_pim_meta(ttsharedir: str) -> dict:
    match = _PIM_ATTR_PATTERN.search(ttsharedir)
    if not match:
        return {}
    attrs = match.group("attrs")

    def _parse_int(key: str):
        m = re.search(rf"{key}\s*=\s*([0-9]+)", attrs)
        return int(m.group(1)) if m else None

    def _parse_bool(key: str):
        m = re.search(rf"{key}\s*=\s*(true|false)", attrs)
        if not m:
            return None
        return m.group(1) == "true"

    def _parse_list(key: str):
        m = re.search(rf"{key}\s*=\s*\[([^\]]+)\]", attrs)
        if not m:
            return None
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]

    def _parse_str(key: str):
        m = re.search(rf'{key}\s*=\s*"([^"]+)"', attrs)
        return m.group(1) if m else None

    meta = {}
    for key in ("pim.a_ptr", "pim.b_ptr", "pim.c_ptr",
                "pim.m_arg", "pim.n_arg", "pim.k_arg", "pim.batch_arg"):
        val = _parse_int(key)
        if val is not None:
            meta[key.split(".")[1]] = val
    block = _parse_list("pim.block")
    if block:
        meta["block"] = block
    transb = _parse_bool("pim.transb")
    if transb is not None:
        meta["transb"] = transb
    dtype = _parse_str("pim.dtype")
    if dtype:
        meta["dtype"] = dtype
    return meta


def _rewrite_matmul_to_pim(ttsharedir: str) -> str:
    """Replace matmul with a placeholder pim.matmul op in TTShared-IR."""

    def _find_arg_index(arg_name: str):
        pattern = rf'%arg(?P<idx>\d+):\s*[^)]*loc\("{arg_name}"'
        match = re.search(pattern, ttsharedir)
        if not match:
            return None
        return int(match.group("idx"))

    def _parse_shape(shape_str: str):
        tokens = shape_str.split("x")
        if len(tokens) < 2:
            return [], shape_str
        dtype = tokens[-1]
        dims = []
        for t in tokens[:-1]:
            try:
                dims.append(int(t))
            except ValueError:
                dims.append(t)
        return dims, dtype

    def _fmt_dims(dims):
        parts = []
        for d in dims:
            parts.append(str(d))
        return "[" + ", ".join(parts) + "]"

    arg_indices = {
        "a_ptr": _find_arg_index("a_ptr"),
        "b_ptr": _find_arg_index("b_ptr"),
        "c_ptr": _find_arg_index("c_ptr"),
        "b": _find_arg_index("b"),
        "m": _find_arg_index("m"),
        "n": _find_arg_index("n"),
        "k": _find_arg_index("k"),
    }

    def _replace(match: re.Match) -> str:
        ins = match.group("ins").strip()
        outs = match.group("outs").strip()
        ret = match.group("ret").strip()
        shapes = re.findall(r"tensor<([^>]+)>", match.group(0))
        a_dims = b_dims = a_dtype = b_dtype = None
        if len(shapes) >= 2:
            a_dims, a_dtype = _parse_shape(shapes[0])
            b_dims, b_dtype = _parse_shape(shapes[1])
        attr_parts = []
        if a_dims is not None:
            attr_parts.append(f"pim.a_shape = {_fmt_dims(a_dims)}")
            attr_parts.append(f'pim.a_dtype = "{a_dtype}"')
        if b_dims is not None:
            attr_parts.append(f"pim.b_shape = {_fmt_dims(b_dims)}")
            attr_parts.append(f'pim.b_dtype = "{b_dtype}"')
        if a_dims and b_dims and len(a_dims) == 2 and len(b_dims) == 2:
            block = [a_dims[0], a_dims[1], b_dims[1]]
            attr_parts.append(f"pim.block = {_fmt_dims(block)}")
            if a_dims[1] == b_dims[0]:
                attr_parts.append("pim.transb = false")
            elif a_dims[1] == b_dims[1]:
                attr_parts.append("pim.transb = true")
        if a_dtype and b_dtype and a_dtype == b_dtype:
            attr_parts.append(f'pim.dtype = "{a_dtype}"')
        if arg_indices["a_ptr"] is not None:
            attr_parts.append(f"pim.a_ptr = {arg_indices['a_ptr']}")
        if arg_indices["b_ptr"] is not None:
            attr_parts.append(f"pim.b_ptr = {arg_indices['b_ptr']}")
        if arg_indices["c_ptr"] is not None:
            attr_parts.append(f"pim.c_ptr = {arg_indices['c_ptr']}")
        if arg_indices["b"] is not None:
            attr_parts.append(f"pim.batch_arg = {arg_indices['b']}")
        if arg_indices["m"] is not None:
            attr_parts.append(f"pim.m_arg = {arg_indices['m']}")
        if arg_indices["n"] is not None:
            attr_parts.append(f"pim.n_arg = {arg_indices['n']}")
        if arg_indices["k"] is not None:
            attr_parts.append(f"pim.k_arg = {arg_indices['k']}")
        attr_parts.append(f'pim.ret = "{ret}"')
        attr_str = ""
        if attr_parts:
            attr_str = " attributes {" + ", ".join(attr_parts) + "}"
        return f"pim.matmul ins({ins}) outs({outs}) -> {ret}{attr_str}"

    return _MATMUL_PATTERN.sub(_replace, ttsharedir)


def _ttsharedir_to_pimir(ttsharedir: str, metadata: dict) -> str:
    # Placeholder for PIM-specific lowering from TritonShared-IR to PIM-IR.
    print("[triton-shared] Converting ttsharedir -> PIM IR (TRITON_USE_PIM set).")
    # Step 1: rewrite matmul to PIM placeholder
    ttsharedir_pim = _rewrite_matmul_to_pim(ttsharedir)
    metadata["pim_meta"] = _extract_pim_meta(ttsharedir_pim)
    # ttsharedir_pim can be dumped for debugging
    dump_dir = os.getenv("TRITON_SHARED_DUMP_PATH")
    if dump_dir:
        Path(os.path.join(dump_dir, "ttshared_pim.mlir")).write_text(ttsharedir_pim)
    return _ttsharedir_to_llir(ttsharedir)


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call([mlir_opt_path, ttshared_path,
            "--convert-linalg-to-affine-loops",
            # Note: eliminate-empty-tensors fails when there are multiple func.return ops
            # in a single kernel which are the results of early returns.
            # See python/examples/test_early_return.py for examples.
            # We disable this pass for now since performance on CPU isn't the main
            # focus at the moment.
            # "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine",
            "--convert-linalg-to-loops",
            "--expand-strided-metadata",
            "--convert-scf-to-cf",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-complex-to-llvm",
            "--convert-vector-to-llvm",
            "--convert-index-to-llvm",
            "--memref-expand",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--convert-cf-to-llvm",
            # Lowering memrefs creates more affine.apply ops.
            # Lowering these affine ops again creates further arith ops,
            # so we have to run these two passes again here.
            "--lower-affine",
            "--convert-arith-to-llvm",
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts",
            "--mlir-print-debuginfo",
            "-o",
            llmlir_path])

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path,
            "--mlir-to-llvmir",
            "-o",
            llir_path])
        _dump_ir_if_needed([ttshared_path, llmlir_path, llir_path])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)

        sanitizer_type = _get_sanitizer_type()

        if sanitizer_type != "":
            # using a sanitizer
            # invoke pass to append sanitizer attributes
            instrumented_src_path = os.path.join(tmpdir, "kernel-instrumented.ll")
        
            opt_path = _get_llvm_bin_path("opt")
            top_level_triton_path = os.path.dirname(triton.__file__)
            sanitizer_attributes_pass_path = str(next(Path(top_level_triton_path).rglob("libSanitizerAttributes.so"), None))

            if not sanitizer_attributes_pass_path:
                raise Exception(f"libSanitizerAttributes.so does not exist.")

            subprocess.check_call([opt_path, "-load-pass-plugin", sanitizer_attributes_pass_path, 
                "-passes=sanitizer-attributes", f"-sanitizer-type={sanitizer_type}", "-S", src_path, 
                "-o", instrumented_src_path])

            # compile to object file
            clang_path = _get_llvm_bin_path("clang++")

            subprocess_args = [clang_path, "-c", instrumented_src_path, "-o", dst_path]

            if sanitizer_type == "asan":
                subprocess_args.extend(["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"])
            elif sanitizer_type == "tsan":
                subprocess_args.extend(["-g", "-fsanitize=thread"])
                
            subprocess.check_call(subprocess_args)
        else:
            llc_path = _get_llvm_bin_path("llc")
            subprocess.check_call([llc_path, src_path, "-filetype=obj", "-relocation-model=pic", "-o", dst_path])
        
        return Path(dst_path).read_bytes()



@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = 'obj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        pim_meta = metadata.pim_meta if hasattr(metadata, "pim_meta") else None
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
            pim_meta,
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        return mod

    def add_stages(self, stages, options, language):
        print("[add stages]")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(_ttir_to_ttsharedir(src))
        if _use_pim_ir():
            print("[triton-shared] Using PIM IR lowering path.")
            stages["llir"] = lambda src, metadata: _optimize_llir(_ttsharedir_to_pimir(src, metadata))
        else:
            stages["llir"] = lambda src, metadata: _optimize_llir(_ttsharedir_to_llir(src))
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
