import hashlib
import tempfile
import sysconfig
import threading

import os, subprocess, tempfile, platform
import importlib.util
import sys

# Thread-local storage for PIM timing reported by pim_runtime.
# _launch_pim() writes stats.total_ms here after each launch so that
# CPUDriver.get_benchmarker() can return accurate PIM-internal timings
# instead of Python wall-clock measurements.
_pim_last_timing = threading.local()

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)

def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")
    
    return sanitizer_type

def _sanitizer_available(sanitizer_type):
    if "LD_PRELOAD" not in os.environ:
        return False
    if f"libclang_rt.{sanitizer_type}.so" not in os.environ["LD_PRELOAD"]:
        return False
    
    return True

def _use_pim():
    return os.getenv("TRITON_USE_PIM", "").lower() in ("1", "true", "on", "yes", "y")

def _use_pim_mlir() -> bool:
    return _use_pim() and os.getenv("TRITON_PIM_MLIR", "").lower() in ("1", "true", "on", "yes", "y")

def _get_ptr(obj):
    if isinstance(obj, int):
        return obj
    if obj is None:
        return 0
    if hasattr(obj, "data_ptr"):
        return int(obj.data_ptr())
    return int(obj)


def _ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator

def _launch_pim(pim_meta, args, grid_m=0, grid_n=0):
    # Lazy import to avoid loading DPU runtime when not needed.
    from . import pim_runtime
    from . import pim_autotune

    if not pim_meta:
        raise RuntimeError("PIM launch requested but pim_meta is missing")

    a_idx = pim_meta.get("a_ptr")
    b_idx = pim_meta.get("b_ptr")
    c_idx = pim_meta.get("c_ptr")
    m_idx = pim_meta.get("m_arg")
    n_idx = pim_meta.get("n_arg")
    k_idx = pim_meta.get("k_arg")
    m_val = pim_meta.get("m_val")
    n_val = pim_meta.get("n_val")
    k_val = pim_meta.get("k_val")

    if a_idx is None or b_idx is None or c_idx is None:
        raise RuntimeError("Missing PIM pointer indices in metadata")
    a_ptr = _get_ptr(args[a_idx])
    b_ptr = _get_ptr(args[b_idx])
    c_ptr = _get_ptr(args[c_idx])

    if m_idx is not None and n_idx is not None and k_idx is not None:
        m = int(args[m_idx])
        n = int(args[n_idx])
        k = int(args[k_idx])
    elif m_val is not None and n_val is not None and k_val is not None:
        m = int(m_val)
        n = int(n_val)
        k = int(k_val)
    else:
        raise RuntimeError("Missing PIM M/N/K metadata")

    block = pim_meta.get("block", None)
    if not block or len(block) != 3:
        raise RuntimeError("Missing PIM block metadata")
    bm, bk, bn = [int(x) for x in block]
    launch_kind = pim_meta.get("launch_kind", "grid2d")

    transb = 1 if pim_meta.get("transb", False) else 0
    schedule = pim_meta.get("schedule", "global_tile_static")
    if schedule == "global_tile_static":
        schedule_policy = 1
    else:
        raise RuntimeError(f"Unsupported PIM schedule policy: {schedule}")
    ndpu = int(os.getenv("TRITON_PIM_NDPU", "1"))
    dpu_binary = os.getenv(
        "TRITON_PIM_DPU_BINARY",
        "/home/dlrkdals/PGEMMlib/PGEMMLib_With_AutoTuner/dpu/gemm_dpu_triton",
    )

    effective_grid_m = grid_m
    effective_grid_n = grid_n
    if launch_kind == "flattened_grouped_mm":
        effective_grid_m = _ceil_div(m, bm)
        effective_grid_n = _ceil_div(n, bn)
    elif effective_grid_m == 0 or effective_grid_n == 0:
        effective_grid_m = _ceil_div(m, bm)
        effective_grid_n = _ceil_div(n, bn)

    # Determine active_dpus — three priority levels:
    #   1. pim_meta["active_dpus"]: set when ACTIVE_DPUS is a kernel constexpr
    #      and the compiler embeds it in metadata (future: Path 1 constexpr route).
    #   2. TRITON_PIM_ACTIVE_DPUS env var: set by CachingAutotuner.bench() (Path 1)
    #      or PIMNativeAutotuner._bench() (Path 2) before each benchmark call.
    #   3. Fallback to legacy sweep via pim_autotune (backward compatibility).
    active_dpus = pim_meta.get("active_dpus")

    if active_dpus is None:
        env_val = os.environ.get("TRITON_PIM_ACTIVE_DPUS")
        if env_val is not None:
            active_dpus = int(env_val)

    if active_dpus is None:
        # Legacy path: run sweep and cache result (kept for backward compat)
        active_dpus = pim_autotune.get_best_active_dpus(
            m, n, k, bm, bk, bn, ndpu, transb, schedule_policy, dpu_binary,
            a_ptr, b_ptr, c_ptr, effective_grid_m, effective_grid_n,
        )

    stats = pim_runtime.pim_launch(
        a_ptr, b_ptr, c_ptr,
        m, k, n,
        bm, bk, bn,
        ndpu, transb, schedule_policy, dpu_binary,
        grid_m=effective_grid_m, grid_n=effective_grid_n,
        forced_active_dpus=active_dpus,
    )

    # Store PIM-internal timing so get_benchmarker() can return accurate results.
    if stats is not None and hasattr(stats, "total_ms"):
        _pim_last_timing.ms = stats.total_ms

    return None

# -------------------- PIM-MLIR runtime stub ---------------
def _pim_runtime_stub() -> str:
    """Return C++ source implementing triton_pim_matmul via dlopen/dlsym.

    This stub is appended to the launcher source when TRITON_USE_PIM=1 so that
    the MLIR-lowered kernel object (which calls triton_pim_matmul directly) can
    resolve the symbol at link time.  The actual GEMM work is delegated to
    triton_pim_kernel_launch inside libPGEMM.so, loaded lazily at first call.
    """
    default_lib = "/home/dlrkdals/Triton/pgemm/lib/libPGEMM.so"
    default_dpu = "/home/dlrkdals/PGEMMlib/PGEMMLib_With_AutoTuner/dpu/gemm_dpu_triton"
    return f"""
// ---- triton_pim_matmul stub (generated by driver.py) ----
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int (*_pim_kl_fn_t)(
    void*, void*, void*,
    uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    const char*, uint32_t, void*
);

static _pim_kl_fn_t _pim_kl_fn = nullptr;

static _pim_kl_fn_t _get_pim_kl() {{
    if (_pim_kl_fn) return _pim_kl_fn;
    const char* lib = getenv("TRITON_PIM_LIB");
    if (!lib) lib = "{default_lib}";
    void* h = dlopen(lib, RTLD_LAZY | RTLD_GLOBAL);
    if (!h) return nullptr;
    _pim_kl_fn = (_pim_kl_fn_t)dlsym(h, "triton_pim_kernel_launch");
    return _pim_kl_fn;
}}

extern "C" void triton_pim_matmul(
    int64_t a, int64_t b, int64_t c,
    int64_t m, int64_t n, int64_t k,
    int64_t tile_m, int64_t tile_n, int64_t tile_k,
    int32_t split_axis, int32_t reuse_policy, int32_t reduction,
    int32_t tasklets, int32_t active_dpus,
    int32_t kernel_variant, int32_t pack_format, int32_t accum_type, int32_t writeback_mode,
    int32_t alignment, int32_t group_m, int32_t batch_count
) {{
    _pim_kl_fn_t fn = _get_pim_kl();
    if (!fn) return;
    const char* dpu = getenv("TRITON_PIM_DPU_BINARY");
    if (!dpu) dpu = "{default_dpu}";
    uint32_t grid_m = (uint32_t)((m + tile_m - 1) / tile_m);
    uint32_t grid_n = (uint32_t)((n + tile_n - 1) / tile_n);
    fn((void*)a, (void*)b, (void*)c,
       (uint32_t)m, (uint32_t)k, (uint32_t)n,
       (uint32_t)tile_m, (uint32_t)tile_k, (uint32_t)tile_n,
       grid_m, grid_n,
       1u, 0u, 1u,
       dpu, (uint32_t)active_dpus, nullptr);
}}
// ---- end triton_pim_matmul stub ----
"""

# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        # Proper support for bfloat16 and float16 is not yet handled.
        # https://github.com/microsoft/triton-shared/issues/348
        # "fp16": "TODO",
        # "bf16": "TODO",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]

def _extracted_type(ty):
    if ty[0] == '*':
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)

def _format_of(ty):
    return {
      "PyObject*": "O",
      "constexpr": "O",
      "float": "f",
      "double": "d",
      "long": "l",
      "int8_t": "b",
      "int16_t": "h",
      "int32_t": "i",
      "int64_t": "l",
      "uint8_t": "B",
      "uint16_t": "H",
      "uint32_t": "I",
      "uint64_t": "K",
    }[ty]

def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = ''.join([_format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    kernel_arg_decls = ', '.join(_ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*" for i, ty in signature.items() if ty != "constexpr")
    kernel_arg_decls += ', ' if kernel_arg_decls else ''

    kernel_parameters = ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}" for i, ty in signature.items() if ty != "constexpr")
    kernel_parameters += ', ' if kernel_parameters else ''

    pim_stub = _pim_runtime_stub() if _use_pim() else ""
    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"
{pim_stub}

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({kernel_arg_decls}
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    // Cast "function" to the real function type.
    // apply parallelization to the triton grid when using ThreadSanitizer (TSan) 
    // to help detect potential data races across program instances during kernel execution
    {"#pragma omp parallel for collapse(3)" if _get_sanitizer_type() == "tsan" else ""}
    for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          // Use some random type "char" here.
          {' '.join(f'StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};' for i, ty in signature.items() if i not in constants and ty[0] == "*")}
          {kernel_name}({kernel_parameters}
                        gridX, gridY, gridZ, x, y, z);
        }}
      }}
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, 'include')
        py_lib_dir = os.path.join(sys.base_prefix, 'libs')
        py_lib = '{name}{major}{minor}.lib'.format(name="python", major=py_version.major, minor=py_version.minor)
    else:
        py_include_dir = os.path.join(sys.base_prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')
        py_lib_dir = os.path.join(sys.base_prefix, 'lib')
        py_lib = '{name}{major}.{minor}'.format(name="python", major=py_version.major, minor=py_version.minor)
    cpu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(cpu_backend_path, "include")

    def launch(
        gridX, gridY, gridZ, stream, cu_function,
        kernel_metadata, launch_metadata,
        launch_enter_hook, launch_exit_hook, *args):
        pim_meta = kernel_metadata[7] if len(kernel_metadata) > 7 else None
        pim_mlir_dispatch = kernel_metadata[8] if len(kernel_metadata) > 8 else False
        if _use_pim() and pim_mlir_dispatch:
            # MLIR-lowered path: kernel calls triton_pim_matmul internally.
            # Override grid to (1,1,1) — pim.matmul handles the full matrix.
            gridX = gridY = gridZ = 1
        elif _use_pim() and pim_meta:
            return _launch_pim(pim_meta, args, grid_m=gridX, grid_n=gridY)
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile one kernel every time.
        # The cu_function parameter actually contains our kernel obj.
        # See CPUUtils.load_binary method.
        kernel_obj = cu_function
        kernel_name = kernel_metadata[6] # see pack_metadata in compiler.py
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)

        key = hashlib.sha256(src.encode("utf-8") + kernel_obj).hexdigest()
        cache = get_cache_manager(key)
        name = "__triton_shared_ref_cpu_kernel_launcher"

        if platform.system() == "Windows":
          filename = f"{name}.pyd"
        else:
          filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
          with tempfile.TemporaryDirectory() as tmpdir:
              sanitizer_type = _get_sanitizer_type()

              if platform.system() == "Windows":
                  if sanitizer_type != "":
                      raise Exception("Sanitizers are not supported on Windows with triton-shared.")

                  obj_path = os.path.join(tmpdir, "kernel.obj")
                  launcher_src_path = os.path.join(tmpdir, "main.cxx")
                  so_path = os.path.join(tmpdir, "kernel.pyd")
                  Path(obj_path).write_bytes(kernel_obj)
                  Path(launcher_src_path).write_text(src)
                  # Compile it together.
                  subprocess.check_call([
                    "cl", "/LD", "/std:c++17", launcher_src_path, obj_path,
                    f"-I{py_include_dir}", f"-I{include_dir}", "/link", f"/LIBPATH:{py_lib_dir}",
                    "/link", f"{py_lib}", f"/OUT:{so_path}"
                  ])
              else:
                  obj_path = os.path.join(tmpdir, "kernel.o")
                  launcher_src_path = os.path.join(tmpdir, "main.cxx")
                  so_path = os.path.join(tmpdir, "kernel.so")
                  Path(obj_path).write_bytes(kernel_obj)
                  Path(launcher_src_path).write_text(src)

                  # Compile it together.
                  if sanitizer_type != "":
                      clang_path = _get_llvm_bin_path("clang++")

                      subprocess_args = [
                          clang_path, "-std=c++17", launcher_src_path, obj_path,
                          f"-I{py_include_dir}", f"-I{include_dir}", f"-L{py_lib_dir}",
                          "-shared", f"-l{py_lib}", "-fPIC", "-o", so_path
                      ]

                      if not _sanitizer_available(sanitizer_type):
                          raise Exception(f"Use LD_PRELOAD=\"path/to/libclang_rt.{sanitizer_type}.so\" TRITON_SHARED_SANITIZER_TYPE={sanitizer_type} python ...")

                      if sanitizer_type == "asan":
                          subprocess_args.extend(["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"])
                      elif sanitizer_type == "tsan":
                          # ensure that openmp is available
                          libomp_path = next(Path(Path(_get_llvm_bin_path("")).parent).rglob("libomp.so"), None)

                          if not libomp_path:
                              raise Exception(f"libomp.so does not exist.")

                          libomp_path = str(libomp_path.parent)

                          subprocess_args.extend(["-g", "-fsanitize=thread", "-fopenmp", f"-Wl,-rpath,{libomp_path}"])
                      
                      subprocess.check_call(subprocess_args)
                  else:
                      extra_libs = ["-ldl"] if _use_pim() else []
                      subprocess.check_call([
                        "g++", "-std=c++17", launcher_src_path, obj_path,
                        f"-I{py_include_dir}", f"-I{include_dir}", f"-L{py_lib_dir}",
                        "-shared", f"-l{py_lib}", "-fPIC", "-o", so_path,
                        *extra_libs,
                      ])

              with open(so_path, "rb") as f:
                cache_path = cache.put(f.read(), filename, binary=True)

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch(gridX, gridY, gridZ,
                          kernel_metadata, launch_metadata,
                          launch_enter_hook, launch_exit_hook,
                          *args)

    return launch


class CPULauncher(object):

    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(launcher_src, kernel_placeholder_name)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)



class CPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
          "max_shared_mem": 2 ** 20,
          "multiprocessor_count": None,
          "sm_clock_rate": None,
          "mem_clock_rate": None,
          "mem_bus_width": None
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # obj of the kernel so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
          None,       # module
          kernel_obj, # function
          None,       # n_regs
          None,        # n_spills
          sys.maxsize, # n_max_threads
        )


class CPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "obj"
        # Patch triton.autotune → PIMNativeAutotuner when PIM is enabled.
        # Idempotent: safe to call multiple times.
        if _use_pim():
            from .pim_autotuner import patch_triton_autotune
            patch_triton_autotune()

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        if not _use_pim():
            from triton.testing import do_bench
            return do_bench

        # PIM-aware benchmarker: uses pim_runtime's internal timing
        # (stored in _pim_last_timing by _launch_pim) rather than Python
        # wall-clock, eliminating Python call-overhead from measurements.
        def _pim_bench(kernel_call, quantiles=(0.5, 0.2, 0.8)):
            _WARMUP = 1
            _REPEAT = 3

            for _ in range(_WARMUP):
                kernel_call()

            times = []
            for _ in range(_REPEAT):
                kernel_call()
                ms = getattr(_pim_last_timing, "ms", None)
                times.append(ms if ms is not None else float("inf"))

            times.sort()
            n = len(times)
            # Return (median, low, high) matching triton do_bench quantile format
            return [times[n // 2], times[0], times[-1]]

        return _pim_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
    
    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
  
