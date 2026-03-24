import ctypes
import os

_LIB = None


class PimCallStats(ctypes.Structure):
    _fields_ = [
        ("pack_ms",       ctypes.c_double),
        ("h2d_ms",        ctypes.c_double),
        ("compute_ms",    ctypes.c_double),
        ("d2h_ms",        ctypes.c_double),
        ("scatter_ms",    ctypes.c_double),
        ("total_ms",      ctypes.c_double),
        ("active_dpus",   ctypes.c_uint32),
        ("tasks_per_dpu", ctypes.c_uint32),
        ("waves",         ctypes.c_uint32),
    ]


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib_path = os.getenv(
        "TRITON_PIM_LIB",
        "/home/dlrkdals/Triton/pgemm/lib/libPGEMM.so",
    )
    _LIB = ctypes.CDLL(lib_path)
    _LIB.triton_pim_init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    _LIB.triton_pim_init.restype = ctypes.c_int
    _LIB.triton_pim_finalize.argtypes = []
    _LIB.triton_pim_finalize.restype = None
    _LIB.triton_pim_kernel_launch.argtypes = [
        ctypes.c_void_p,   # A
        ctypes.c_void_p,   # B
        ctypes.c_void_p,   # C
        ctypes.c_uint32,   # M
        ctypes.c_uint32,   # K
        ctypes.c_uint32,   # N
        ctypes.c_uint32,   # BM
        ctypes.c_uint32,   # BK
        ctypes.c_uint32,   # BN
        ctypes.c_uint32,   # grid_m (from Triton grid lambda; 0 = auto-compute)
        ctypes.c_uint32,   # grid_n (from Triton grid lambda; 0 = auto-compute)
        ctypes.c_uint32,   # nr_of_dpus
        ctypes.c_uint32,   # transb
        ctypes.c_uint32,   # schedule_policy
        ctypes.c_char_p,   # dpu_binary_path
        ctypes.c_uint32,   # forced_active_dpus (0 = use schedule model)
        ctypes.c_void_p,   # out_stats (PimCallStats*, NULL = skip)
    ]
    _LIB.triton_pim_kernel_launch.restype = ctypes.c_int
    return _LIB


def pim_init(dpu_binary_path, nr_of_dpus):
    lib = _load_lib()
    return lib.triton_pim_init(dpu_binary_path.encode(), nr_of_dpus)


def pim_finalize():
    lib = _load_lib()
    lib.triton_pim_finalize()


def pim_launch(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    k,
    n,
    bm,
    bk,
    bn,
    nr_of_dpus,
    transb,
    schedule_policy,
    dpu_binary_path,
    grid_m=0,
    grid_n=0,
    forced_active_dpus=0,
):
    lib = _load_lib()
    stats = PimCallStats()
    ret = lib.triton_pim_kernel_launch(
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(c_ptr),
        m,
        k,
        n,
        bm,
        bk,
        bn,
        grid_m,
        grid_n,
        nr_of_dpus,
        transb,
        schedule_policy,
        dpu_binary_path.encode(),
        forced_active_dpus,
        ctypes.byref(stats),
    )
    if ret != 0:
        raise RuntimeError(f"triton_pim_kernel_launch failed: {ret}")
    return stats
