"""
Shared PIM config generation.
ACTIVE_DPUS is a first-class triton.Config kwarg alongside BLOCK_M/N/K.

Used by:
  - Path 1 (CachingAutotuner via triton_heuristics.py): get_configs(m, n, k, ndpu)
  - Path 2 (PIMNativeAutotuner via @triton.autotune):   expand_with_active_dpus(configs)
"""

import triton
from .pim_autotune import _load_disk_cache, _dpu_candidates

# Default block size candidates (BM, BK, BN) when no LUT data is available
_DEFAULT_BLOCK_SIZES = [
    (64,  64,  32),
    (128, 64,  32),
    (128, 128, 32),
    (128, 128, 64),
]

# Default ACTIVE_DPUS candidates used at decorator-time (M/N/K unknown).
# Matches _MIN_ACTIVE_DPUS floor defined in pim_autotune.
_DEFAULT_ACTIVE_DPUS = [256, 512, 1024, 2048]


def get_configs(m: int, n: int, k: int, ndpu: int = 1) -> list:
    """
    Return a triton.Config list covering the full (BM, BN, BK, ACTIVE_DPUS) search space.

    Called at runtime by Path 1 (CachingAutotuner) when M/N/K are known.

    Strategy:
      - If the pim_autotune LUT has sweep data for (M, N, K), use the top-3
        ACTIVE_DPUS candidates per block size to seed the search (smaller space).
      - Otherwise fall back to the full Cartesian product of _DEFAULT_BLOCK_SIZES
        × halving DPU candidates.
    """
    disk = _load_disk_cache(m, n, k)
    configs = []

    if disk:
        for tile_key, tile_data in disk.items():
            parts = tile_key.split(",")
            if len(parts) != 3:
                continue
            try:
                bm, bk, bn = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue

            # Keep top-3 ACTIVE_DPUS per tile size (sorted by measured total_ms)
            candidates = sorted(
                tile_data.get("candidates", []),
                key=lambda x: x["total_ms"],
            )[:3]
            for cand in candidates:
                if cand["active_dpus"] <= ndpu:
                    configs.append(triton.Config({
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "BLOCK_K": bk,
                        "ACTIVE_DPUS": cand["active_dpus"],
                    }))

    if not configs:
        # No LUT data → search over default candidates.
        # _dpu_candidates() already enforces _MIN_ACTIVE_DPUS as the floor.
        for bm, bk, bn in _DEFAULT_BLOCK_SIZES:
            total_tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
            for dpus in _dpu_candidates(ndpu, total_tiles):
                configs.append(triton.Config({
                    "BLOCK_M": bm,
                    "BLOCK_N": bn,
                    "BLOCK_K": bk,
                    "ACTIVE_DPUS": dpus,
                }))

    return configs


def expand_with_active_dpus(configs: list) -> list:
    """
    Expand user-provided configs by adding the ACTIVE_DPUS dimension.

    Called at decorator-time by Path 2 (PIMNativeAutotuner) when M/N/K are
    not yet known.  If a config already declares ACTIVE_DPUS it is kept
    unchanged; otherwise one variant is generated per _DEFAULT_ACTIVE_DPUS value.
    """
    result = []
    for cfg in configs:
        if "ACTIVE_DPUS" in cfg.kwargs:
            result.append(cfg)
        else:
            for dpus in _DEFAULT_ACTIVE_DPUS:
                result.append(triton.Config(
                    {**cfg.kwargs, "ACTIVE_DPUS": dpus},
                    num_warps=cfg.num_warps,
                    num_stages=cfg.num_stages,
                ))
    # Guard: never return an empty list
    return result if result else configs
