"""
PIM autotune: automatically sweeps active_dpus for each (M, N, K, BM, BK, BN)
combination and caches the result.

Cache key  : (M, N, K)
Cache value: per (BM, BK, BN) tile config → best_active_dpus + all candidate records
Disk path  : $TRITON_CACHE_DIR/pim_autotune/M{m}_N{n}_K{k}.json

Flow inside _launch_pim():
  1. First call for (M,N,K,BM,BK,BN) → sweep runs, result cached
  2. Subsequent calls            → in-memory hit, no sweep
  3. New process, same sizes     → disk cache hit, no sweep
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# In-memory cache: (m, n, k, bm, bk, bn) -> best_active_dpus
# ---------------------------------------------------------------------------
_cache: dict = {}

# Per (m, n, k) -> {"bm", "bn", "bk", "active_dpus"} from profile table
_best_config_lut: dict = {}

_SWEEP_WARMUP = 1
_SWEEP_REPEAT = 3


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    base = os.getenv("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache"))
    d = Path(base) / "pim_autotune"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_file(m: int, n: int, k: int) -> Path:
    return _cache_dir() / f"M{m}_N{n}_K{k}.json"


def _load_disk_cache(m: int, n: int, k: int) -> dict:
    p = _cache_file(m, n, k)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_disk_cache(m: int, n: int, k: int, data: dict) -> None:
    p = _cache_file(m, n, k)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

_MIN_ACTIVE_DPUS = 256  # never sweep below this value

def _dpu_candidates(ndpu: int, total_tiles: int) -> list:
    """Generate active_dpus candidates by halving from max down to _MIN_ACTIVE_DPUS."""
    max_v = min(ndpu, total_tiles)
    candidates = []
    v = max_v
    while v >= _MIN_ACTIVE_DPUS:
        candidates.append(v)
        v //= 2
    # Always include _MIN_ACTIVE_DPUS itself as the floor candidate.
    if not candidates or candidates[-1] != _MIN_ACTIVE_DPUS:
        if _MIN_ACTIVE_DPUS <= max_v:
            candidates.append(_MIN_ACTIVE_DPUS)
    return candidates if candidates else [max(max_v, 1)]


def _sweep(m, n, k, bm, bk, bn, ndpu, transb, schedule_policy, dpu_binary,
           a_ptr, b_ptr, c_ptr, grid_m, grid_n):
    """Sweep all active_dpus candidates for a fixed (M,N,K,BM,BK,BN).

    Returns (best_active_dpus, candidate_records).
    """
    from . import pim_runtime

    total_tiles = grid_m * grid_n
    candidates = _dpu_candidates(ndpu, total_tiles)

    records = []
    for active_dpus in candidates:
        # warmup
        for _ in range(_SWEEP_WARMUP):
            pim_runtime.pim_launch(
                a_ptr, b_ptr, c_ptr, m, k, n, bm, bk, bn,
                ndpu, transb, schedule_policy, dpu_binary,
                grid_m=grid_m, grid_n=grid_n,
                forced_active_dpus=active_dpus,
            )

        # measure
        times = []
        last_stats = None
        for _ in range(_SWEEP_REPEAT):
            last_stats = pim_runtime.pim_launch(
                a_ptr, b_ptr, c_ptr, m, k, n, bm, bk, bn,
                ndpu, transb, schedule_policy, dpu_binary,
                grid_m=grid_m, grid_n=grid_n,
                forced_active_dpus=active_dpus,
            )
            times.append(last_stats.total_ms)

        times.sort()
        median_ms = times[len(times) // 2]

        records.append({
            "active_dpus":  active_dpus,
            "total_ms":     round(median_ms,               3),
            "pack_ms":      round(last_stats.pack_ms,      3),
            "h2d_ms":       round(last_stats.h2d_ms,       3),
            "compute_ms":   round(last_stats.compute_ms,   3),
            "d2h_ms":       round(last_stats.d2h_ms,       3),
            "scatter_ms":   round(last_stats.scatter_ms,   3),
            "tasks_per_dpu": last_stats.tasks_per_dpu,
            "waves":        last_stats.waves,
        })

    best = min(records, key=lambda r: r["total_ms"])
    return best["active_dpus"], records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_best_active_dpus(m, n, k, bm, bk, bn, ndpu, transb, schedule_policy,
                         dpu_binary, a_ptr, b_ptr, c_ptr, grid_m, grid_n) -> int:
    """Return the best active_dpus for the given config.

    Runs a sweep on first call; uses in-memory or disk cache on subsequent calls.
    """
    mem_key = (m, n, k, bm, bk, bn)
    if mem_key in _cache:
        return _cache[mem_key]

    # Check disk cache
    disk_data = _load_disk_cache(m, n, k)
    tile_key = f"{bm},{bk},{bn}"
    if tile_key in disk_data:
        best = disk_data[tile_key]["best_active_dpus"]
        _cache[mem_key] = best
        return best

    # Cache miss — run sweep
    print(
        f"[PIM autotune] sweep M={m} N={n} K={k} "
        f"BM={bm} BK={bk} BN={bn} ...",
        flush=True,
    )
    best_dpus, records = _sweep(
        m, n, k, bm, bk, bn, ndpu, transb, schedule_policy, dpu_binary,
        a_ptr, b_ptr, c_ptr, grid_m, grid_n,
    )
    best_ms = min(r["total_ms"] for r in records)
    print(
        f"[PIM autotune] best active_dpus={best_dpus}  ({best_ms:.1f} ms)",
        flush=True,
    )

    # Persist: merge this tile config into the (M,N,K) cache file
    disk_data[tile_key] = {"best_active_dpus": best_dpus, "candidates": records}
    _save_disk_cache(m, n, k, disk_data)

    _cache[mem_key] = best_dpus
    return best_dpus


def get_best_config_lut(m: int, n: int, k: int, ndpu: int):
    """Return the globally best (bm, bn, bk, active_dpus) from disk cache.

    Reads the per-(M,N,K) cache file produced by previous sweep runs and finds
    the (BM, BN, BK, active_dpus) combination with the lowest total_ms,
    subject to active_dpus <= ndpu.

    Also pre-populates _cache for every tile config found so that subsequent
    get_best_active_dpus() calls inside _launch_pim() are instant hits.

    Returns dict {bm, bn, bk, active_dpus, total_ms} or None if no cache yet.
    """
    lut_key = (m, n, k, ndpu)
    if lut_key in _best_config_lut:
        return _best_config_lut[lut_key]

    disk_data = _load_disk_cache(m, n, k)
    if not disk_data:
        return None

    best = None
    for tile_key, tile_data in disk_data.items():
        parts = tile_key.split(",")
        if len(parts) != 3:
            continue
        try:
            bm, bk, bn = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue

        candidates = tile_data.get("candidates", [])
        eligible = [c for c in candidates if c["active_dpus"] <= ndpu]
        if not eligible:
            continue

        best_cand = min(eligible, key=lambda c: c["total_ms"])

        # Pre-populate _cache so _launch_pim() skips the sweep entirely
        _cache[(m, n, k, bm, bk, bn)] = best_cand["active_dpus"]

        if best is None or best_cand["total_ms"] < best["total_ms"]:
            best = {
                "bm": bm, "bn": bn, "bk": bk,
                "active_dpus": best_cand["active_dpus"],
                "total_ms":    best_cand["total_ms"],
            }

    if best:
        _best_config_lut[lut_key] = best
        print(
            f"[PIM autotune] LUT  M={m} N={n} K={k} ndpu<={ndpu} → "
            f"BM={best['bm']} BN={best['bn']} BK={best['bk']} "
            f"active_dpus={best['active_dpus']}  ({best['total_ms']:.1f} ms)",
            flush=True,
        )

    return best
