"""UPMEM DPU hardware specification and derived performance helpers."""
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PIMHardwareSpec:
    """Immutable compile-time constants for one UPMEM DIMM rank."""
    max_dpus: int          = 2560   # standard rank size
    max_tasklets: int      = 24     # hardware limit per DPU
    default_tasklets: int  = 16     # safe default (leaves headroom)
    mram_size_mb: int      = 64     # MRAM per DPU (bytes: 64 × 1 MiB)
    wram_size_kb: int      = 64     # WRAM (working memory) per DPU
    iram_size_kb: int      = 24     # IRAM (instruction cache) per DPU
    transfer_align_bytes: int = 8   # DMA transfer alignment requirement
    dpu_freq_mhz: int      = 350    # approximate DPU clock (model only)

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    def max_active_dpus(self, tile_m: int, tile_n: int,
                        m: int, n: int) -> int:
        """Return ceil(M/tm) × ceil(N/tn) capped to max_dpus."""
        dpus = math.ceil(m / tile_m) * math.ceil(n / tile_n)
        return min(dpus, self.max_dpus)

    def wave_count(self, tile_m: int, tile_n: int,
                   m: int, n: int, active_dpus: int) -> int:
        """Number of sequential DPU waves for (m, n, tile_m, tile_n)."""
        total = math.ceil(m / tile_m) * math.ceil(n / tile_n)
        return math.ceil(total / max(1, active_dpus))

    def fits_in_wram(self, tile_m: int, tile_n: int,
                     tile_k: int, elem_bytes: int = 1) -> bool:
        """True if one tile of C fits in WRAM (conservative)."""
        c_bytes = tile_m * tile_n * 4  # accum always int32/fp32
        ab_bytes = (tile_m + tile_n) * tile_k * elem_bytes
        return (c_bytes + ab_bytes) <= self.wram_size_kb * 1024


# Singleton for the reference UPMEM hardware.
UPMEM_PIM = PIMHardwareSpec()


@dataclass
class PIMConfig:
    """A concrete PIM execution configuration (mirrors ExecutionPlanAttr)."""
    tile_m: int
    tile_n: int
    tile_k: int
    active_dpus: int
    tasklets: int = 16
    split_axis: int = 2    # SplitAxis::N
    pack_format: int = 1   # PackFormat::NONE
    group_m: int = 0

    # ------------------------------------------------------------------ #
    def score(self, m: int, n: int, k: int,
              spec: PIMHardwareSpec = UPMEM_PIM) -> float:
        """Simple utilisation-based score; higher is better."""
        used = min(
            math.ceil(m / self.tile_m) * math.ceil(n / self.tile_n),
            self.active_dpus,
            spec.max_dpus,
        )
        util = used / spec.max_dpus
        waves = spec.wave_count(self.tile_m, self.tile_n, m, n, used)
        return util / max(1, waves)

    def env_overrides(self) -> dict:
        """Return env-var overrides consumed by the PIM launcher stub."""
        return {"TRITON_PIM_ACTIVE_DPUS": str(self.active_dpus)}


class PIMConfigSpace:
    """Generate candidate PIMConfigs for a given problem size."""

    def __init__(self, spec: PIMHardwareSpec = UPMEM_PIM):
        self.spec = spec

    def candidates(
        self,
        m: int, n: int, k: int,
        tile_m: int, tile_n: int, tile_k: int,
    ) -> List[PIMConfig]:
        """Return configs varying active_dpus from max down to 256."""
        max_dpus = self.spec.max_active_dpus(tile_m, tile_n, m, n)
        steps: List[int] = []
        d = max_dpus
        while d >= 256:
            steps.append(d)
            d //= 2
        if not steps:
            steps = [max_dpus]
        return [
            PIMConfig(
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                active_dpus=d,
                tasklets=self.spec.default_tasklets,
            )
            for d in steps
        ]


class PIMPerformanceModel:
    """Rank PIMConfigs by estimated efficiency without running benchmarks."""

    def __init__(self, spec: PIMHardwareSpec = UPMEM_PIM):
        self.spec = spec

    def best(
        self,
        configs: List[PIMConfig],
        m: int, n: int, k: int,
    ) -> PIMConfig:
        """Return the highest-scoring config."""
        return max(configs, key=lambda c: c.score(m, n, k, self.spec))

    def rank(
        self,
        configs: List[PIMConfig],
        m: int, n: int, k: int,
    ) -> List[Tuple[float, PIMConfig]]:
        """Return (score, config) pairs sorted descending."""
        scored = [(c.score(m, n, k, self.spec), c) for c in configs]
        return sorted(scored, key=lambda x: x[0], reverse=True)
