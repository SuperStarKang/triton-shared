"""
PIMNativeAutotuner — Triton native Autotuner subclass for PIM hardware.

This is Path 2: user code decorated with @triton.autotune.
When the triton_shared backend is active (TRITON_USE_PIM=1), triton.autotune
is replaced by a factory that creates PIMNativeAutotuner instead of Autotuner.

What PIMNativeAutotuner adds on top of Autotuner:
  __init__  : automatically expands configs with ACTIVE_DPUS dimension,
              and forces cache_results=True for disk persistence.
  _bench()  : injects ACTIVE_DPUS as an environment variable before each
              benchmark call so that driver._launch_pim() can read it
              without needing a sweep.

Everything else (key computation, memory cache, disk cache via
check_disk_cache(), execution via self.fn.run()) is inherited unchanged
from triton.runtime.autotuner.Autotuner.
"""

import os

from triton.runtime.autotuner import Autotuner

from .pim_config import expand_with_active_dpus


class PIMNativeAutotuner(Autotuner):
    """
    Drop-in replacement for triton.Autotuner with PIM awareness.

    Inherits all behaviour from Autotuner; the only overrides are:

      __init__  — adds ACTIVE_DPUS to every config that lacks it, and
                  enables cache_results so results survive across processes.
      _bench()  — sets TRITON_PIM_ACTIVE_DPUS in the environment before
                  calling the parent's _bench(), then clears it afterwards.
                  This lets _launch_pim() pick up the correct DPU count
                  without running a separate sweep.
    """

    def __init__(self, fn, arg_names, configs, key,
                 reset_to_zero=None, restore_value=None, **kwargs):
        # Expand configs: add ACTIVE_DPUS variants for any config that
        # does not already declare it.
        expanded = expand_with_active_dpus(configs)

        # Force disk caching so the best config survives process restarts,
        # matching the behaviour of inductor's CachingAutotuner.
        kwargs["cache_results"] = True

        super().__init__(
            fn, arg_names, expanded, key,
            reset_to_zero=reset_to_zero,
            restore_value=restore_value,
            **kwargs,
        )

    def _bench(self, *args, config, **meta):
        """
        Wrap the parent's _bench() to inject ACTIVE_DPUS for the PIM driver.

        Flow:
          _bench()
            └─ os.environ["TRITON_PIM_ACTIVE_DPUS"] = str(active_dpus)
            └─ super()._bench()          (parent times the call)
                 └─ kernel_call()
                      └─ self.fn.run()
                           └─ _launch_pim()  reads TRITON_PIM_ACTIVE_DPUS
                                             skips internal sweep
        """
        active_dpus = config.kwargs.get("ACTIVE_DPUS")
        if active_dpus is not None:
            os.environ["TRITON_PIM_ACTIVE_DPUS"] = str(active_dpus)
        try:
            return super()._bench(*args, config=config, **meta)
        finally:
            if active_dpus is not None:
                os.environ.pop("TRITON_PIM_ACTIVE_DPUS", None)


# ---------------------------------------------------------------------------
# Patch factory — replaces triton.autotune when triton_shared is active
# ---------------------------------------------------------------------------

def _make_pim_autotune_decorator():
    """
    Return a drop-in replacement for triton.autotune that creates
    PIMNativeAutotuner instances when TRITON_USE_PIM is enabled.
    Falls back to the original triton.autotune otherwise.
    """
    import triton
    _orig = triton.autotune

    def pim_autotune(configs, key, reset_to_zero=None, restore_value=None,
                     pre_hook=None, post_hook=None, prune_configs_by=None,
                     warmup=None, rep=None, use_cuda_graph=False,
                     do_bench=None, cache_results=False):
        if os.environ.get("TRITON_USE_PIM", "").lower() in ("1", "true", "on", "yes", "y"):
            def decorator(fn):
                return PIMNativeAutotuner(
                    fn, fn.arg_names, configs, key,
                    reset_to_zero=reset_to_zero,
                    restore_value=restore_value,
                    pre_hook=pre_hook,
                    post_hook=post_hook,
                    prune_configs_by=prune_configs_by,
                    warmup=warmup,
                    rep=rep,
                    use_cuda_graph=use_cuda_graph,
                    do_bench=do_bench,
                )
            return decorator
        return _orig(configs, key,
                     reset_to_zero=reset_to_zero,
                     restore_value=restore_value,
                     pre_hook=pre_hook, post_hook=post_hook,
                     prune_configs_by=prune_configs_by,
                     warmup=warmup, rep=rep,
                     use_cuda_graph=use_cuda_graph,
                     do_bench=do_bench,
                     cache_results=cache_results)

    return pim_autotune


def patch_triton_autotune():
    """
    Replace triton.autotune with the PIM-aware version.
    Called once when the triton_shared driver is loaded.
    Idempotent — safe to call multiple times.
    """
    import triton
    if getattr(triton.autotune, "_pim_patched", False):
        return
    patched = _make_pim_autotune_decorator()
    patched._pim_patched = True
    triton.autotune = patched
