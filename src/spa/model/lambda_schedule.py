"""λ schedules over diffusion steps: shape library, matched-mean normalization, and the per-step hook.

Spec: dev ``docs/plan/57_lambda_schedule.md``. SPA's λ is a non-learned inference-time scalar held
**constant across all sampler steps** today. This module makes it a function of position along the
trajectory, so the question "does the adherence/designability trade-off survive a non-constant λ?" becomes
runnable. Nothing here changes default behaviour: the hook is opt-in, and the ``constant`` shape reproduces
today's numbers exactly.

WHY THE SHAPES ARE WHAT THEY ARE
Two measurements set the design, and they overturned the original expectation (dev ``plan/57`` §1):

  * ``results/28``: RFdiffusion3's trajectory does **not** settle topology early. Its first HALF is inert
    (TM to final at the random baseline, zero secondary structure), and the fold is decided between
    trajectory fraction **0.5 and 0.7**. Hence ``DECISION_WINDOW`` below, and hence a **mid-peaked bump**
    is the favoured candidate rather than the monotone decay first proposed.
  * ``results/27``: SPA's influence on the coordinate prediction peaks mid and is ~0 late, so a schedule
    that eases λ off late is removing something already near zero.

⭐ MATCHED MEAN λ IS THE WHOLE POINT OF THE NORMALIZATION
Every shape is rescaled so its mean over the *actual discrete step grid* equals the target λ. Without
that, "which shape wins" is confounded with "which shape delivers more total conditioning", and the sweep
answers nothing. With it, the comparison is a clean test of **distribution against total**. The realized
mean is exact by construction, not approximate, and ``tests/test_lambda_schedule.py`` asserts it.

NAMING NOTE. `plan/57` §3 calls the monotone family "ease-out quad/cubic/sine (Penner)". Penner's
easeOut* functions *rise* with deceleration; what a decaying λ needs is their mirror. The shapes here are
therefore named for what the **λ curve** does (``decay_quad`` and so on) rather than for the easing
primitive, because a reader checking the curve against the name should not have to invert it mentally.
"""

from __future__ import annotations

import math

import torch.nn as nn

# The window in which RFdiffusion3 actually decides the fold, measured in `results/28`:
# TM>0.5 at trajectory fraction ~0.51, TM>0.9 at ~0.68, lDDT>0.9 at ~0.71.
DECISION_WINDOW = (0.50, 0.70)

# Default support for the shapes that need one. Widened slightly beyond DECISION_WINDOW so a bump does
# not switch on discontinuously at the moment the fold starts forming.
_DEFAULT_SUPPORT = (0.35, 0.85)


def _raised_cosine(f: float, a: float, b: float) -> float:
    """Smooth 0 -> 1 -> 0 hump on [a, b], zero outside. Cosine rather than a triangle so the shape is
    differentiable at the seam, which is the same reason `plan/25` used a cosine ramp for the spatial
    feather."""
    if not (a <= f <= b) or b <= a:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * (f - a) / (b - a)))


def _shapes(support=_DEFAULT_SUPPORT, teeth: int = 3):
    a, b = support
    return {
        # --- the control: today's behaviour ---
        "constant":     lambda f: 1.0,
        # --- monotone. The family originally proposed, now DEMOTED to controls by results/28 ---
        "linear_decay": lambda f: 1.0 - f,
        "decay_quad":   lambda f: (1.0 - f) ** 2,
        "decay_cubic":  lambda f: (1.0 - f) ** 3,
        "decay_sine":   lambda f: math.cos(math.pi * f / 2.0),
        "linear_rise":  lambda f: f,                      # the opposite-prediction control
        # --- discontinuous: the IP-Adapter analogue, generalized to an intermediate level ---
        "window":       lambda f: 1.0 if a <= f <= b else 0.0,
        # --- non-monotone, from Gregg's chained-Penner-with-mirrors family ---
        "bump":         lambda f: _raised_cosine(f, a, b),                  # ⭐ FAVOURED (results/28)
        "valley":       lambda f: 1.0 - _raised_cosine(f, a, b),
        # --- oscillating: the adversarial arm, PREDICTED TO LOSE at matched mean (plan/57 §3) ---
        "sawtooth":     lambda f: (f * teeth) % 1.0,
    }


def shape_names(**kw) -> list[str]:
    return list(_shapes(**kw).keys())


def build(name: str, n_steps: int, target_lambda: float = 1.0, *,
          support=_DEFAULT_SUPPORT, teeth: int = 3) -> list[float]:
    """Per-step λ values for ``name``, rescaled so the realized mean is exactly ``target_lambda``.

    Args:
        name: a key from :func:`shape_names`.
        n_steps: number of sampler steps the schedule must cover.
        target_lambda: the mean λ every shape is held to, so shapes differ only in *distribution*.
        support: ``(a, b)`` trajectory-fraction support for ``window``, ``bump`` and ``valley``.
        teeth: period count for ``sawtooth``.

    Returns:
        ``n_steps`` floats, ``mean == target_lambda`` to floating-point exactness.

    Raises:
        ValueError: unknown ``name``, ``n_steps < 1``, or a shape whose mean is zero over this grid
            (which would make normalization undefined rather than silently producing zeros).
    """
    shapes = _shapes(support=support, teeth=teeth)
    if name not in shapes:
        raise ValueError(f"unknown λ shape {name!r}; known: {sorted(shapes)}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    fn = shapes[name]
    # frac 0 = noisiest step, frac 1 = final step, matching how results/28 reports the trajectory
    fracs = [0.0] * n_steps if n_steps == 1 else [i / (n_steps - 1) for i in range(n_steps)]
    raw = [float(fn(f)) for f in fracs]
    if any(v < 0 for v in raw):
        raise ValueError(f"shape {name!r} produced a negative weight; λ must be >= 0")
    mean = sum(raw) / n_steps
    if mean <= 0:
        raise ValueError(
            f"shape {name!r} has zero mean over {n_steps} steps (support={support}), so matched-mean "
            f"normalization is undefined. Widen the support or use more steps.")
    scale = target_lambda / mean
    return [v * scale for v in raw]


class ScheduledLambda(nn.Module):
    """Applies a per-step λ by wrapping the host's diffusion module. Opt-in; install explicitly.

    Mechanics, all verified rather than assumed (dev ``plan/56`` §2.3a, ``results/28``):

    * ``net.diffusion_module`` is called **exactly once per sampler step**. Recycling does not multiply
      the count, because ``n_recycle`` is passed *into* the module as a kwarg. Measured: 99 wrapper calls
      for a 100-step run. This is why a step counter is sound here, and why `novelty` §12.1's warning
      about a forward-counter double-counting applies to hooking the *net*, not this hook point.
    * ``SPAAdapter.set_scale`` writes a registered buffer on every block, so λ can be changed between
      forwards.
    * It must subclass ``nn.Module``: ``diffusion_module`` is a registered submodule and torch's
      ``__setattr__`` refuses a plain object in that slot. The wrapped module is stored via
      ``object.__setattr__`` so it is not re-registered as a child and no parameter is double-counted.

    ⚠️ ``set_scale`` receives a Python float and the host runs in bf16; dev ``07`` I.4 records that λ must
    reach the host in its dtype. That cast lives in the adapter, so this class must not pre-cast.
    """

    def __init__(self, module, adapter, values: list[float]):
        super().__init__()
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_adapter", adapter)
        self.values = list(values)
        self.calls = 0
        self.applied: list[dict] = []

    def forward(self, *args, **kwargs):
        i = min(self.calls, len(self.values) - 1)       # clamp: never index past the schedule
        lam = float(self.values[i])
        self._adapter.set_scale(lam)
        t = kwargs.get("t")
        self.applied.append({
            "call": self.calls,
            "step_index": i,
            "lambda": lam,
            "t": float(t.flatten()[0].item()) if t is not None else None,
        })
        self.calls += 1
        return self._module(*args, **kwargs)

    # ------------------------------------------------------------------------------------------
    def install(self, net):
        """Replace ``net.diffusion_module`` with this wrapper. Returns the original for restoration."""
        original = net.diffusion_module
        net.diffusion_module = self
        return original

    @staticmethod
    def uninstall(net, original):
        net.diffusion_module = original

    def realized_mean(self) -> float:
        """Mean λ actually applied, which is what a matched-mean comparison depends on. If this differs
        from the target, the run is not comparable and the driver should abort rather than report it."""
        if not self.applied:
            return float("nan")
        return sum(r["lambda"] for r in self.applied) / len(self.applied)
