"""Tests for the λ-over-diffusion-steps schedule (dev ``docs/plan/57_lambda_schedule.md``).

The load-bearing property is **matched mean λ**: every shape must deliver the same total conditioning so
that a sweep tests *distribution* rather than *total strength*. If that breaks, the experiment silently
measures the wrong thing, so it is asserted to floating-point tolerance rather than eyeballed.
"""

from __future__ import annotations

import pytest

from spa.model.lambda_schedule import (
    DECISION_WINDOW,
    ScheduledLambda,
    build,
    shape_names,
)


# ----------------------------------------------------------------------------------------------------
# matched mean: the property the whole comparison rests on
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", shape_names())
@pytest.mark.parametrize("n_steps", [10, 99, 100, 200])
@pytest.mark.parametrize("target", [0.5, 1.0, 2.0])
def test_matched_mean_is_exact(name, n_steps, target):
    """Every shape, at every step count and target, must realize the target mean."""
    v = build(name, n_steps, target)
    assert len(v) == n_steps
    assert sum(v) / n_steps == pytest.approx(target, rel=1e-12)


@pytest.mark.parametrize("name", shape_names())
def test_lambda_is_never_negative(name):
    """A negative λ would flip the sign of SPA's contribution, which is not a schedule but a different
    experiment."""
    assert min(build(name, 100, 1.0)) >= 0.0


# ----------------------------------------------------------------------------------------------------
# the constant shape must be indistinguishable from today's behaviour
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", [0.5, 1.0, 2.0, 3.0])
def test_constant_is_exactly_the_target_everywhere(target):
    """The identity control. If `constant` deviated anywhere, every schedule result would be measured
    against a moving baseline."""
    v = build("constant", 100, target)
    assert all(x == pytest.approx(target, rel=1e-12) for x in v)


# ----------------------------------------------------------------------------------------------------
# shape semantics: each curve must actually do what its name says
# ----------------------------------------------------------------------------------------------------

def test_monotone_shapes_are_monotone():
    for name in ("linear_decay", "decay_quad", "decay_cubic", "decay_sine"):
        v = build(name, 100, 1.0)
        assert all(v[i] >= v[i + 1] - 1e-12 for i in range(len(v) - 1)), f"{name} is not non-increasing"
    rise = build("linear_rise", 100, 1.0)
    assert all(rise[i] <= rise[i + 1] + 1e-12 for i in range(len(rise) - 1))


def test_bump_peaks_inside_the_measured_decision_window():
    """`results/28` puts the fold decision at trajectory fraction 0.50 to 0.70. The favoured shape has to
    put its mass there, or it is not the shape the measurement motivated."""
    n = 100
    v = build("bump", n, 1.0)
    peak_frac = v.index(max(v)) / (n - 1)
    lo, hi = DECISION_WINDOW
    assert lo <= peak_frac <= hi, f"bump peaks at {peak_frac:.3f}, outside {DECISION_WINDOW}"
    # and it must be a genuine hump: low at both ends
    assert v[0] < max(v) * 0.25 and v[-1] < max(v) * 0.25


def test_valley_is_the_mirror_of_bump():
    n = 100
    bump, valley = build("bump", n, 1.0), build("valley", n, 1.0)
    assert valley.index(min(valley)) == bump.index(max(bump))
    assert valley[0] > min(valley) and valley[-1] > min(valley)


def test_window_is_binary_on_its_support():
    v = build("window", 100, 1.0, support=(0.4, 0.6))
    assert len({round(x, 9) for x in v}) == 2, "window should take exactly two values"
    assert v[0] == 0.0 and v[-1] == 0.0


def test_sawtooth_oscillates_the_requested_number_of_times():
    """The adversarial arm. It must genuinely oscillate, since its whole purpose is to deliver the same
    mean in spikes (plan/57 §3)."""
    v = build("sawtooth", 300, 1.0, teeth=3)
    drops = sum(1 for i in range(len(v) - 1) if v[i + 1] < v[i] - 1e-9)
    assert drops == 3, f"expected 3 teeth, saw {drops} resets"


# ----------------------------------------------------------------------------------------------------
# input validation: fail loudly rather than silently produce a wrong schedule
# ----------------------------------------------------------------------------------------------------

def test_unknown_shape_raises():
    with pytest.raises(ValueError, match="unknown λ shape"):
        build("ease_out_quad", 100, 1.0)          # the name plan/57 used; deliberately not an alias


def test_zero_steps_raises():
    with pytest.raises(ValueError, match="n_steps must be"):
        build("constant", 0, 1.0)


def test_support_too_narrow_to_sample_raises_rather_than_zeroing():
    """A support so narrow that no step lands inside it would make the mean zero. That must raise, not
    silently return an all-zero schedule that looks like a successful λ=0 run."""
    with pytest.raises(ValueError, match="zero mean"):
        build("window", 10, 1.0, support=(0.401, 0.402))


# ----------------------------------------------------------------------------------------------------
# the hook: step accounting, clamping, and the realized mean it reports
# ----------------------------------------------------------------------------------------------------

class _FakeAdapter:
    def __init__(self):
        self.scales = []

    def set_scale(self, v):
        self.scales.append(float(v))


class _FakeModule:
    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return {"X_L": None}


def test_hook_applies_each_value_in_order():
    values = [0.1, 0.5, 2.0, 1.4]
    ad, mod = _FakeAdapter(), _FakeModule()
    hook = ScheduledLambda(mod, ad, values)
    for _ in values:
        hook.forward()
    assert ad.scales == values
    assert mod.calls == len(values)
    assert hook.realized_mean() == pytest.approx(sum(values) / len(values))


def test_hook_clamps_when_called_more_often_than_expected():
    """If the sampler ever calls the module more times than the schedule has entries, the hook must hold
    the last value rather than raise mid-rollout and lose the run."""
    ad, mod = _FakeAdapter(), _FakeModule()
    hook = ScheduledLambda(mod, ad, [1.0, 2.0])
    for _ in range(5):
        hook.forward()
    assert ad.scales == [1.0, 2.0, 2.0, 2.0, 2.0]


def test_hook_records_provenance_per_step():
    ad, mod = _FakeAdapter(), _FakeModule()
    hook = ScheduledLambda(mod, ad, [1.0, 2.0])
    hook.forward(); hook.forward()
    assert [r["step_index"] for r in hook.applied] == [0, 1]
    assert [r["lambda"] for r in hook.applied] == [1.0, 2.0]


def test_hook_is_an_nn_module_so_it_can_occupy_a_submodule_slot():
    """`diffusion_module` is a registered submodule; torch refuses a plain object there. Regression test
    for a real failure hit while building the CFG probe."""
    import torch.nn as nn

    hook = ScheduledLambda(_FakeModule(), _FakeAdapter(), [1.0])
    assert isinstance(hook, nn.Module)

    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_module = nn.Identity()

    host = Host()
    original = hook.install(host)
    assert host.diffusion_module is hook
    ScheduledLambda.uninstall(host, original)
    assert host.diffusion_module is original


def test_wrapped_module_is_not_registered_as_a_child():
    """Storing the wrapped module via object.__setattr__ keeps it out of the parent's child list, so no
    host parameter is double-counted in a state dict or an optimizer."""
    import torch.nn as nn

    inner = nn.Linear(4, 4)
    hook = ScheduledLambda(inner, _FakeAdapter(), [1.0])
    assert all(child is not inner for child in hook.children())
    assert len(list(hook.parameters())) == 0
