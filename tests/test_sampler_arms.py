"""The sampler-arm registry is a single source of truth, so these tests guard that property.

Context: dev ``docs/plan/81_sampler_redo_scope.md`` §7. RFD3 has two published sampler
configurations and the failure mode this module prevents is a run that sets the step count but not
``gamma_0``, producing a third configuration that matches neither. The tests below pin the values,
pin the shell helper to the same values, and pin the one consumer whose output paths depend on the
historical key spellings.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from spa.eval.sampler_arms import (ALIASES, ARMS, DEFAULT_ARM, hydra_overrides,
                                   resolve, resolve_with_legacy)

REPO = Path(__file__).resolve().parents[1]


def test_arms_are_the_two_published_configurations():
    # RFdiffusion3 paper Fig. S4f, and the released checkpoint's own train_cfg. If either of these
    # changes, every absolute designability number in the paper is comparing against something else.
    assert ARMS["rfd3"] == {"num_timesteps": 200, "gamma_0": 0.6, "step_scale": 1.5}
    assert ARMS["ours"] == {"num_timesteps": 100, "gamma_0": 0.8, "step_scale": 1.5}


def test_default_is_the_checkpoints_own_settings():
    # The default must stay numerically identical to inheriting from the checkpoint, so that adding
    # the arm machinery to a driver changes provenance without changing results.
    assert DEFAULT_ARM == "ours"
    assert resolve() == ARMS["ours"]


def test_every_arm_specifies_all_three_knobs():
    # The whole point: an arm can never be partially specified.
    for name, settings in ARMS.items():
        assert set(settings) == {"num_timesteps", "gamma_0", "step_scale"}, name


def test_aliases_resolve_and_do_not_shadow_real_arms():
    for alias, target in ALIASES.items():
        assert alias not in ARMS, f"alias {alias!r} shadows a real arm"
        assert resolve(alias) == ARMS[target]


def test_unknown_arm_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown sampler arm"):
        resolve("200steps")


def test_hydra_overrides_are_settable_eval_keys():
    ov = hydra_overrides("rfd3")
    assert ov == ["eval.num_timesteps=200", "eval.gamma_0=0.6", "eval.step_scale=1.5"]
    # configs/eval/default.yaml must declare all three, or Hydra rejects the override without a `+`.
    declared = (REPO / "configs/eval/default.yaml").read_text()
    for key in ARMS["rfd3"]:
        assert f"\n{key}:" in declared, f"{key} is not a declared eval key"


@pytest.mark.parametrize("arm,ts", [("ours", None), ("ours", 100), ("rfd3", 200), ("rfd3", None)])
def test_legacy_num_timesteps_is_accepted_when_it_agrees(arm, ts):
    assert resolve_with_legacy(arm, ts) == ARMS[arm]


def test_legacy_num_timesteps_is_rejected_when_it_disagrees():
    # This is the exact historical hazard: 200 steps with the checkpoint's gamma_0 = 0.8.
    with pytest.raises(SystemExit, match="disagrees"):
        resolve_with_legacy("ours", 200)


def test_shell_helper_agrees_with_the_python_module():
    """`scripts/_sampler_arm.sh` must not drift from the module it reads."""
    for arm in sorted(ARMS):
        out = subprocess.run(
            ["bash", "-c", f'ARM={arm}; . "{REPO}/scripts/_sampler_arm.sh"; '
                           'printf "%s\\n" "${SAMPLER_ARGS[@]}"'],
            capture_output=True, text=True, cwd=REPO,
            env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(Path.home())},
        )
        assert out.returncode == 0, out.stderr
        emitted = [ln for ln in out.stdout.splitlines() if ln.startswith("eval.")]
        assert emitted == hydra_overrides(arm), f"{arm}: shell {emitted} != python {hydra_overrides(arm)}"


def test_shell_helper_aborts_on_an_unknown_arm():
    out = subprocess.run(
        ["bash", "-c", f'ARM=nope; . "{REPO}/scripts/_sampler_arm.sh"; echo REACHED'],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(Path.home())},
    )
    assert out.returncode == 2
    assert "REACHED" not in out.stdout


def test_bench_m10_keys_still_name_its_archived_output_paths():
    """`bench_m10_overhead.py` derives from this module but must keep its `s100`/`s200` path keys.

    Those directory names are cited in dev `docs/results/34`; a rename would orphan the artifacts.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bench_m10", REPO / "scripts/eval/bench_m10_overhead.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._SAMPLERS == {"100": {"num_timesteps": 100, "gamma_0": 0.8},
                             "200": {"num_timesteps": 200, "gamma_0": 0.6}}


# --------------------------------------------------------------------------------------------------
# Driver wiring. These are static checks on purpose: the failure they guard against is a driver that
# sources the helper (so it LOGS an arm) but never splices SAMPLER_ARGS into the command it runs,
# which would generate at the checkpoint's settings while the log claimed otherwise. That is exactly
# the class of silent mismatch this whole module exists to prevent, and it cannot be caught by
# running the drivers, which need a GPU, staged weights and a container.
# --------------------------------------------------------------------------------------------------

#: Drivers that shell out to a Hydra entry point and must splice the resolved overrides in.
_SPLICING_DRIVERS = [
    "scripts/cloud/run_eval.sh",
    "scripts/cloud/run_variant_desig.sh",
    "scripts/cloud/run_finetune_eval.sh",
    "scripts/cloud/run_trivial_baseline.sh",
    "scripts/eval/run_enzyme_tier0.sh",
]

#: Drivers that pass the arm through to a python entry point by flag instead.
_FLAG_DRIVERS = ["scripts/cloud/run_threeway_sweep.sh"]


@pytest.mark.parametrize("driver", _SPLICING_DRIVERS)
def test_driver_sources_the_helper_and_splices_the_args(driver):
    text = (REPO / driver).read_text()
    assert "_sampler_arm.sh" in text, f"{driver} does not source the helper"
    assert '"${SAMPLER_ARGS[@]}"' in text, f"{driver} sources the helper but never uses SAMPLER_ARGS"


@pytest.mark.parametrize("driver", _FLAG_DRIVERS)
def test_driver_passes_the_arm_by_flag(driver):
    assert "--sampler-arm" in (REPO / driver).read_text()


@pytest.mark.parametrize("driver", _SPLICING_DRIVERS + _FLAG_DRIVERS)
def test_no_driver_keeps_the_bare_num_timesteps_hazard(driver):
    """A step-count env var with no gamma_0 beside it is the exact configuration to forbid.

    `run_eval.sh` shipped `NUM_TIMESTEPS="${NUM_TIMESTEPS:-}"` for months, so setting it to 200 gave a
    run at the paper's step count and the checkpoint's gamma_0 = 0.8: neither published configuration.
    """
    text = (REPO / driver).read_text()
    assert 'NUM_TIMESTEPS="${NUM_TIMESTEPS' not in text, f"{driver} still exposes a bare NUM_TIMESTEPS"
    assert "eval.num_timesteps=$" not in text, f"{driver} hand-builds a step-count override"


@pytest.mark.parametrize("script", [
    "scripts/eval/probe_two_steer.py",
    "scripts/eval/probe_hard_soft_free.py",
    "scripts/eval/probe_localization.py",
    "scripts/eval/run_scaffold_eval.py",
])
def test_generating_probe_resolves_an_arm_rather_than_a_bare_step_count(script):
    text = (REPO / script).read_text()
    assert "resolve_with_legacy" in text, f"{script} does not resolve a sampler arm"
    assert "add_arm_argument" in text, f"{script} does not expose --sampler-arm"
    # The old pattern: feeding args.num_timesteps straight into the eval config, gamma_0 untouched.
    assert '"num_timesteps": args.num_timesteps' not in text, f"{script} still wires the bare flag"
