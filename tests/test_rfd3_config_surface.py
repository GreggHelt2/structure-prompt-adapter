"""The RFD3 config surface, both halves: everything reachable, nothing silently dropped.

Context: dev ``docs/plan/81_sampler_redo_scope.md`` §7d. RFdiffusion3's ``SampleDiffusionConfig`` has
~21 fields and SPA exposed three. The other eighteen were not merely undocumented, they were
unreachable, and worse, ``+eval.gamma_min=2.0`` was *accepted* by Hydra and then silently ignored:
the run reported success and generated at the checkpoint's value with nothing downstream able to
tell. These tests pin the fix.
"""

from __future__ import annotations

import pytest
import yaml
from omegaconf import OmegaConf
from pathlib import Path

from spa.eval.generate import (_ENGINE_DERIVED, _LEGACY_SAMPLER_KEYS, _engine_fields,
                               _sampler_fields, resolve_engine_overrides,
                               resolve_sampler_overrides)

REPO = Path(__file__).resolve().parents[1]


def ev(**kw):
    """An ``eval`` group with every sampler entry unset, plus whatever the test sets."""
    base = {k: None for k in _LEGACY_SAMPLER_KEYS}
    base["sampler"] = None
    base.update(kw)
    return OmegaConf.create(base)


def test_nothing_set_yields_an_empty_override_dict():
    # An empty dict is what leaves the released checkpoint's own values standing, which is the
    # historical default behaviour and must not change.
    assert resolve_sampler_overrides(ev()) == {}


def test_the_three_legacy_top_level_keys_still_work():
    # Every driver in this repo and every invocation recorded in the dev docs uses these.
    got = resolve_sampler_overrides(ev(num_timesteps=200, gamma_0=0.6, step_scale=1.5))
    assert got == {"num_timesteps": 200, "gamma_0": 0.6, "step_scale": 1.5}


def test_namespaced_block_reaches_previously_unreachable_fields():
    got = resolve_sampler_overrides(ev(sampler={"gamma_min": 2.0, "noise_scale": 1.01}))
    assert got == {"gamma_min": 2.0, "noise_scale": 1.01}


def test_legacy_and_namespaced_compose():
    got = resolve_sampler_overrides(ev(num_timesteps=200, sampler={"gamma_min": 2.0}))
    assert got == {"num_timesteps": 200, "gamma_min": 2.0}


# --------------------------------------------------------------------------------------------------
# The three refusals. Each one is a configuration that previously ran and produced a wrong or
# unverifiable result.
# --------------------------------------------------------------------------------------------------

def test_top_level_sampler_field_raises_instead_of_being_dropped():
    """THE trap. `+eval.gamma_min=2.0` was accepted by Hydra and silently ignored."""
    with pytest.raises(RuntimeError, match="SILENTLY IGNORED"):
        resolve_sampler_overrides(ev(gamma_min=2.0))


def test_the_error_names_the_correct_spelling():
    with pytest.raises(RuntimeError, match=r"eval\.sampler\.gamma_min"):
        resolve_sampler_overrides(ev(gamma_min=2.0))


def test_unknown_namespaced_field_raises_and_lists_the_valid_ones():
    with pytest.raises(RuntimeError, match="not a field of RFD3's SampleDiffusionConfig"):
        resolve_sampler_overrides(ev(sampler={"gamma_naught": 0.6}))


def test_one_knob_set_two_ways_raises_rather_than_picking_one():
    with pytest.raises(RuntimeError, match="set two ways"):
        resolve_sampler_overrides(ev(num_timesteps=200, sampler={"num_timesteps": 100}))


def test_the_same_value_both_ways_is_not_an_error():
    got = resolve_sampler_overrides(ev(num_timesteps=200, sampler={"num_timesteps": 200}))
    assert got["num_timesteps"] == 200


# --------------------------------------------------------------------------------------------------
# Coercion. OmegaConf can hand back strings.
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [("true", True), ("false", False), ("1", True), ("0", False)])
def test_bool_fields_parse_text_rather_than_casting(raw, want):
    # bool("false") is True, which would silently enable a flag the caller disabled.
    got = resolve_sampler_overrides(ev(sampler={"use_classifier_free_guidance": raw}))
    assert got["use_classifier_free_guidance"] is want


def test_non_boolean_text_on_a_bool_field_raises():
    with pytest.raises(RuntimeError, match="not a boolean"):
        resolve_sampler_overrides(ev(sampler={"use_classifier_free_guidance": "maybe"}))


def test_numeric_fields_coerce_to_their_declared_type():
    got = resolve_sampler_overrides(ev(num_timesteps="200", gamma_0="0.6"))
    assert got["num_timesteps"] == 200 and isinstance(got["num_timesteps"], int)
    assert got["gamma_0"] == pytest.approx(0.6) and isinstance(got["gamma_0"], float)


# --------------------------------------------------------------------------------------------------
# Structural guarantees: the two decisions in §7d that could silently rot.
# --------------------------------------------------------------------------------------------------

def test_the_field_list_is_introspected_not_copied():
    """A copied list would drift the moment the host gains a knob, which is the whole bug."""
    from rfd3.model.inference_sampler import SampleDiffusionConfig
    assert set(_sampler_fields()) == set(SampleDiffusionConfig.__dataclass_fields__)
    # And the three legacy names must really be sampler fields, not a private invention.
    assert set(_LEGACY_SAMPLER_KEYS) <= set(_sampler_fields())


def test_eval_top_level_keys_do_not_collide_with_sampler_fields():
    """Why the block is namespaced rather than a top-level passthrough.

    If SPA passed through any `eval.<k>` matching a sampler field, a future eval key named `p`,
    `kind` or `solver` would silently become a sampler override. This asserts the sets stay disjoint
    apart from the three legacy names, so the guard above can never fire on an unrelated key.
    """
    declared = set(yaml.safe_load((REPO / "configs/eval/default.yaml").read_text()))
    collisions = declared & set(_sampler_fields())
    assert collisions == set(_LEGACY_SAMPLER_KEYS), (
        f"configs/eval/default.yaml now declares {sorted(collisions - set(_LEGACY_SAMPLER_KEYS))}, "
        "which collide with RFD3 sampler field names. Rename the eval key, or the guard in "
        "resolve_sampler_overrides will reject a legitimate config."
    )


def test_eval_default_declares_the_namespaced_block():
    declared = yaml.safe_load((REPO / "configs/eval/default.yaml").read_text())
    assert "sampler" in declared and declared["sampler"] is None


# ==================================================================================================
# The ENGINE half. Same mechanism, one structural difference: engine fields are dataclass keyword
# arguments, so they are set by construction and cannot silently fail to land the way a sampler
# override merged onto the checkpoint's train_cfg can.
# ==================================================================================================

def evx(**kw):
    """An ``eval`` group for the engine tests."""
    base = {"engine": None, "specification": None, "seed": None}
    base.update(kw)
    return OmegaConf.create(base)


def test_engine_defaults_to_no_overrides():
    assert resolve_engine_overrides(evx()) == {}


def test_engine_block_reaches_previously_unreachable_fields():
    got = resolve_engine_overrides(evx(engine={"low_memory_mode": True, "global_prefix": "run_"}))
    assert got == {"low_memory_mode": True, "global_prefix": "run_"}


def test_engine_fields_coerce_from_text():
    got = resolve_engine_overrides(evx(engine={"low_memory_mode": "true", "num_nodes": "2"}))
    assert got["low_memory_mode"] is True
    assert got["num_nodes"] == 2 and isinstance(got["num_nodes"], int)


def test_derived_engine_fields_are_refused_with_the_key_that_controls_them():
    """Accepting both would make the effective value depend on argument order."""
    for field, owner_fragment in [("seed", "eval.seed"),
                                  ("diffusion_batch_size", "eval.num_designs"),
                                  ("ckpt_path", "paths.rfd3_ckpt")]:
        with pytest.raises(RuntimeError, match="derived by SPA"):
            resolve_engine_overrides(evx(engine={field: 1}))
        try:
            resolve_engine_overrides(evx(engine={field: 1}))
        except RuntimeError as exc:
            assert owner_fragment in str(exc)


def test_unknown_engine_field_raises():
    with pytest.raises(RuntimeError, match="not a field of RFD3's RFD3InferenceConfig"):
        resolve_engine_overrides(evx(engine={"lo_memory_mode": True}))


def test_top_level_engine_field_raises_instead_of_being_dropped():
    with pytest.raises(RuntimeError, match="SILENTLY IGNORED"):
        resolve_engine_overrides(evx(low_memory_mode=True))


def test_derived_engine_fields_that_are_also_eval_keys_do_not_trip_the_guard():
    """`specification` and `seed` are BOTH eval keys and engine fields, and are legitimately set."""
    assert resolve_engine_overrides(evx(seed=0, specification={"length": 100})) == {}


def test_engine_field_list_is_introspected_not_copied():
    from rfd3.engine import RFD3InferenceConfig
    assert set(_engine_fields()) == set(RFD3InferenceConfig.__dataclass_fields__)


def test_every_derived_name_is_a_real_engine_field():
    """A typo in _ENGINE_DERIVED would silently un-refuse a derived field."""
    assert set(_ENGINE_DERIVED) <= set(_engine_fields())


def test_the_whole_engine_surface_is_now_reachable():
    """The point of the change: nothing is unreachable any more."""
    settable = set(_engine_fields()) - set(_ENGINE_DERIVED)
    got = resolve_engine_overrides(evx(engine={k: None for k in settable}))
    assert got == {}, "None means 'unset', so an all-None block must be a no-op"
    # And each one individually resolves rather than raising.
    for field in sorted(settable):
        resolve_engine_overrides(evx(engine={field: None}))


def test_eval_top_level_keys_do_not_collide_with_engine_fields():
    """Why this block is namespaced too. Only the derived pair may overlap."""
    declared = set(yaml.safe_load((REPO / "configs/eval/default.yaml").read_text()))
    collisions = declared & set(_engine_fields())
    assert collisions <= set(_ENGINE_DERIVED), (
        f"configs/eval/default.yaml declares {sorted(collisions - set(_ENGINE_DERIVED))}, which "
        "collide with RFD3InferenceConfig field names and would trip the guard."
    )


def test_eval_default_declares_the_engine_block():
    declared = yaml.safe_load((REPO / "configs/eval/default.yaml").read_text())
    assert "engine" in declared and declared["engine"] is None


# ==================================================================================================
# Integration. The unit tests above prove the RESOLVERS behave; this proves the resolved values reach
# a real engine built from the real checkpoint. Skipped where the checkpoint or a GPU is absent, the
# same guard test_identity_at_init.py uses, so the suite stays green elsewhere.
# ==================================================================================================

import os

import torch

_CKPT = os.environ.get("SPA_RFD3_CKPT", os.path.expanduser("~/projects/spa/models/rfdiffusion3/rfd3_latest.ckpt"))

#: Fields `RFD3InferenceEngine.__init__` STORES. `low_memory_mode` and `verbose` are consumed without
#: being stored, so they cannot be observed after construction; that they arrive at all is proven by
#: construction succeeding, since both are required keyword arguments.
_OBSERVABLE = ["global_prefix", "prevalidate_inputs", "skip_existing", "align_trajectory_structures"]


@pytest.mark.skipif(not (os.path.exists(_CKPT) and torch.cuda.is_available()),
                    reason="real RFD3 ckpt and/or CUDA device not available")
def test_engine_overrides_reach_a_real_engine():
    from spa.eval.generate import build_eval_engine

    def cfg(**extra):
        base = {"num_timesteps": None, "gamma_0": None, "step_scale": None, "sampler": None,
                "engine": None, "specification": None, "seed": 0, "num_designs": 1, "length": 32}
        base.update(extra)
        return OmegaConf.create({"paths": {"rfd3_ckpt": _CKPT}, "eval": base})

    unset = build_eval_engine(cfg())
    assert {k: getattr(unset, k) for k in _OBSERVABLE} == {
        "global_prefix": None, "prevalidate_inputs": True,
        "skip_existing": True, "align_trajectory_structures": False,
    }, "RFD3's own defaults changed, or the block leaked when unset"

    want = {"global_prefix": "spa_test_", "prevalidate_inputs": False,
            "skip_existing": False, "align_trajectory_structures": True}
    # ⛔ Deliberately NOT setting low_memory_mode here. It latches a process-global env var that
    # RFD3 never clears (see resolve_engine_overrides), and doing so in-process broke an unrelated
    # training test later in the same pytest session. Its warning is covered by a unit test instead.
    overridden = build_eval_engine(cfg(engine=dict(want, verbose=True)))
    assert {k: getattr(overridden, k) for k in _OBSERVABLE} == want


def test_low_memory_mode_warns_that_it_is_a_process_global_latch(capsys):
    """RFD3 sets RFD3_LOW_MEMORY_MODE=1 and never clears it, changing P_LL handling process-wide."""
    got = resolve_engine_overrides(evx(engine={"low_memory_mode": True}))
    assert got == {"low_memory_mode": True}, "the field must still be reachable, just loudly"
    out = capsys.readouterr()
    assert "process-global" in out.err and "RFD3_LOW_MEMORY_MODE" in out.err


def test_low_memory_mode_false_does_not_warn(capsys):
    resolve_engine_overrides(evx(engine={"low_memory_mode": False}))
    assert "RFD3_LOW_MEMORY_MODE" not in capsys.readouterr().err
