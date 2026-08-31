"""The RFD3 sampler override surface: everything reachable, nothing silently dropped.

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

from spa.eval.generate import (_LEGACY_SAMPLER_KEYS, _sampler_fields,
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
