"""`eval.specification` must reach RFD3 as a FULLY plain dict (dev ``90`` §3a correction).

Regression test for a real failure on 2026-09-04: nested selection mappings stayed as OmegaConf
containers, and RFD3's selection validator rejects those, so `eval.specification` could carry scalars
but not `select_hotspots`, `select_fixed_atoms` or any other selection dict.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from spa.eval.generate import resolve_specification


def _ev(spec):
    return OmegaConf.create({"specification": spec})


def test_nested_selection_mappings_are_plain_dicts_not_omegaconf():
    ev = _ev({
        "input": "x.pdb",
        "contig": "100,/0,A17-131",
        "select_hotspots": {"A56": "CG,OH", "A115": "CG,SD"},
    })
    spec = resolve_specification(ev)
    assert type(spec) is dict
    assert type(spec["select_hotspots"]) is dict          # the actual bug: this was a DictConfig
    assert not OmegaConf.is_config(spec["select_hotspots"])
    assert spec["select_hotspots"]["A56"] == "CG,OH"


def test_the_shallow_conversion_this_replaced_would_still_fail():
    """Guard against a 'simplification' back to dict(), which is shallow and reintroduces the bug."""
    ev = _ev({"select_fixed_atoms": {"IAI": ""}})
    shallow = dict(ev.get("specification"))
    assert OmegaConf.is_config(shallow["select_fixed_atoms"])      # what broke every diffused cell
    assert not OmegaConf.is_config(resolve_specification(ev)["select_fixed_atoms"])


def test_empty_and_absent_specifications_stay_empty_dicts():
    assert resolve_specification(OmegaConf.create({})) == {}
    assert resolve_specification(_ev(None)) == {}


def test_a_plain_dict_passes_through_unchanged():
    assert resolve_specification({"specification": {"ligand": "IAI"}}) == {"ligand": "IAI"}
