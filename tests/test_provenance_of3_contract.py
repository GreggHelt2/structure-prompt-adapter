"""The refold stage's oracle identity must reach the provenance record.

⛔ WHY THIS TEST EXISTS, and it is not hypothetical. Queue row 25 generated deterministically and
then refolded on STOCK OpenFold3 for 15 h 57 m while being labelled contract v2. Nothing failed and
nothing logged, because an unset opt-in is byte-identical to a chosen opt-out, so the evidence a
reviewer looks for is ABSENT rather than wrong (root CLAUDE.md, 2026-09-18). It recurred on row 37:
`RUN_PROVENANCE.json` carried RFD3's determinism and nothing at all for OpenFold3, so contract v3
and MSA-free both had to be recovered from a subprocess log (dev results/73 §0).

⭐ The load-bearing case is `test_an_absent_key_is_not_recorded_as_off`: absent and off are
DIFFERENT answers, and conflating them is the whole defect.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from spa.eval.provenance import collect

_REFOLDER = {"_target_": "spa.eval.openfold3.OF3Refolder",
             "ckpt_path": "/nonexistent/of3.pt",
             "runner_yaml": "/nonexistent/of3_nokernel.yml",
             "use_msa_server": False}


def _cfg(**refolder):
    rf = {**_REFOLDER, **refolder} if refolder != {"__none__": True} else None
    ev = {"out_dir": "/tmp/x", "conditions": ["spa"]}
    ev["flywheel"] = {"refolder": rf} if rf is not None else {}
    return OmegaConf.create({"eval": ev})


def test_a_deterministic_refold_records_its_contract():
    of3 = collect(_cfg(deterministic=True))["openfold3"]
    assert of3["deterministic_requested"] is True
    assert of3["contract"] == 3, "the shim's CONTRACT must be read, not assumed"
    assert of3["refolder"] == "spa.eval.openfold3.OF3Refolder"


def test_declining_determinism_records_no_contract_rather_than_zero():
    """⚠️ null, never 0: unknown and off are different answers (dev plan/101 §6).

    determinism-exempt: this is the TEST of the declined-determinism branch, so the whole point is
    to pass deterministic=False and assert the record says so. No refold runs here; `collect()`
    only reads a config dict. The checker's rule (dev plan/106 §4a) is about RUNS, and a test that
    never wires a refolder cannot produce one.
    """
    of3 = collect(_cfg(deterministic=False))["openfold3"]
    assert of3["deterministic_requested"] is False
    assert of3["contract"] is None


def test_an_absent_key_is_not_recorded_as_off():
    """⭐ THE ROW 25 CASE. OF3Refolder's default is None -> deterministic, so an absent key means
    the refold IS shimmed. Recording it as `False` would reproduce the defect in the record."""
    of3 = collect(_cfg())["openfold3"]
    assert of3["deterministic_requested"] is None
    assert of3["deterministic_default_applies"] is True
    assert of3["contract"] == 3


def test_the_msa_setting_is_recorded_because_it_is_part_of_the_oracle():
    """MSA-free is the configuration of record for every number in the project (dev plan/96),
    and row 37 had to read `--use-msa-server=False` out of a log to confirm it."""
    assert collect(_cfg(deterministic=True))["openfold3"]["use_msa_server"] is False


def test_a_generation_only_run_records_no_refold_block():
    rec = collect(_cfg(__none__=True))
    assert rec.get("openfold3") is None
    assert rec["schema"] == "spa-run-provenance/1"


def test_a_malformed_refolder_never_takes_down_a_run():
    """Provenance must never fail a run that has already spent GPU time."""
    cfg = OmegaConf.create({"eval": {"out_dir": "/tmp/x", "flywheel": {"refolder": "not-a-dict"}}})
    rec = collect(cfg)
    assert rec["schema"] == "spa-run-provenance/1"
