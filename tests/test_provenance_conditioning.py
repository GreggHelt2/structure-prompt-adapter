"""The ligand-steering disclosure is emitted automatically (dev ``90`` §5.0b, route (c)).

The point of these tests is that the disclosure cannot be forgotten: it is derived from the config, not
passed in by a caller. A monomer run must gain nothing, and a ligand run must gain both the structured
record and a line in ``anomalies``, which is the field a reader actually scans.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from spa.eval.provenance import collect, describe_conditioning


def _cfg(**eval_kw):
    base = {"eval": {"conditions": ["spa"], "lambda_scale": 1.0, "num_designs": 8,
                     "seed": 42, "out_dir": "/tmp/x"}}
    base["eval"].update(eval_kw)
    return OmegaConf.create(base)


def test_a_monomer_run_gains_no_conditioning_field():
    assert describe_conditioning(_cfg()) is None
    rec = collect(_cfg())
    assert "conditioning" not in rec
    assert not any("LIGAND" in a for a in rec["anomalies"])


def test_a_ligand_in_the_specification_triggers_the_disclosure():
    cond = describe_conditioning(_cfg(specification={"input": "x.pdb", "ligand": "IAI"}))
    assert cond is not None
    assert cond["ligand"] == "IAI"
    assert cond["ligand_tokens_steered"] is True
    assert cond["isolation"] == "NOT ISOLATED"
    assert "may NOT be described as isolating" in cond["disclosure"]


def test_a_ligand_on_the_motif_path_triggers_it_too():
    cond = describe_conditioning(_cfg(motif={"source_pdb": "e.pdb", "ligand": "NAI,ACT"}))
    assert cond is not None and cond["ligand"] == "NAI,ACT"


def test_the_disclosure_reaches_anomalies_where_a_reader_will_see_it():
    rec = collect(_cfg(specification={"input": "x.pdb", "ligand": "IAI"}))
    assert rec["conditioning"]["isolation"] == "NOT ISOLATED"
    assert any("LIGAND TOKENS STEERED" in a for a in rec["anomalies"])


def test_collect_never_raises_on_a_malformed_config():
    """Provenance must never take down a run that has already spent GPU time."""
    rec = collect(OmegaConf.create({"eval": {"specification": {"ligand": "IAI"}}}))
    assert rec["schema"] == "spa-run-provenance/1"
