"""The SPA conditioning a run armed must reach the provenance record (dev plan/159b).

⛔ WHY THIS TEST EXISTS. The design-side λ profile and the prompt-side selection are armed POST-COMPOSE
via ``set_profile``/``set_prompts`` and serialize NOWHERE on their own, so the multi-region conditioning
that the paper's composability results depend on was unrecoverable from any artifact. This block records
it: per soft prompt the PROMPT-side selection (which prompt rows attended) and the DESIGN-side selection
(which design residues steered), plus any Fixed Motif's source->design span.

⭐ The load-bearing invariant is that a BASELINE arm gains NO block, so ``spa_conditioning`` absent is the
exact signal for baseline (dev plan/159 §4.1). All design/prompt spans are HALF-OPEN [lo, hi).
"""
from __future__ import annotations

from omegaconf import OmegaConf

from spa.eval.provenance import (
    collect,
    describe_spa_conditioning,
    design_span,
    motif_slot,
    of3_contract_applied,
    prompt_keep_range,
    soft_slot,
    spa_conditioning,
)


def _cfg(**eval_kw):
    base = {"eval": {"conditions": ["spa"], "lambda_scale": 1.0, "num_designs": 8,
                     "seed": 42, "out_dir": "/tmp/x"}}
    base["eval"].update(eval_kw)
    return OmegaConf.create(base)


# --- the cfg-derived path (the main generate.py capture) ----------------------------------------

def test_a_baseline_run_gains_no_block():
    """conditions without `spa` and no motif == baseline; the block must be ABSENT (the baseline signal)."""
    cfg = OmegaConf.create({"eval": {"conditions": ["baseline"], "out_dir": "/tmp/x"}})
    assert describe_spa_conditioning(cfg) is None
    assert "spa_conditioning" not in collect(cfg)


def test_a_spa_arm_with_no_prompt_emits_no_slot():
    """`spa` in conditions but no prompt source resolvable -> no soft slot (not a null-prompt slot)."""
    assert describe_spa_conditioning(_cfg()) is None


def test_a_whole_prompt_spa_run_records_one_whole_slot():
    block = describe_spa_conditioning(_cfg(prompt_pdb="A0A2X2KHU0.pdb"))
    assert block is not None
    assert block["capture_source"] == "cfg"
    assert block["n_soft_prompts"] == 1 and block["n_fixed_motifs"] == 0
    slot = block["soft_prompts"][0]
    assert slot["slot"] == 1
    assert slot["prompt_id"] == "A0A2X2KHU0"
    assert slot["prompt_side"]["kind"] == "whole"
    assert slot["design_side"]["kind"] == "whole"


def test_subregion_and_profile_record_the_two_independent_spans():
    block = describe_spa_conditioning(_cfg(
        prompt_pdb="A0A2X2KHU0.pdb",
        subregion={"keep_range": [10, 50]},
        profile={"tokens": 100, "steered": [[0, 40]]},
    ))
    slot = block["soft_prompts"][0]
    assert slot["prompt_side"]["kind"] == "keep_range"
    assert slot["prompt_side"]["keep_range"] == [10, 50]
    assert slot["prompt_side"]["M_attended"] == 40
    assert slot["design_side"]["kind"] == "profile"
    assert slot["design_side"]["spans"] == [[0, 40]]
    assert slot["design_side"]["n_steered"] == 40


def test_a_motif_records_its_source_and_source_span():
    block = describe_spa_conditioning(_cfg(
        conditions=["baseline"],  # motif applies to every condition; no soft prompt here
        motif={"source_pdb": "1CTT.pdb", "contig": "A2-20"},
    ))
    assert block is not None
    assert block["n_soft_prompts"] == 0 and block["n_fixed_motifs"] == 1
    m = block["fixed_motifs"][0]
    assert m["slot"] == 1
    assert m["source"] == "1CTT.pdb"
    assert m["source_span"] == "A2-20"
    # design_span is best-effort (depends on the contig grammar); when present it is a half-open [lo, hi).
    assert m["design_span"] is None or (isinstance(m["design_span"], list) and len(m["design_span"]) == 2)


def test_the_block_reaches_the_record_under_schema_2():
    rec = collect(_cfg(prompt_pdb="A0A2X2KHU0.pdb"))
    assert rec["schema"] == "spa-run-provenance/2"
    assert rec["spa_conditioning"]["capture_source"] == "cfg"


# --- the driver-passed path (the probe drivers) -------------------------------------------------

def test_the_builders_produce_a_well_formed_two_steer_block():
    block = spa_conditioning(
        soft=[soft_slot("G1", "g1.pdb", prompt_side=prompt_keep_range(0, 54),
                        design_side=design_span(0, 54, lambda_region=2.0)),
              soft_slot("G2", "g2.pdb", design_side=design_span(60, 114, lambda_region=1.5))],
        motif=[motif_slot("src.pdb", "A2-20", [54, 60])],
    )
    assert block["capture_source"] == "driver"
    assert block["n_soft_prompts"] == 2 and block["n_fixed_motifs"] == 1
    assert [s["slot"] for s in block["soft_prompts"]] == [1, 2]
    assert block["soft_prompts"][0]["design_side"]["spans"] == [[0, 54]]
    assert block["soft_prompts"][0]["design_side"]["lambda_region"] == 2.0
    assert block["soft_prompts"][1]["prompt_side"]["kind"] == "whole"
    assert block["fixed_motifs"][0]["design_span"] == [54, 60]


def test_an_explicit_block_overrides_the_cfg_derived_one():
    driver = spa_conditioning(soft=[soft_slot("DRIVER", "d.pdb")])
    rec = collect(_cfg(prompt_pdb="A0A2X2KHU0.pdb"), spa_conditioning=driver)
    assert rec["spa_conditioning"]["capture_source"] == "driver"
    assert rec["spa_conditioning"]["soft_prompts"][0]["prompt_id"] == "DRIVER"


def test_spa_conditioning_builder_is_none_when_empty():
    assert spa_conditioning(soft=[], motif=[]) is None


def test_collect_never_raises_on_a_malformed_cfg():
    rec = collect(OmegaConf.create({"eval": {"conditions": ["spa"], "profile": "not-a-dict"}}))
    assert rec["schema"] == "spa-run-provenance/2"


# --- the OF3 applied-contract banner parse (residual #1) ----------------------------------------

def test_of3_contract_applied_parses_the_banner_not_the_warning(tmp_path):
    """⚠️ The shim's own warning line contains "contract v1 or v2"; the parser must read the BANNER line,
    anchored on the `[of3_determinism_patch]` prefix, not that warning (of3_determinism_patch.py:183-188)."""
    log = tmp_path / "of3_batch" / "of3_subprocess.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "[of3_determinism_patch] contract v3: scatter_add_ routed through torch's deterministic path.\n"
        "Refolds will NOT match default-build runs, nor contract v1 or v2 runs, at the same seed.\n"
    )
    assert of3_contract_applied(tmp_path) == 3


def test_of3_contract_applied_is_none_when_the_log_is_absent(tmp_path):
    assert of3_contract_applied(tmp_path) is None
