"""Tier-2 prerequisites: the refold query must carry every chain AND the ligand (dev ``90`` §2.1/§3.1).

The load-bearing assertions are the two that reproduce the original defects: a complex must NOT be fused
into one chain, and a monomer query must be byte-identical to before any of this existed.
"""

from __future__ import annotations

import pytest

from spa.eval.openfold3 import OF3Refolder
from spa.eval.proteinmpnn import _build_command


def _r(**kw):
    return OF3Refolder(ckpt_path="c.pt", runner_yaml="r.yml", out_dir="/tmp/x", **kw)


# ------------------------------------------------------------------------------------- M2, chains

def test_a_monomer_query_is_unchanged():
    """No '/' means no behaviour change, so every historical refold is byte-identical."""
    body = _r()._chain("PVLSCGEWQCL")
    assert body == {"chains": [{"molecule_type": "protein", "chain_ids": ["A"],
                                "sequence": "PVLSCGEWQCL"}]}


def test_a_two_chain_sequence_becomes_TWO_chains_not_one_fused_one():
    """The original defect: '/' was deleted and the chains were submitted as one covalent chain."""
    body = _r()._chain("AAAA/BBB")
    assert [c["chain_ids"] for c in body["chains"]] == [["A"], ["B"]]
    assert [c["sequence"] for c in body["chains"]] == ["AAAA", "BBB"]
    # and the fusion that used to happen would have been a single 7-mer:
    assert not any(c["sequence"] == "AAAABBB" for c in body["chains"])


def test_chain_count_is_preserved_for_three_chains():
    assert len(_r()._chain("AA/BB/CC")["chains"]) == 3


# ------------------------------------------------------------------------------------ L2, ligand

def test_no_ligand_configured_emits_no_ligand_chain():
    assert all(c["molecule_type"] == "protein" for c in _r()._chain("AAA")["chains"])


def test_a_ccd_ligand_is_appended_as_its_own_chain():
    body = _r(ligand_ccd="IAI")._chain("AAAA")
    lig = body["chains"][-1]
    assert lig["molecule_type"] == "ligand" and lig["ccd_codes"] == "IAI"
    assert lig["chain_ids"] == ["B"]                    # after the one protein chain


def test_a_ligand_takes_the_id_after_the_LAST_protein_chain():
    lig = _r(ligand_ccd="IAI")._chain("AA/BB")["chains"][-1]
    assert lig["chain_ids"] == ["C"]


def test_smiles_wins_over_ccd_when_both_are_given():
    lig = _r(ligand_ccd="IAI", ligand_smiles="CCO")._chain("AAA")["chains"][-1]
    assert lig["smiles"] == "CCO" and "ccd_codes" not in lig


# ----------------------------------------------------------------- ProteinMPNN chain restriction

def _cmd(**kw):
    from pathlib import Path
    return _build_command(repo_dir=Path("/r"), pdb_path=Path("/d.pdb"), out_dir=Path("/o"),
                          num_seqs=8, sampling_temp=0.1, seed=42, batch_size=1,
                          weights_dir=Path("/w"), model_name="v_48_020", ca_only=False,
                          conda_env=None, **kw)


def test_without_design_chains_no_restriction_flag_is_passed():
    """Unset must stay ProteinMPNN's own default, which designs every chain, correct for a monomer."""
    assert "--pdb_path_chains" not in _cmd()


def test_design_chains_restricts_design_so_a_fixed_target_is_not_redesigned():
    cmd = _cmd(design_chains=["A"])
    assert cmd[cmd.index("--pdb_path_chains") + 1] == "A"
    assert _cmd(design_chains="A")[-1] == "A"
