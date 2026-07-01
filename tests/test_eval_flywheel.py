"""End-to-end smoke test for the SPA validation-flywheel driver (dev 05_validation_pipeline.md §1–§4).

Wires Stage 1 (generate RFD3 ± SPA) → Stage 2 (ProteinMPNN inverse fold) → Stage 3 (OpenFold3 refold,
**STUBBED**: ``refolder=None`` ⇒ adherence-only) → Stage 4 (score / aggregate / Δ vs baseline) on a
TINY matrix and asserts the artifacts line up. Gated on the real RFD3 ckpt + a CUDA device (Stage 1
runs the real sampler, exactly like ``test_eval_generate.py``), so the suite stays green elsewhere.

OF3 is stubbed by passing **no** refolder: designability is skipped (``refolds=None``) and only the
adherence headline (TM-score vs the prompt's source structure) is scored — the SPA-vs-baseline number
that needs no OF3 (dev ``05`` §3). The adherence reference is a small random-CA PDB written here and
pointed at via ``eval.flywheel.prompt_struct``; the generation prompt is a separate random ``[N,1536]``
cache (so ESM3 need not load — same fast path as ``test_eval_generate.py``).

FAST by construction: conditions=[baseline,spa], λ=1, K=2, length 16, few sampler steps, 2 seqs/backbone.
"""

import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# Heavy GPU integration test — set these env vars to run it locally; otherwise it skips (below).
CKPT = os.environ.get("SPA_RFD3_CKPT")
PROTEINMPNN_REPO = os.environ.get("SPA_PROTEINMPNN_REPO")
LENGTH = int(os.environ.get("SPA_EVAL_TEST_LENGTH", "16"))
TIMESTEPS = int(os.environ.get("SPA_EVAL_TEST_TIMESTEPS", "8"))
K = 2

pytestmark = pytest.mark.skipif(
    not (CKPT and os.path.exists(CKPT) and PROTEINMPNN_REPO and os.path.isdir(PROTEINMPNN_REPO)
         and torch.cuda.is_available()),
    reason="set SPA_RFD3_CKPT + SPA_PROTEINMPNN_REPO and have a CUDA device to run this GPU test",
)


def _make_prompt_pdb(path, n=LENGTH):
    """Write a tiny random-CA PDB to serve as the adherence reference (a stand-in prompt structure)."""
    from biotite.structure import AtomArray
    from biotite.structure.io.pdb import PDBFile

    arr = AtomArray(n)
    arr.coord = (np.random.RandomState(0).randn(n, 3) * 3.0).astype("float32")
    arr.chain_id = np.array(["A"] * n)
    arr.res_id = np.arange(1, n + 1)
    arr.res_name = np.array(["GLY"] * n)
    arr.atom_name = np.array(["CA"] * n)
    arr.element = np.array(["C"] * n)
    pdb = PDBFile()
    pdb.set_structure(arr)
    pdb.write(str(path))
    return path


def _cfg(out_dir, seq_dir, prompt_cache, prompt_struct):
    return OmegaConf.create(
        {
            "paths": {"rfd3_ckpt": CKPT, "proteinmpnn_repo": PROTEINMPNN_REPO},
            "hardware": {"device": "cuda:0"},
            "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                      "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
            "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                        "strip_bos_eos": True, "use_clss": False},
            "eval": {
                "ckpt": None,                       # untrained zero-init adapter (identity)
                "conditions": ["baseline", "spa"],
                "lambda_scale": 1.0,
                "num_designs": K,
                "length": LENGTH,
                "specification": None,
                "num_timesteps": TIMESTEPS,
                "seed": 0,
                "prompt_pdb": None,
                "prompt_cache": str(prompt_cache),  # random [N,1536] -> no ESM3 load (fast)
                "use_sequence": False,
                "prompt_id": "smoke",
                "out_dir": str(out_dir),
                "proteinmpnn": {
                    "num_seqs": 2, "sampling_temp": 0.1, "seed": 0, "batch_size": 1,
                    "model_name": "v_48_020", "weights_dir": None, "ca_only": False,
                    "conda_env": None, "designs": None, "design_dir": None, "out_dir": str(seq_dir),
                },
                "score": {"scrmsd_cutoff": 2.0, "plddt_cutoff": 80.0, "tm_norm": "prompt",
                          "diversity": True, "novelty": False},
                "flywheel": {"prompt_struct": str(prompt_struct), "refolder": None},  # OF3 stubbed
            },
        }
    )


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """Run the full flywheel ONCE (OF3 stubbed) and share the returned artifacts across asserts."""
    from spa.eval.flywheel import run_flywheel

    tmp = tmp_path_factory.mktemp("spa_flywheel")
    prompt_cache = tmp / "prompt.pt"
    torch.save(torch.randn(12, 1536), prompt_cache)            # stand-in ESM3 [N,1536] cache entry
    prompt_struct = _make_prompt_pdb(tmp / "prompt.pdb")
    cfg = _cfg(tmp / "designs", tmp / "seqs", prompt_cache, prompt_struct)
    return run_flywheel(cfg)                                   # refolder=None -> adherence-only


def test_designs_generated_both_conditions(result):
    conds = {d.condition for d in result["designs"]}
    assert conds == {"baseline", "spa"}
    assert len(result["designs"]) == 2 * K                    # K per condition
    for d in result["designs"]:
        assert d.path.exists() and d.path.suffix == ".pdb"


def test_inverse_folded(result):
    seqsets = result["seqsets"]
    assert len(seqsets) == len(result["designs"])             # one SequenceSet per design
    names = {ss.name for ss in seqsets}
    assert names == {d.path.stem for d in result["designs"]}  # keyed by design name
    for ss in seqsets:
        assert len(ss.sequences) == 2                         # num_seqs


def test_scored_adherence_present_designability_skipped(result):
    scores = result["scores"]
    assert len(scores) == len(result["designs"])
    for s in scores:
        assert s.tm_score is not None                         # adherence scored (OF3 not needed)
        assert s.scrmsd is None and s.designable is None      # designability skipped (refolds=None)


def test_aggregate_and_delta(result):
    summaries, deltas = result["summaries"], result["deltas"]
    conds = {s.condition for s in summaries}
    assert conds == {"baseline", "spa"}
    for s in summaries:
        assert s.tm.n >= 1 and s.tm.mean is not None          # adherence distribution populated
        assert s.success_rate is None                         # no designability scored
    assert len(deltas) == 1                                   # one non-baseline (spa, λ=1) vs baseline
    assert deltas[0].condition == "spa"
    assert deltas[0].d_tm_mean is not None                    # headline Δ(SPA − baseline) TM present


def test_results_json_written(result):
    import json

    path = result["results_path"]
    assert path.exists()
    payload = json.loads(path.read_text())
    assert {"scores", "summaries", "deltas"} <= payload.keys()
    assert len(payload["scores"]) == len(result["designs"])
    assert len(payload["deltas"]) == 1
