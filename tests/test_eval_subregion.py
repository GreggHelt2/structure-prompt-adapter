"""Unit tests for the sub-region "scaffolding" eval wiring (dev 17 §7 / 16 §9.5).

CPU-only — exercises the two new pure helpers in ``spa.eval.generate`` (``subregion_keep`` +
``subregion_key_padding_mask``) and the score-side path (``score_design`` with a self-aligned
sub-region motif tuple, ``design[S]`` vs ``prompt[S]``). No RFD3 engine / GPU needed; the GPU
end-to-end run is exercised separately by ``scripts/eval/run_scaffold_eval.py``.

Guardrails covered: mask polarity (``True`` at rows ∉ S), the all-of-N degenerate (→ no mask),
index-range guards, keep dedup/sort, and that a well-aligned S region scores ≈0 sub-region
motif-RMSD while the free region does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


# --------------------------------------------------------------------------------------------------
# subregion_keep — parse eval.subregion.keep into a sorted, de-duped index list (or None)
# --------------------------------------------------------------------------------------------------


def _cfg(subregion):
    return OmegaConf.create({"eval": {"subregion": subregion}})


def test_subregion_keep_none_when_unset():
    from spa.eval.generate import subregion_keep

    assert subregion_keep(_cfg(None)) is None
    assert subregion_keep(OmegaConf.create({"eval": {}})) is None


def test_subregion_keep_sorts_and_dedups():
    from spa.eval.generate import subregion_keep

    assert subregion_keep(_cfg({"keep": [5, 1, 3, 1, 2]})) == [1, 2, 3, 5]


def test_subregion_keep_empty_raises():
    from spa.eval.generate import subregion_keep

    with pytest.raises(ValueError, match="empty"):
        subregion_keep(_cfg({"keep": []}))


def test_subregion_keep_missing_keep_raises():
    from spa.eval.generate import subregion_keep

    with pytest.raises(ValueError, match="no `keep`"):
        subregion_keep(_cfg({"foo": 1}))


# --------------------------------------------------------------------------------------------------
# subregion_key_padding_mask — [K,N] bool, True at rows ∉ S (masked); None when S spans all N
# --------------------------------------------------------------------------------------------------


def test_mask_polarity_true_off_S():
    from spa.eval.generate import subregion_key_padding_mask

    mask = subregion_key_padding_mask([0, 1, 2], N=5, K=2, device=torch.device("cpu"))
    assert mask.shape == (2, 5) and mask.dtype == torch.bool
    # S = {0,1,2} attended (False); complement {3,4} masked (True)
    assert mask[0].tolist() == [False, False, False, True, True]
    assert torch.equal(mask[0], mask[1])                      # same mask broadcast over the batch


def test_mask_none_when_keep_spans_all():
    from spa.eval.generate import subregion_key_padding_mask

    assert subregion_key_padding_mask([0, 1, 2, 3], N=4, K=3, device=torch.device("cpu")) is None


def test_mask_out_of_range_raises():
    from spa.eval.generate import subregion_key_padding_mask

    with pytest.raises(ValueError, match="out of range"):
        subregion_key_padding_mask([0, 5], N=5, K=1, device=torch.device("cpu"))   # index 5 == N


# --------------------------------------------------------------------------------------------------
# Score wiring — score_design(motif=(prompt_struct, keep)) computes sub-region motif-RMSD over S,
# self-aligned (design length == prompt length, S indexes both). A well-aligned S scores ≈0.
# --------------------------------------------------------------------------------------------------


def _ca(coords):
    """A CA-only biotite AtomArray from an [N,3] coord array (GLY residues, chain A, 1..N)."""
    from biotite.structure import AtomArray

    n = len(coords)
    arr = AtomArray(n)
    arr.coord = np.asarray(coords, dtype="float32")
    arr.chain_id = np.array(["A"] * n)
    arr.res_id = np.arange(1, n + 1)
    arr.res_name = np.array(["GLY"] * n)
    arr.atom_name = np.array(["CA"] * n)
    arr.element = np.array(["C"] * n)
    return arr


def test_score_design_subregion_motif_rmsd():
    from spa.eval.score import score_design

    rng = np.random.RandomState(0)
    N = 30
    keep = list(range(0, 12))                                 # S = first 12 residues
    prompt = rng.randn(N, 3) * 8.0

    # Design: S region = a rigid-body copy of prompt[S] (rotate + translate) -> Kabsch RMSD ≈ 0;
    # F region = unrelated random coords.
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    design = rng.randn(N, 3) * 8.0                            # F rows unrelated
    design[keep] = prompt[keep] @ R.T + np.array([10.0, -5.0, 3.0])

    ds = score_design(_ca(design), prompt=_ca(prompt), motif=(_ca(prompt), keep))
    assert ds.motif_rmsd is not None
    assert ds.motif_rmsd < 1e-3                               # S region rigid-body matches -> ≈0
    assert ds.motif_satisfied is True                        # < 1.0 Å cutoff
    # adherence over the whole design is scored too (prompt supplied)
    assert ds.tm_score is not None
