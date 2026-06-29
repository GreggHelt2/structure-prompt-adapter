"""Stage-4 scoring smoke test (dev 05_validation_pipeline.md §3 / 06_experiments.md §6).

Exercises ``spa.eval.score`` end-to-end on CPU with tiny synthetic Cα backbones — no RFD3, no
ProteinMPNN, no OF3, no CUDA. Three things are checked, matching the dev task spec:

1. **TM / RMSD math** — a structure vs itself gives TM ≈ 1.0 and Cα-RMSD ≈ 0; a rigidly
   rotated+translated copy is unchanged (superposition-invariant); a perturbed copy degrades
   monotonically (TM < 1, RMSD > 0) but stays bounded.
2. **scRMSD designability** — best-of-K self-consistency against **stand-in refolds** (a good +
   a bad copy) picks the good refold and flips the ``designable`` flag at the cutoff; the optional
   pLDDT gate also flips it.
3. **Aggregation** — a small synthetic :class:`DesignScore` set rolls up into per-(condition, λ)
   success rate + distributions, and Δ(SPA − baseline).

Gated like the other eval tests: skipped unless tmtools + biotite + numpy import (so the suite stays
green where the scoring deps aren't installed). CPU-only and sub-second by construction.
"""

import math

import pytest


def _have_deps() -> bool:
    try:
        import biotite  # noqa: F401
        import numpy  # noqa: F401
        import tmtools  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _have_deps(), reason="tmtools / biotite / numpy not available")


# --------------------------------------------------------------------------------------------------
# Tiny synthetic Cα backbones (no RFD3 needed — geometry only matters up to superposition)
# --------------------------------------------------------------------------------------------------


def _make_ca(coords):
    """A one-Cα-per-residue poly-G ``AtomArray`` from an ``[N, 3]`` coord array."""
    from biotite.structure import Atom, array

    atoms = [
        Atom(coords[i], chain_id="A", res_id=i + 1, res_name="GLY", atom_name="CA", element="C")
        for i in range(len(coords))
    ]
    return array(atoms)


def _backbone(n=24, seed=0):
    """A non-degenerate random-walk Cα trace (well-defined for TM-align / Kabsch)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(n, 3)), axis=0).astype(np.float64)


def _rotate_translate(coords, theta=0.7, shift=(5.0, -2.0, 3.0)):
    import numpy as np

    R = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                  [math.sin(theta), math.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    return coords @ R.T + np.array(shift)


def _perturb(coords, sigma, seed=1):
    import numpy as np

    rng = np.random.default_rng(seed)
    return coords + rng.normal(scale=sigma, size=coords.shape)


# --------------------------------------------------------------------------------------------------
# 1. TM-score / Cα-RMSD math
# --------------------------------------------------------------------------------------------------


def test_tm_and_rmsd_self_is_perfect():
    from spa.eval.score import ca_rmsd, tm_score

    s = _make_ca(_backbone())
    tm_d, tm_p = tm_score(s, s)
    assert tm_d == pytest.approx(1.0, abs=1e-3)
    assert tm_p == pytest.approx(1.0, abs=1e-3)
    assert ca_rmsd(s, s) == pytest.approx(0.0, abs=1e-4)


def test_superposition_invariant():
    # A rigid rotation + translation must leave TM ≈ 1 and post-superpose RMSD ≈ 0.
    from spa.eval.score import ca_rmsd, tm_score

    coords = _backbone()
    a = _make_ca(coords)
    b = _make_ca(_rotate_translate(coords))
    tm_d, tm_p = tm_score(a, b)
    assert tm_d == pytest.approx(1.0, abs=1e-3)
    assert ca_rmsd(a, b) == pytest.approx(0.0, abs=1e-4)


def test_perturbation_degrades_monotonically():
    from spa.eval.score import ca_rmsd, tm_score

    coords = _backbone()
    a = _make_ca(coords)
    small = _make_ca(_perturb(coords, sigma=0.3))
    large = _make_ca(_perturb(coords, sigma=2.0))

    rmsd_small = ca_rmsd(a, small)
    rmsd_large = ca_rmsd(a, large)
    assert 0.0 < rmsd_small < rmsd_large

    tm_small, _ = tm_score(a, small)
    tm_large, _ = tm_score(a, large)
    assert tm_large < tm_small <= 1.0


def test_adherence_reports_both_norms_and_prompt_rmsd():
    from spa.eval.score import adherence

    coords = _backbone()
    design = _make_ca(_perturb(coords, sigma=0.3))
    prompt = _make_ca(coords)
    adh = adherence(design, prompt, tm_norm="prompt")
    # equal length -> the two normalizations coincide, and prompt-RMSD is defined
    assert adh.n_design == adh.n_prompt == len(coords)
    assert adh.tm_norm_design == pytest.approx(adh.tm_norm_prompt, abs=1e-6)
    assert adh.tm_score == pytest.approx(adh.tm_norm_prompt, abs=1e-6)
    assert adh.prompt_rmsd is not None and adh.prompt_rmsd > 0.0


def test_adherence_prompt_rmsd_none_on_length_mismatch():
    from spa.eval.score import adherence

    design = _make_ca(_backbone(n=24, seed=0))
    prompt = _make_ca(_backbone(n=30, seed=2))  # different length -> no 1:1 correspondence
    adh = adherence(design, prompt)
    assert adh.prompt_rmsd is None
    assert 0.0 <= adh.tm_score <= 1.0


# --------------------------------------------------------------------------------------------------
# 2. scRMSD / designability against stand-in refolds
# --------------------------------------------------------------------------------------------------


def test_best_of_k_designable_against_standin_refold():
    from spa.eval.score import is_designable, self_consistency

    coords = _backbone()
    design = _make_ca(coords)
    good = _make_ca(_perturb(coords, sigma=0.3))   # < 2 Å scRMSD
    bad = _make_ca(_perturb(coords, sigma=3.0))    # >> 2 Å scRMSD

    res = self_consistency(design, [bad, good])
    assert res.best_refold_idx == 1                 # best-of-K picks the good refold
    assert res.scrmsd < 2.0
    assert is_designable(res.scrmsd, scrmsd_cutoff=2.0) is True

    # A pool of only-bad refolds is not designable.
    res_bad = self_consistency(design, [bad])
    assert res_bad.scrmsd >= 2.0
    assert is_designable(res_bad.scrmsd, scrmsd_cutoff=2.0) is False


def test_length_mismatched_refold_is_skipped():
    from spa.eval.score import self_consistency

    design = _make_ca(_backbone(n=24))
    mismatched = _make_ca(_backbone(n=20, seed=5))   # contributes NaN, skipped
    good = _make_ca(_perturb(_backbone(n=24), sigma=0.3))
    res = self_consistency(design, [mismatched, good])
    assert math.isnan(res.per_refold[0])             # mismatch -> NaN
    assert res.best_refold_idx == 1                   # falls through to the valid refold


def test_plddt_gate_flips_designability():
    from spa.eval.score import is_designable

    assert is_designable(1.0, plddt=90.0, scrmsd_cutoff=2.0, plddt_cutoff=80.0) is True
    assert is_designable(1.0, plddt=70.0, scrmsd_cutoff=2.0, plddt_cutoff=80.0) is False  # low pLDDT
    assert is_designable(float("nan"), scrmsd_cutoff=2.0) is False
    assert is_designable(None, scrmsd_cutoff=2.0) is False


def test_score_design_composes_adherence_and_designability():
    from types import SimpleNamespace

    from spa.eval.score import score_design

    coords = _backbone()
    design = SimpleNamespace(
        atom_array=_make_ca(coords), path="x/spa_design_0.pdb",
        condition="spa", lambda_scale=1.0, n_residues=len(coords),
    )
    prompt = _make_ca(coords)
    refolds = [_make_ca(_perturb(coords, sigma=0.3))]
    ds = score_design(design, prompt=prompt, refolds=refolds)
    assert ds.name == "spa_design_0" and ds.condition == "spa" and ds.lambda_scale == 1.0
    assert ds.tm_score is not None and ds.prompt_rmsd is not None
    assert ds.scrmsd is not None and ds.designable is True


# --------------------------------------------------------------------------------------------------
# 3. Aggregation + Δ(SPA − baseline)
# --------------------------------------------------------------------------------------------------


def _synthetic_scores():
    from spa.eval.score import DesignScore

    def mk(name, cond, lam, scrmsd, tm, designable):
        return DesignScore(
            name=name, condition=cond, lambda_scale=lam, n_residues=50,
            tm_score=tm, tm_norm_prompt=tm, scrmsd=scrmsd, designable=designable,
        )

    return [
        # baseline (λ=0): 1 of 2 designable, low adherence
        mk("b0", "baseline", 0.0, 1.5, 0.30, True),
        mk("b1", "baseline", 0.0, 3.0, 0.25, False),
        # spa λ=1.0: both designable, high adherence
        mk("s0", "spa", 1.0, 1.0, 0.80, True),
        mk("s1", "spa", 1.0, 1.2, 0.75, True),
    ]


def test_aggregate_rate_and_distribution():
    from spa.eval.score import aggregate

    summaries = aggregate(_synthetic_scores())
    by_key = {(s.condition, s.lambda_scale): s for s in summaries}

    base = by_key[("baseline", 0.0)]
    assert base.n_designs == 2 and base.n_designable == 1
    assert base.success_rate == pytest.approx(0.5)
    assert base.tm.mean == pytest.approx(0.275)

    spa = by_key[("spa", 1.0)]
    assert spa.success_rate == pytest.approx(1.0)
    assert spa.tm.mean == pytest.approx(0.775)
    assert spa.scrmsd.min == pytest.approx(1.0)


def test_delta_vs_baseline():
    from spa.eval.score import aggregate, delta_vs_baseline

    deltas = delta_vs_baseline(aggregate(_synthetic_scores()))
    assert len(deltas) == 1
    d = deltas[0]
    assert d.condition == "spa" and d.lambda_scale == 1.0
    assert d.d_success_rate == pytest.approx(0.5)         # 1.0 - 0.5
    assert d.d_tm_mean == pytest.approx(0.5)              # 0.775 - 0.275
    assert d.spa_tm_mean == pytest.approx(0.775)
    assert d.baseline_tm_mean == pytest.approx(0.275)


def test_aggregate_diversity_among_designable():
    from spa.eval.score import DesignScore, aggregate

    coords = _backbone()
    structs = {
        "d0": _make_ca(coords),
        "d1": _make_ca(_perturb(coords, sigma=0.5)),
        "d2": _make_ca(_perturb(coords, sigma=0.7, seed=9)),
    }
    scores = [
        DesignScore(name="d0", condition="spa", lambda_scale=1.0, n_residues=24, scrmsd=1.0, designable=True),
        DesignScore(name="d1", condition="spa", lambda_scale=1.0, n_residues=24, scrmsd=1.1, designable=True),
        DesignScore(name="d2", condition="spa", lambda_scale=1.0, n_residues=24, scrmsd=1.2, designable=True),
    ]
    summaries = aggregate(scores, structs_by_name=structs)
    assert len(summaries) == 1
    assert summaries[0].diversity_tm is not None
    assert 0.0 <= summaries[0].diversity_tm <= 1.0


# --------------------------------------------------------------------------------------------------
# 4. Motif satisfaction (Run-B hard⊕soft; dev 14 §3)
# --------------------------------------------------------------------------------------------------

_MOTIF = [8, 9, 10, 11, 12, 13]   # a 6-residue contiguous motif (0-based positional indices)


def test_motif_rmsd_self_and_superposition_invariant():
    from spa.eval.score import motif_rmsd

    coords = _backbone()
    source = _make_ca(coords)
    assert motif_rmsd(source, source, _MOTIF) == pytest.approx(0.0, abs=1e-4)
    # a rigid rotation+translation of the whole design leaves the motif RMSD ≈ 0 (Kabsch over the motif)
    rotated = _make_ca(_rotate_translate(coords))
    assert motif_rmsd(rotated, source, _MOTIF) == pytest.approx(0.0, abs=1e-4)


def test_motif_rmsd_ignores_scaffold():
    # Perturb the whole backbone, then restore ONLY the motif rows -> motif RMSD ≈ 0 even though the
    # scaffold moved a lot. Proves the metric superposes/scores over the motif subset alone.
    from spa.eval.score import motif_rmsd

    coords = _backbone()
    pert = _perturb(coords, sigma=2.0)
    pert[_MOTIF] = coords[_MOTIF]
    design = _make_ca(pert)
    source = _make_ca(coords)
    assert motif_rmsd(design, source, _MOTIF) == pytest.approx(0.0, abs=1e-4)


def test_motif_rmsd_detects_motif_break():
    # Perturb ONLY the motif rows -> motif RMSD clearly > 0 (the hard constraint was violated).
    from spa.eval.score import motif_rmsd

    coords = _backbone()
    pert = coords.copy()
    pert[_MOTIF] = _perturb(coords[_MOTIF], sigma=2.0, seed=3)
    design = _make_ca(pert)
    source = _make_ca(coords)
    assert motif_rmsd(design, source, _MOTIF) > 0.5


def test_motif_rmsd_raises_on_bad_indices():
    from spa.eval.score import motif_rmsd

    s = _make_ca(_backbone(n=24))
    with pytest.raises(ValueError):
        motif_rmsd(s, s, [])                              # empty
    with pytest.raises(ValueError):
        motif_rmsd(s, s, [8, 999])                        # out of range
    with pytest.raises(ValueError):
        motif_rmsd(s, s, [1, 2, 3], source_residues=[1, 2])  # count mismatch


def test_score_design_with_motif():
    from types import SimpleNamespace

    from spa.eval.score import score_design

    coords = _backbone()
    design = SimpleNamespace(
        atom_array=_make_ca(coords), path="x/spa_design_0.pdb",
        condition="spa", lambda_scale=1.0, n_residues=len(coords),
    )
    source = _make_ca(coords)
    refolds = [_make_ca(_perturb(coords, sigma=0.3))]
    ds = score_design(design, prompt=source, refolds=refolds, motif=(source, _MOTIF))
    # design == source over the motif -> design-side motif_rmsd ≈ 0, satisfied
    assert ds.motif_rmsd == pytest.approx(0.0, abs=1e-4)
    assert ds.motif_satisfied is True
    # refold-side computed off the best refold (perturbed copy) -> finite, ≥ 0
    assert ds.motif_rmsd_refold is not None and ds.motif_rmsd_refold >= 0.0
    # non-motif fields still populated; a no-motif design leaves all three None
    assert ds.scrmsd is not None and ds.tm_score is not None
    ds_nomotif = score_design(design, prompt=source, refolds=refolds)
    assert ds_nomotif.motif_rmsd is None and ds_nomotif.motif_satisfied is None
    assert ds_nomotif.motif_rmsd_refold is None


def test_aggregate_and_delta_motif():
    # Two conditions, both with the motif pinned (motif_rmsd ~0) -> satisfied rate 1.0 and
    # d_motif_rmsd_mean ≈ 0 (the hard pin is SPA-independent — the headline claim).
    from spa.eval.score import DesignScore, aggregate, delta_vs_baseline

    def mk(name, cond, lam, mrmsd):
        return DesignScore(
            name=name, condition=cond, lambda_scale=lam, n_residues=50,
            tm_score=0.3, scrmsd=1.0, designable=True,
            motif_rmsd=mrmsd, motif_satisfied=bool(mrmsd < 1.0),
        )

    scores = [
        mk("b0", "baseline", 0.0, 0.05), mk("b1", "baseline", 0.0, 0.07),
        mk("s0", "spa", 1.0, 0.06), mk("s1", "spa", 1.0, 0.08),
    ]
    summaries = aggregate(scores)
    by = {(s.condition, s.lambda_scale): s for s in summaries}
    assert by[("baseline", 0.0)].motif_satisfied_rate == pytest.approx(1.0)
    assert by[("baseline", 0.0)].motif_rmsd.mean == pytest.approx(0.06)
    assert by[("spa", 1.0)].motif_rmsd.mean == pytest.approx(0.07)

    deltas = delta_vs_baseline(summaries)
    assert deltas[0].d_motif_rmsd_mean == pytest.approx(0.01, abs=1e-6)  # 0.07 - 0.06, ≈ 0

    # A purely non-motif aggregate leaves the motif rollups empty/None (no regression).
    plain = aggregate(_synthetic_scores())
    base = {(s.condition, s.lambda_scale): s for s in plain}[("baseline", 0.0)]
    assert base.motif_satisfied_rate is None and base.motif_rmsd.n == 0
