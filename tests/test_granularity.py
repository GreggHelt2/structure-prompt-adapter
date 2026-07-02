"""Unit tests for the multi-granularity sub-region sampler (dev ``17`` §3–§5 / ``16`` §9.5–§9.7).

Pure-numpy tests (granularity mix, segment windows, domain split via a synthetic 2-domain structure,
mask coverage, determinism) always run. The transform-wiring test (that ``conditioning=multigranularity``
composes an UNCONDITIONAL native pipeline — no revealed motif) needs the RFD3 ckpt + foundry and is
skipped otherwise.
"""
import collections
import os

import numpy as np
import pytest

from spa.data.granularity import (
    GRANULARITIES,
    detect_two_domains,
    domain_region,
    sample_granularity,
    segment_region,
    subregion_pad_mask,
)

_ROOT = os.environ.get("SPA_PROJECT_ROOT", os.path.expanduser("~/projects/spa"))
_CKPT = os.path.join(_ROOT, "models/rfdiffusion3/rfd3_latest.ckpt")
_FOUNDRY_CFG = os.path.join(_ROOT, "needed_repos/foundry/models/rfd3/configs/datasets/train")


# --------------------------------------------------------------------------------------------------
# granularity sampling
# --------------------------------------------------------------------------------------------------

def test_sample_granularity_single_weight():
    r = np.random.RandomState(0)
    for name in GRANULARITIES:
        assert all(sample_granularity({name: 1.0}, r) == name for _ in range(50))


def test_sample_granularity_mix_matches_weights():
    r = np.random.RandomState(1)
    c = collections.Counter(sample_granularity({"global": 0.4, "segment": 0.4, "domain": 0.2}, r)
                            for _ in range(20000))
    frac = {k: v / 20000 for k, v in c.items()}
    assert abs(frac["global"] - 0.4) < 0.03 and abs(frac["segment"] - 0.4) < 0.03
    assert abs(frac["domain"] - 0.2) < 0.03


def test_sample_granularity_accepts_sequence_and_rejects_zero():
    r = np.random.RandomState(2)
    assert sample_granularity((1.0, 0.0, 0.0), r) == "global"
    with pytest.raises(ValueError):
        sample_granularity({"global": 0.0, "segment": 0.0, "domain": 0.0}, r)


# --------------------------------------------------------------------------------------------------
# segment windows
# --------------------------------------------------------------------------------------------------

def test_segment_region_bounds():
    r = np.random.RandomState(3)
    for n in (20, 50, 128, 256):
        for _ in range(500):
            s, e = segment_region(n, 12, r)
            assert 0 <= s < e <= n
            assert (e - s) >= 12 or (s == 0 and e == n)  # >= min_seg, unless the whole structure


def test_segment_region_short_is_whole():
    r = np.random.RandomState(4)
    assert segment_region(8, 12, r) == (0, 8)  # n <= min_seg -> whole (== global)


# --------------------------------------------------------------------------------------------------
# domain split (synthetic 2-domain structure)
# --------------------------------------------------------------------------------------------------

def _write_ca_pdb(path, coords):
    """Write a CA-only PDB (chain A, GLY, sequential res ids) from an ``[N,3]`` array via biotite."""
    import biotite.structure as struc
    from biotite.structure.io import save_structure

    atoms = [
        struc.Atom(list(map(float, xyz)), chain_id="A", res_id=i + 1, res_name="GLY",
                   atom_name="CA", element="C")
        for i, xyz in enumerate(coords)
    ]
    save_structure(str(path), struc.array(atoms))


def _two_domain_coords(n1, n2, sep=100.0, seed=0):
    """Two compact random blobs (indices 0..n1-1, then n1..n1+n2-1) separated by ``sep`` Å."""
    r = np.random.RandomState(seed)
    d1 = r.uniform(0, 10, size=(n1, 3))                      # blob near origin
    d2 = r.uniform(0, 10, size=(n2, 3)) + np.array([sep, 0, 0])  # blob far along +x
    return np.concatenate([d1, d2], axis=0)


def test_detect_two_domains_clean_split(tmp_path):
    n1, n2 = 60, 50
    pdb = tmp_path / "twodom.pdb"
    _write_ca_pdb(pdb, _two_domain_coords(n1, n2))
    info = detect_two_domains(str(pdb), contact_thresh=8.0, min_dom=40)
    assert info["n_res"] == n1 + n2
    assert info["boundary"] is not None
    assert abs(info["boundary"] - n1) <= 2          # boundary lands at the true inter-blob split
    assert info["score"] < 0.05                     # ~0 cross-domain contacts -> very clean


def test_detect_two_domains_too_short(tmp_path):
    pdb = tmp_path / "short.pdb"
    _write_ca_pdb(pdb, _two_domain_coords(20, 20))  # n=40 < 2*min_dom=80
    info = detect_two_domains(str(pdb), min_dom=40)
    assert info["boundary"] is None and info["score"] is None


def test_domain_region_picks_one_domain(tmp_path):
    n1, n2 = 60, 50
    n = n1 + n2
    pdb = tmp_path / "twodom.pdb"
    _write_ca_pdb(pdb, _two_domain_coords(n1, n2))
    seen = set()
    for seed in range(20):
        reg = domain_region(str(pdb), n, min_dom=40, domain_score_max=0.4,
                            rng=np.random.RandomState(seed))
        assert reg is not None
        s, e = reg
        # each pick is exactly one of the two domains (boundary ~ n1)
        assert (s == 0 and abs(e - n1) <= 2) or (abs(s - n1) <= 2 and e == n)
        seen.add(s == 0)
    assert seen == {True, False}                    # both domains selected across seeds


def test_domain_region_none_on_length_mismatch(tmp_path):
    pdb = tmp_path / "twodom.pdb"
    _write_ca_pdb(pdb, _two_domain_coords(60, 50))  # n_ca = 110
    # prompt length disagrees with Cα count -> boundary can't map to prompt rows -> fall back
    assert domain_region(str(pdb), n=99, min_dom=40, rng=np.random.RandomState(0)) is None


# --------------------------------------------------------------------------------------------------
# subregion_pad_mask — the dataset entry point
# --------------------------------------------------------------------------------------------------

def test_pad_mask_global_is_none():
    r = np.random.RandomState(5)
    g, pad = subregion_pad_mask(100, weights={"global": 1.0}, rng=r)
    assert g == "global" and pad is None


def test_pad_mask_segment_masks_complement():
    r = np.random.RandomState(6)
    for _ in range(200):
        g, pad = subregion_pad_mask(120, weights={"segment": 1.0}, min_seg=12, rng=r)
        if pad is None:                              # sampled window covered all -> global
            assert g == "global"
            continue
        assert g == "segment" and pad.dtype == np.bool_ and pad.shape == (120,)
        kept = ~pad                                  # S = kept (unmasked) rows
        assert kept.any() and pad.any()             # S non-empty AND something masked (well-defined)
        idx = np.where(kept)[0]
        assert (np.diff(idx) == 1).all()            # S is a single contiguous window
        assert kept.sum() >= 12                     # >= min_seg residues kept


def test_pad_mask_domain(tmp_path):
    n1, n2 = 60, 50
    n = n1 + n2
    pdb = tmp_path / "twodom.pdb"
    _write_ca_pdb(pdb, _two_domain_coords(n1, n2))
    g, pad = subregion_pad_mask(n, weights={"domain": 1.0}, min_dom=40, domain_score_max=0.4,
                                pdb_path=str(pdb), rng=np.random.RandomState(0))
    assert g == "domain" and pad is not None
    kept = ~pad
    assert kept.any() and pad.any()
    idx = np.where(kept)[0]
    assert (np.diff(idx) == 1).all()                # a contiguous domain


def test_pad_mask_domain_falls_back_to_segment_when_short(tmp_path):
    pdb = tmp_path / "short.pdb"
    _write_ca_pdb(pdb, _two_domain_coords(20, 20))  # too short for a domain split
    g, pad = subregion_pad_mask(40, weights={"domain": 1.0}, min_seg=12, min_dom=40,
                                pdb_path=str(pdb), rng=np.random.RandomState(0))
    assert g in ("segment", "global")               # fell back off the domain path
    if pad is not None:
        assert (~pad).sum() >= 12 and pad.any()


def test_pad_mask_domain_falls_back_without_path():
    # domain requested but no pdb_path -> must fall back, never crash
    g, pad = subregion_pad_mask(120, weights={"domain": 1.0}, min_seg=12,
                                pdb_path=None, rng=np.random.RandomState(0))
    assert g in ("segment", "global")


def test_pad_mask_deterministic_under_seed():
    a = subregion_pad_mask(120, weights={"segment": 1.0}, rng=np.random.RandomState(7))
    b = subregion_pad_mask(120, weights={"segment": 1.0}, rng=np.random.RandomState(7))
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


# --------------------------------------------------------------------------------------------------
# native-condition selection: multigranularity == unconditional (SPA does the sub-region conditioning)
# --------------------------------------------------------------------------------------------------

def test_select_conditions_multigranularity_equals_unconditional():
    """multigranularity's RFD3-native conditioning must be identical to unconditional (no revealed
    motif) — the sub-region conditioning is SPA-side (prompt mask), not native. Pure dict selection."""
    from spa.data.dataset import _select_conditions

    all_c = {
        "unconditional": {"name": "unconditional", "frequency": 0.5},
        "island": {"name": "island", "frequency": 0.5, "island_sampling_kwargs": {}},
    }
    mg = _select_conditions({k: dict(v) for k, v in all_c.items()}, "multigranularity")
    uncond = _select_conditions({k: dict(v) for k, v in all_c.items()}, "unconditional")
    assert mg == uncond == {"unconditional": {"name": "unconditional", "frequency": 0.5}}
    assert "island" not in mg  # never reveals a native motif


@pytest.mark.skipif(not os.path.exists(_CKPT) or not os.path.isdir(_FOUNDRY_CFG),
                    reason="needs the RFD3 ckpt + foundry train cfgs")
def test_multigranularity_transform_has_no_coupled_island():
    """The composed multigranularity pipeline must NOT wire in our Run-B CoupledIslandCondition
    (that sampler is added only for conditioning=island/mixed). RFD3's stock IslandCondition may still
    appear in the inert ConditionalRoute machinery — it's frequency-gated off — so we only assert the
    coupled sampler's absence."""
    from spa.data.conditions import CoupledIslandCondition
    from spa.data.dataset import build_train_transform

    def find(obj, cls, seen=None, depth=0):
        if obj is None or depth > 8:
            return []
        seen = seen if seen is not None else set()
        if id(obj) in seen:
            return []
        seen.add(id(obj))
        hits = [obj] if isinstance(obj, cls) else []
        if isinstance(obj, (list, tuple, set)):
            for x in obj:
                hits += find(x, cls, seen, depth + 1)
        elif isinstance(obj, dict):
            for x in obj.values():
                hits += find(x, cls, seen, depth + 1)
        elif hasattr(obj, "__dict__"):
            for x in vars(obj).values():
                hits += find(x, cls, seen, depth + 1)
        return hits

    tf = build_train_transform(_CKPT, _FOUNDRY_CFG, conditioning="multigranularity")
    assert not find(tf, CoupledIslandCondition)
