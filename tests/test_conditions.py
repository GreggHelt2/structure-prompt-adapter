"""Unit tests for the coupled-island motif sampler (dev ``10`` §7.3).

The pure-numpy distribution tests always run; the wiring test (that ``build_train_transform`` with
``coupled_motif=True`` actually instantiates a :class:`CoupledIslandCondition`) needs the RFD3 ckpt +
foundry and is skipped otherwise.
"""
import collections
import os

import numpy as np
import pytest

from spa.data.conditions import CoupledIslandCondition

_ROOT = os.environ.get("SPA_PROJECT_ROOT", os.path.expanduser("~/projects/spa"))
_CKPT = os.path.join(_ROOT, "models/rfdiffusion3/rfd3_latest.ckpt")
_FOUNDRY_CFG = os.path.join(_ROOT, "needed_repos/foundry/models/rfd3/configs/datasets/train")


def _cond(**over):
    kw = dict(
        name="island", frequency=1.0,
        island_sampling_kwargs={"island_len_min": 1, "island_len_max": 12, "n_islands_min": 2, "n_islands_max": 5},
        p_diffuse_motif_sidechains=0.8, p_diffuse_subgraph_atoms=0.0, subgraph_sampling_kwargs={},
        p_fix_motif_coordinates=1.0, p_fix_motif_sequence=0.0, p_unindex_motif_tokens=0.0,
    )
    kw.update(over)
    return CoupledIslandCondition(**kw)


def _seglens(m):
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return (np.where(d == -1)[0] - np.where(d == 1)[0]).tolist()


def test_bounds_no_degenerate_and_full_range():
    c = _cond()
    for n in (32, 50, 100, 150, 256, 384):
        ks = collections.Counter()
        for _ in range(2000):
            m = c._sample_coupled_islands(n)
            s = _seglens(m)
            assert all(L >= c.island_floor for L in s), f"island < floor at n={n}: {s}"   # no 1-2 res motifs
            assert 1 <= len(s) <= 5, f"n_islands out of [1,5] at n={n}: {len(s)}"
            assert int(m.sum()) <= min(c.motif_abs_max, n - 1)                              # leaves scaffold
            ks[len(s)] += 1
        assert ks[1] > 0 and ks[5] > 0, f"expected the full 1..5 island range at n={n}: {dict(ks)}"


def test_weighted_toward_fewer_islands():
    c = _cond()
    ks = collections.Counter(len(_seglens(c._sample_coupled_islands(150))) for _ in range(6000))
    assert ks[1] > ks[2] > ks[3] > ks[4] >= ks[5], f"not monotone toward fewer: {dict(ks)}"


def test_total_scales_with_protein_length():
    c = _cond()
    frac = lambda n: np.mean([c._sample_coupled_islands(n).sum() for _ in range(2000)]) / n  # noqa: E731
    assert frac(64) > frac(384)  # small proteins floored higher %, large ones capped (abs 50)


@pytest.mark.skipif(not os.path.exists(_CKPT) or not os.path.isdir(_FOUNDRY_CFG),
                    reason="needs the RFD3 ckpt + foundry train cfgs")
def test_wired_into_transform():
    """coupled_motif=True must actually put a CoupledIslandCondition in the built pipeline (not stock)."""
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

    tf_coupled = build_train_transform(_CKPT, _FOUNDRY_CFG, conditioning="island", coupled_motif=True)
    assert find(tf_coupled, CoupledIslandCondition), "coupled_motif=True did not wire in CoupledIslandCondition"
    tf_stock = build_train_transform(_CKPT, _FOUNDRY_CFG, conditioning="island", coupled_motif=False)
    assert not find(tf_stock, CoupledIslandCondition), "coupled_motif=False should use the stock sampler"
