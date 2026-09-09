"""Tests for the opt-in bitwise-determinism shim (dev ``plan/91``).

The GPU end-to-end proof (two independent processes, byte-identical PDBs) is a several-minute run and
lives in ``docs/results``; these are the fast invariants that must never regress:

1. the replacement is numerically equivalent to what it replaces,
2. it is order-stable where the original is not,
3. it refuses a call shape it does not actually implement,
4. **the toggle is off by default and patches nothing at import**, which is the property that makes
   an unset run byte-identical to not having this code at all.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spa.eval import determinism as det


def _inputs(n_atoms=400, n_tokens=50, chans=32, device="cpu", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(0)
    source = torch.randn(1, n_atoms, chans, generator=g).to(device=device, dtype=dtype)
    index = torch.sort(torch.randint(0, n_tokens, (n_atoms,), generator=g))[0].to(device)
    zeros = torch.zeros(1, n_tokens, chans, device=device, dtype=dtype)
    return zeros, index, source


def test_matches_index_reduce():
    """Same values as the op it replaces, to fp32 tolerance."""
    zeros, index, source = _inputs()
    got = det.deterministic_scatter_mean(zeros, -2, index, source)
    want = zeros.index_reduce(-2, index, source, "mean", include_self=False)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_empty_token_yields_zero():
    """A token with no atoms must give 0, matching ``include_self=False`` on a zero-filled output."""
    source = torch.ones(1, 4, 3)
    index = torch.tensor([0, 0, 2, 2])  # token 1 gets nothing
    zeros = torch.zeros(1, 3, 3)
    got = det.deterministic_scatter_mean(zeros, -2, index, source)
    assert torch.equal(got[0, 1], torch.zeros(3))
    torch.testing.assert_close(got, zeros.index_reduce(-2, index, source, "mean", include_self=False))


def test_does_not_mutate_zeros():
    zeros, index, source = _inputs()
    before = zeros.clone()
    det.deterministic_scatter_mean(zeros, -2, index, source)
    assert torch.equal(zeros, before)


def test_rejects_unimplemented_dim():
    """It only implements the token axis. Anything else must raise, never silently misbehave."""
    zeros, index, source = _inputs()
    with pytest.raises(NotImplementedError, match="token axis"):
        det.deterministic_scatter_mean(zeros, 0, index, source)


def test_bit_stable_over_repeats():
    """Order-stability, on whatever device is available. On CUDA this is the whole point: the op it
    replaces gives a different bit pattern on nearly every call."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zeros, index, source = _inputs(device=device)
    outs = [det.deterministic_scatter_mean(zeros, -2, index, source) for _ in range(20)]
    assert len({o.cpu().numpy().tobytes() for o in outs}) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_original_op_is_actually_nondeterministic_on_cuda():
    """Guards the premise. If a future torch ships a deterministic ``index_reduce_cuda`` this test
    fails, which is the signal to re-evaluate whether the shim is still needed at all."""
    zeros, index, source = _inputs(device="cuda")
    outs = [zeros.index_reduce(-2, index, source, "mean", include_self=False) for _ in range(20)]
    assert len({o.cpu().numpy().tobytes() for o in outs}) > 1


def test_off_by_default_patches_nothing_at_import():
    """⛔ The load-bearing toggle property: importing this module must not change RFD3's behaviour."""
    assert det.state()["enabled"] is False
    assert det.state()["patched"] == []


def test_maybe_enable_is_a_noop_when_unset():
    class _Eval(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    class _Cfg:
        eval = _Eval()

    assert det.maybe_enable(_Cfg()).get("enabled") is False
