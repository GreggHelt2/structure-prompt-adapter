"""Unit tests for the SPA cross-attention module (kickoff step 2).

Module-level checks of the contract from dev ``03`` §2/§5: output shape, the identity-at-init
property (zero-init ``Wo`` -> exactly-zero term, the local reflection of the project-wide
correctness gate), the λ inference knob, and key-padding masking. The full RFD3-level bit-for-bit
gate is the separate ``test_identity_at_init.py`` (kickoff step 4).
"""

import torch

from spa.model import SPACrossAttention, SPAPromptKV

D, I, M, C_QUERY, C_KV, C_MODEL, N_HEAD = 2, 7, 5, 768, 1536, 768, 8


def _make():
    promptkv = SPAPromptKV(c_kv=C_KV, c_model=C_MODEL, input_rmsnorm=True)
    spa = SPACrossAttention(c_query=C_QUERY, c_model=C_MODEL, n_head=N_HEAD)
    return promptkv, spa


def test_shapes():
    promptkv, spa = _make()
    query = torch.randn(D, I, C_QUERY)
    k, v = promptkv(torch.randn(D, M, C_KV))
    assert k.shape == (D, M, C_MODEL) and v.shape == (D, M, C_MODEL)
    out = spa(query, k, v)
    assert out.shape == (D, I, C_QUERY)


def test_identity_at_init_is_exactly_zero():
    # Zero-init Wo => the SPA term is exactly 0 at init, regardless of Wq/Wk/Wv. This is the
    # mechanism the project-wide wrapped-no-prompt == vanilla-RFD3 invariant rests on.
    promptkv, spa = _make()
    query = torch.randn(D, I, C_QUERY)
    k, v = promptkv(torch.randn(D, M, C_KV))
    out = spa(query, k, v)
    assert torch.count_nonzero(out) == 0


def test_grows_in_after_training_perturbation():
    # After Wo becomes non-zero (as it would during training), the term is live and finite.
    promptkv, spa = _make()
    torch.nn.init.xavier_uniform_(spa.to_out.weight)  # simulate "grown-in" Wo
    query = torch.randn(D, I, C_QUERY)
    k, v = promptkv(torch.randn(D, M, C_KV))
    out = spa(query, k, v)
    assert torch.count_nonzero(out) > 0 and torch.isfinite(out).all()


def test_lambda_scales_output_linearly():
    promptkv, spa = _make()
    torch.nn.init.xavier_uniform_(spa.to_out.weight)
    query = torch.randn(D, I, C_QUERY)
    k, v = promptkv(torch.randn(D, M, C_KV))

    spa.set_scale(1.0)
    base = spa(query, k, v)
    spa.set_scale(0.0)
    assert torch.count_nonzero(spa(query, k, v)) == 0          # λ=0 -> unconditional
    spa.set_scale(2.0)
    assert torch.allclose(spa(query, k, v), 2.0 * base, atol=1e-5)  # linear in λ


def test_key_padding_mask_excludes_positions():
    # Masking a prompt token must change the attended output (that token no longer participates).
    promptkv, spa = _make()
    torch.nn.init.xavier_uniform_(spa.to_out.weight)
    query = torch.randn(D, I, C_QUERY)
    k, v = promptkv(torch.randn(D, M, C_KV))

    unmasked = spa(query, k, v)
    mask = torch.zeros(D, M, dtype=torch.bool)
    mask[:, -1] = True  # ignore the last prompt token
    masked = spa(query, k, v, key_padding_mask=mask)
    assert not torch.allclose(unmasked, masked)
    assert torch.isfinite(masked).all()


def test_c_model_not_divisible_by_heads_raises():
    try:
        SPACrossAttention(c_model=768, n_head=7)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible c_model/n_head")
