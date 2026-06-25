"""Wrapper + loader mechanics (kickoff step 3), exercised against a lightweight fake host.

These verify the attach/freeze/side-channel machinery WITHOUT loading the heavy real RFD3 (that
is the separate, real bit-for-bit gate in ``test_identity_at_init.py``, kickoff step 4). The fake
host mirrors the real attribute chain ``diffusion_module.diffusion_transformer.blocks[i].
attention_pair_bias`` and the ``forward(Q_L, C_L, P_LL, **kwargs) -> [D,I,768]`` residual contract.
"""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from spa.model import SPAWrappedAttention, attach_spa

D, I, N, C_QUERY, C_KV = 2, 9, 6, 768, 1536
N_BLOCKS = 18


class FakeAttn(nn.Module):
    """Mimics LocalAttentionPairBias: returns a [D,I,768] residual term; has use_checkpointing."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(C_QUERY, C_QUERY)
        self.use_checkpointing = True

    def forward(self, Q_L, C_L=None, P_LL=None, **kwargs):
        return self.proj(Q_L)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_pair_bias = FakeAttn()


class FakeTransformer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.blocks = nn.ModuleList(FakeBlock() for _ in range(n))


class FakeDiffusionModule(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.diffusion_transformer = FakeTransformer(n)


class FakeRFD3(nn.Module):
    """Exposes the real attribute chain; run_blocks mimics the block residual (blocks.py:692)."""

    def __init__(self, n=N_BLOCKS):
        super().__init__()
        self.diffusion_module = FakeDiffusionModule(n)

    def run_blocks(self, A_I):
        for blk in self.diffusion_module.diffusion_transformer.blocks:
            A_I = A_I + blk.attention_pair_bias(A_I)
        return A_I


def _cfg():
    return OmegaConf.create(
        {
            "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8,
                      "shared_kv": True, "zero_init_output": True, "lambda_init": 1.0,
                      "input_rmsnorm": True},
            "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                        "strip_bos_eos": True, "use_clss": False},
        }
    )


def _clss_cached() -> bool:
    """True iff the CLSS ckpt is already in the HF cache (so variant-A tests need no network)."""
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download("guyyanai/CLSS", "CLSS-sub.lckpt", local_files_only=True)
        return True
    except Exception:
        return False


clss_only = pytest.mark.skipif(not _clss_cached(), reason="CLSS ckpt not in the HF cache")


def _cfg_clss(n_tokens: int = 4):
    cfg = _cfg()
    cfg.variant = OmegaConf.create({"name": "A", "projector": "clss", "n_tokens": n_tokens,
                                    "use_clss": True, "clss_model_name": "CLSS-sub.lckpt",
                                    "strip_bos_eos": True})
    return cfg


def _cfg_global(n_tokens: int = 4):
    cfg = _cfg()
    cfg.variant = OmegaConf.create({"name": "B", "projector": "global_fanout", "n_tokens": n_tokens,
                                    "use_clss": False, "strip_bos_eos": True})
    return cfg


def test_attach_wraps_all_blocks():
    model = FakeRFD3()
    adapter = attach_spa(model, _cfg())
    blocks = model.diffusion_module.diffusion_transformer.blocks
    assert all(isinstance(b.attention_pair_bias, SPAWrappedAttention) for b in blocks)
    assert len(adapter.cross_attn) == N_BLOCKS


def test_identity_with_no_prompt_matches_vanilla():
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    attach_spa(model, _cfg())          # no prompt set on the context
    assert torch.equal(vanilla, model.run_blocks(A_I))


def test_identity_at_init_holds_even_with_prompt():
    # Zero-init Wo => the SPA term is exactly 0 at init, so even WITH a prompt the wrapped model
    # reproduces vanilla. This is the local stand-in for the project-wide identity gate.
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    adapter = attach_spa(model, _cfg())
    adapter.set_prompt(torch.randn(D, N, C_KV))
    assert torch.equal(vanilla, model.run_blocks(A_I))


def test_prompt_changes_output_once_grown_in():
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    adapter = attach_spa(model, _cfg())
    for ca in adapter.cross_attn:            # simulate a "grown-in" Wo after training
        nn.init.xavier_uniform_(ca.to_out.weight)
    adapter.set_prompt(torch.randn(D, N, C_KV))
    out = model.run_blocks(A_I)
    assert not torch.allclose(vanilla, out) and torch.isfinite(out).all()


def test_set_scale_zero_and_clear_prompt_restore_identity():
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    adapter = attach_spa(model, _cfg())
    for ca in adapter.cross_attn:
        nn.init.xavier_uniform_(ca.to_out.weight)
    adapter.set_prompt(torch.randn(D, N, C_KV))

    adapter.set_scale(0.0)                    # λ=0 -> unconditional
    assert torch.equal(vanilla, model.run_blocks(A_I))
    adapter.set_scale(1.0)
    adapter.clear_prompt()                    # no prompt -> base only
    assert torch.equal(vanilla, model.run_blocks(A_I))


def test_freeze_host_and_gather_params():
    model = FakeRFD3()
    adapter = attach_spa(model, _cfg())
    # Host frozen: every wrapped original's params are non-trainable.
    for b in model.diffusion_module.diffusion_transformer.blocks:
        assert all(not p.requires_grad for p in b.attention_pair_bias.orig.parameters())
    # SPA trainable and gathered in the adapter (projector has none; prompt_kv + cross_attn do).
    assert all(p.requires_grad for p in adapter.parameters())
    assert sum(p.numel() for p in adapter.parameters()) > 0


def test_use_checkpointing_propagates_to_orig():
    model = FakeRFD3()
    attach_spa(model, _cfg())
    wrapper = model.diffusion_module.diffusion_transformer.blocks[0].attention_pair_bias
    wrapper.use_checkpointing = False        # how RFD3 sets it at runtime (blocks.py:623)
    wrapper(torch.randn(D, I, C_QUERY))
    assert wrapper.orig.use_checkpointing is False


def test_per_block_kv_not_yet_supported():
    cfg = _cfg()
    cfg.model.shared_kv = False
    try:
        attach_spa(FakeRFD3(), cfg)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for per-block K/V")


# --- Variant A (CLSS 1×32) -------------------------------------------------------------------------

@clss_only
def test_variant_a_projector_shape_and_freeze():
    from spa.model.projectors import CLSSProjector

    adapter = attach_spa(FakeRFD3(), _cfg_clss(n_tokens=4))
    proj = adapter.projector
    assert isinstance(proj, CLSSProjector)
    # the loaded CLSS structure_adapter is FROZEN; the trainable fan-out is on (the freeze-hook fix)
    assert all(not p.requires_grad for p in proj.structure_adapter.parameters())
    assert all(p.requires_grad for p in proj.fanout.parameters())
    # N×1536 per-residue prompt -> n_tokens fanned tokens
    assert proj(torch.randn(D, N, C_KV)).shape == (D, 4, C_KV)


@clss_only
def test_variant_a_identity_at_init():
    # zero-init Wo => SPA term 0 even WITH a prompt; variant A still reproduces vanilla at init.
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    adapter = attach_spa(model, _cfg_clss())
    adapter.set_prompt(torch.randn(D, N, C_KV))
    assert torch.equal(vanilla, model.run_blocks(A_I))


@clss_only
def test_variant_a_grows_in_and_changes_output():
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    adapter = attach_spa(model, _cfg_clss())
    for ca in adapter.cross_attn:                  # simulate a grown-in Wo after training
        nn.init.xavier_uniform_(ca.to_out.weight)
    adapter.set_prompt(torch.randn(D, N, C_KV))
    out = model.run_blocks(A_I)
    assert not torch.allclose(vanilla, out) and torch.isfinite(out).all()


@clss_only
def test_variant_a_drops_mismatched_prompt_mask():
    # variant A reduces N -> n_tokens, so a per-residue [D,N] mask can't apply -> dropped (no crash).
    adapter = attach_spa(FakeRFD3(), _cfg_clss(n_tokens=4))
    adapter.set_prompt(torch.randn(D, N, C_KV), key_padding_mask=torch.zeros(D, N, dtype=torch.bool))
    assert adapter.context.key_padding_mask is None


# --- Variant B (1×1536 mean-pooled ESM3, no CLSS) --------------------------------------------------

def test_variant_b_projector_shape_and_all_trainable():
    from spa.model.projectors import GlobalFanoutProjector

    adapter = attach_spa(FakeRFD3(), _cfg_global(n_tokens=4))
    proj = adapter.projector
    assert isinstance(proj, GlobalFanoutProjector)
    assert all(p.requires_grad for p in adapter.parameters())   # no frozen encoder (unlike A)
    assert proj(torch.randn(D, N, C_KV)).shape == (D, 4, C_KV)


def test_variant_b_identity_at_init():
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    adapter = attach_spa(model, _cfg_global())
    adapter.set_prompt(torch.randn(D, N, C_KV))
    assert torch.equal(vanilla, model.run_blocks(A_I))


def test_variant_b_grows_in_and_drops_mismatched_mask():
    model = FakeRFD3()
    A_I = torch.randn(D, I, C_QUERY)
    vanilla = model.run_blocks(A_I)
    adapter = attach_spa(model, _cfg_global(n_tokens=4))
    for ca in adapter.cross_attn:
        nn.init.xavier_uniform_(ca.to_out.weight)
    adapter.set_prompt(torch.randn(D, N, C_KV), key_padding_mask=torch.zeros(D, N, dtype=torch.bool))
    assert adapter.context.key_padding_mask is None             # N-mask can't fit 4 tokens -> dropped
    out = model.run_blocks(A_I)
    assert not torch.allclose(vanilla, out) and torch.isfinite(out).all()
