"""THE standing correctness gate: wrapped-no-prompt == vanilla RFdiffusion3, bit-for-bit.

Kickoff step 4 (dev 02_attachment_points.md §8, 03_spa_architecture.md §1). With SPA attached but
no prompt stashed (and via zero-init Wo, even WITH a prompt at init), the model must reproduce
vanilla RFD3 exactly — the training-stability guarantee and the experiment baseline.

This exercises the REAL RFD3: it builds the inference engine, captures a real token-track
transformer input (the 18 `diffusion_transformer.blocks` SPA wraps, on the EMA `shadow` copy that
inference actually uses), and replays it vanilla vs. wrapped. A single transformer forward is
deterministic (asserted), so the comparison is bit-for-bit. Skipped unless the real ckpt + a CUDA
device are present, so the suite stays green elsewhere.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from spa.model import attach_spa

CKPT = os.environ.get("SPA_RFD3_CKPT", "/home/user1/projects/spa/models/rfdiffusion3/rfd3_latest.ckpt")
LENGTH = int(os.environ.get("SPA_RFD3_TEST_LENGTH", "16"))

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CKPT) and torch.cuda.is_available()),
    reason="real RFD3 ckpt and/or CUDA device not available",
)

SPA_CFG = OmegaConf.create(
    {
        "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
        "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False},
    }
)


def _clone_any(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _clone_any(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_any(v) for v in x)
    return x


class _Stop(Exception):
    """Abort the rollout once the first real token-transformer input is captured."""


@pytest.fixture(scope="module")
def rfd3():
    """Build RFD3, capture a real token-transformer input, attach SPA to the inference (shadow) net."""
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    cfg = RFD3InferenceConfig(ckpt_path=CKPT, specification={"length": LENGTH},
                              diffusion_batch_size=1, seed=0)
    engine = RFD3InferenceEngine(**cfg)
    engine.initialize()
    shadow = dict(engine.trainer.state["model"].named_modules())["_forward_module.shadow"]
    dt = shadow.diffusion_module.diffusion_transformer

    box = {}

    def hook(_m, args, kwargs):
        box["a"], box["kw"] = _clone_any(args), _clone_any(kwargs)
        raise _Stop

    handle = dt.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        engine.run(inputs=None, out_dir=None)
    except _Stop:
        pass
    finally:
        handle.remove()

    a, kw = box["a"], box["kw"]

    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(0)
            return dt(*a, **kw)

    vanilla = fwd()
    deterministic = torch.equal(vanilla, fwd())
    adapter = attach_spa(shadow, SPA_CFG).to(a[0].device)
    return SimpleNamespace(dt=dt, a=a, kw=kw, fwd=fwd, vanilla=vanilla,
                           deterministic=deterministic, adapter=adapter, device=a[0].device)


def _reset(adapter):
    """Restore identity-at-init state: zero Wo, λ=1, no prompt (keeps tests order-independent)."""
    for ca in adapter.cross_attn:
        torch.nn.init.zeros_(ca.to_out.weight)
        torch.nn.init.zeros_(ca.to_out.bias)
        ca.set_scale(1.0)
    adapter.clear_prompt()


def test_transformer_forward_is_deterministic(rfd3):
    # Precondition for the bit-for-bit comparisons below.
    assert rfd3.deterministic


def test_wrapped_no_prompt_equals_vanilla_bitforbit(rfd3):
    _reset(rfd3.adapter)
    assert torch.equal(rfd3.vanilla, rfd3.fwd())


def test_identity_holds_with_prompt_at_init(rfd3):
    # Zero-init Wo => exact identity even with a real prompt stashed (and λ stays in bf16).
    _reset(rfd3.adapter)
    D, L = rfd3.a[0].shape[0], rfd3.a[0].shape[1]
    rfd3.adapter.set_prompt(torch.randn(D, L + 2, SPA_CFG.model.c_kv, device=rfd3.device))
    assert torch.equal(rfd3.vanilla, rfd3.fwd())


def test_grown_in_adapter_changes_output(rfd3):
    # Sanity: once Wo is non-zero, the SPA path is live and perturbs the real transformer output.
    _reset(rfd3.adapter)
    D, L = rfd3.a[0].shape[0], rfd3.a[0].shape[1]
    rfd3.adapter.set_prompt(torch.randn(D, L + 2, SPA_CFG.model.c_kv, device=rfd3.device))
    for ca in rfd3.adapter.cross_attn:
        torch.nn.init.xavier_uniform_(ca.to_out.weight)
    assert not torch.equal(rfd3.vanilla, rfd3.fwd())
    _reset(rfd3.adapter)
