"""SPA training harness (kickoff step 6) + local overfit sanity (step 7), against real RFD3.

Validates the mechanism end-to-end on a synthetic-but-real example: the loss falls under
optimization, gradients flow to SPA only (RFD3 frozen), λ controls the SPA effect, and the SPA
checkpoint round-trips. Skipped unless the real ckpt + CUDA are present.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from spa.model import attach_spa

CKPT = os.environ.get("SPA_RFD3_CKPT", "/home/user1/projects/spa/models/rfdiffusion3/rfd3_latest.ckpt")
ESM3_CACHE = "/home/user1/projects/spa/training_data/processed/esm3_cache"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CKPT) and torch.cuda.is_available()),
    reason="real RFD3 ckpt and/or CUDA device not available",
)


def _cfg(tmp_dir, max_steps=60):
    return OmegaConf.create(
        {
            "paths": {"rfd3_ckpt": CKPT, "esm3_cache_dir": ESM3_CACHE},
            "hardware": {"device": "cuda:0", "batch_size": 1},
            "data": {"name": "toy"},
            "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                      "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
            "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                        "strip_bos_eos": True, "use_clss": False},
            # lr kept modest: the bf16 forward is ~2e-4 non-deterministic, so an aggressive lr makes the
            # single-example overfit occasionally diverge (flaky). 1e-3 converges in 60 steps and is stable.
            "train": {"lr": 1.0e-3, "weight_decay": 0.0, "grad_clip": 1.0, "cfg_drop_rate": 0.0,
                      "se3_augment": False, "n_cycle": 1, "max_steps": max_steps, "seed": 0,
                      "capture_length": 24, "synthetic_target_offset": 3.0, "ckpt_dir": str(tmp_dir),
                      "loss": {"sigma_data": 16, "weight": 4.0, "lddt_weight": 0.25,
                               "alpha_ligand": 10.0, "unindexed_t_alpha": 0.75}},
        }
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from spa.train.harness import (
        build_engine, build_loss, frozen_rfd3_net, load_spa, save_spa,
        set_host_train_mode, spa_training_step,
    )

    tmp = tmp_path_factory.mktemp("spa_ckpt")
    cfg = _cfg(tmp)
    device = torch.device(cfg.hardware.device)

    engine = build_engine(cfg)
    net = frozen_rfd3_net(engine)
    from spa.data.synthetic import capture_synthetic_example
    example = capture_synthetic_example(engine, net, cfg, device)

    adapter = attach_spa(net, cfg).to(device)
    set_host_train_mode(net)
    loss_fn = build_loss(cfg).to(device)
    host_p = net.diffusion_module.diffusion_transformer.blocks[0].attention_pair_bias.orig.to_q.weight

    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.train.lr)
    gen = torch.Generator().manual_seed(0)
    losses, host_grad_seen = [], []
    for _ in range(cfg.train.max_steps):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = spa_training_step(net, adapter, loss_fn, example, cfg, gen)
        loss.backward()
        host_grad_seen.append(host_p.grad is None)
        opt.step()
        losses.append(loss.item())

    spa_grad = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in adapter.parameters())

    # λ response: zero the SPA scale -> loss should rise back toward the untrained value.
    adapter.set_scale(0.0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss_l0, _ = spa_training_step(net, adapter, loss_fn, example, cfg, torch.Generator().manual_seed(0))
    adapter.set_scale(1.0)

    # checkpoint round-trip
    ckpt = os.path.join(tmp, "spa.pt")
    save_spa(adapter, ckpt)
    before = adapter.cross_attn[0].to_out.weight.detach().clone()
    torch.nn.init.normal_(adapter.cross_attn[0].to_out.weight)   # perturb
    load_spa(adapter, ckpt)
    restored = torch.equal(adapter.cross_attn[0].to_out.weight, before)

    return SimpleNamespace(losses=losses, host_grad_none=all(host_grad_seen), spa_grad=spa_grad,
                           loss_l0=loss_l0.item(), restored=restored)


def test_loss_falls(trained):
    assert min(trained.losses[5:]) < trained.losses[0]


def test_gradient_flows_to_spa_only(trained):
    assert trained.host_grad_none          # frozen RFD3 never accumulates gradient
    assert trained.spa_grad > 0            # SPA params do


def test_lambda_controls_effect(trained):
    # With λ=0 the SPA term vanishes -> loss returns above the trained (λ=1) value.
    assert trained.loss_l0 > trained.losses[-1]


def test_checkpoint_round_trips(trained):
    assert trained.restored
