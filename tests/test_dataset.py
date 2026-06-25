"""CDDBPromptDataset (kickoff step 8 Part B) — faithful CDDB featurization -> harness example.

Validates that a REAL CDDB structure, featurized through RFD3's training pipeline + joined to its
cached ESM3 prompt, yields an example dict that ``RFD3.forward`` + ``DiffusionLoss`` consume: the
dims align (prompt ``N`` == RFD3 token count), the loss is finite and falls under a short overfit,
and gradient flows to SPA only (RFD3 frozen). Skipped unless the real ckpt + CUDA + the split
manifest + a cached ESM3 prompt are present.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from spa.model import attach_spa

ROOT = "/home/user1/projects/spa"
CKPT = os.environ.get("SPA_RFD3_CKPT", f"{ROOT}/models/rfdiffusion3/rfd3_latest.ckpt")
ESM3_CACHE = f"{ROOT}/training_data/processed/esm3_cache"
DATA = f"{ROOT}/training_data/proteina-atomistica_data_vrelease/atomistica_data_release"
TRAIN_MANIFEST = f"{ROOT}/training_data/train/manifest.parquet"
FOUNDRY_TF = f"{ROOT}/needed_repos/foundry/models/rfd3/configs/datasets/train"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(TRAIN_MANIFEST) and torch.cuda.is_available()),
    reason="real RFD3 ckpt / split manifest / CUDA not available",
)


def _cfg():
    return OmegaConf.create({
        "paths": {"rfd3_ckpt": CKPT, "esm3_cache_dir": ESM3_CACHE, "foundry_train_cfg_dir": FOUNDRY_TF},
        "hardware": {"device": "cuda:0", "batch_size": 1},
        "data": {"name": "cddb", "pdb_dir": f"{DATA}/pdb", "splits_root": f"{ROOT}/training_data",
                 "length_cap": 512, "require_cached_prompt": True},
        "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
        "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False},
        "train": {"lr": 1.0e-4, "weight_decay": 0.0, "grad_clip": 1.0, "cfg_drop_rate": 0.0,
                  "se3_augment": False, "n_cycle": 1, "seed": 0, "capture_length": 24,
                  "loss": {"sigma_data": 16, "weight": 4.0, "lddt_weight": 0.25,
                           "alpha_ligand": 10.0, "unindexed_t_alpha": 0.75}},
    })


@pytest.fixture(scope="module")
def gate():
    from spa.data.dataset import CDDBPromptDataset, move_to_device
    from spa.train.harness import (
        build_engine, build_loss, frozen_rfd3_net, set_host_train_mode, spa_training_step,
    )

    cfg = _cfg()
    device = torch.device(cfg.hardware.device)
    ds = CDDBPromptDataset(cfg, split="train")
    if len(ds) == 0:
        pytest.skip("no train-split structure has a cached ESM3 prompt")

    engine = build_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = attach_spa(net, cfg).to(device)
    set_host_train_mode(net)
    loss_fn = build_loss(cfg).to(device)

    ex = move_to_device(ds[0], device)
    n_tokens = int(ex["feats"]["atom_to_token_map"].max()) + 1  # RFD3 token-track length (SPA query)
    prompt_n = ex["prompt"].shape[1]
    D, L = ex["t"].shape[0], ex["X_noisy_L"].shape[1]
    dims_ok = (
        tuple(ex["X_noisy_L"].shape) == tuple(ex["X_gt_L_in_input_frame"].shape) == (D, L, 3)
        and tuple(ex["crd_mask_L"].shape) == (L,) and ex["crd_mask_L"].dtype == torch.bool
        and tuple(ex["prompt"].shape) == (D, prompt_n, 1536)
    )

    host_p = net.diffusion_module.diffusion_transformer.blocks[0].attention_pair_bias.orig.to_q.weight
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.train.lr)
    g = torch.Generator().manual_seed(0)
    losses, host_grad_none = [], []
    for _ in range(30):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = spa_training_step(net, adapter, loss_fn, ex, cfg, g)
        loss.backward()
        host_grad_none.append(host_p.grad is None)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    spa_grad = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in adapter.parameters())

    return SimpleNamespace(losses=losses, dims_ok=dims_ok, n_tokens=n_tokens, prompt_n=prompt_n,
                           host_grad_none=all(host_grad_none), spa_grad=spa_grad,
                           finite=all(map(math.isfinite, losses)))


def test_example_contract(gate):
    assert gate.dims_ok and gate.finite


def test_prompt_aligns_to_tokens(gate):
    # the per-residue ESM3 prompt (N) must align to the SPA query token track (I)
    assert gate.prompt_n == gate.n_tokens


def test_loss_falls(gate):
    assert min(gate.losses[5:]) < gate.losses[0]


def test_gradient_flows_to_spa_only(gate):
    assert gate.host_grad_none      # frozen RFD3 never accumulates gradient
    assert gate.spa_grad > 0        # SPA params do


def test_run_b_motif_mask_and_loss():
    """Run B (mixed): `conditioning=island` yields a native fixed-coord motif; the SPA prompt masks the
    motif rows (non-overlap, dev 10 §7.2) leaving a scaffold; the masked example drives the loss with
    gradient to SPA only. island is stochastic (~50% motif) so we sample a few draws to find one."""
    from spa.data.dataset import CDDBPromptDataset, move_to_device
    from spa.train.harness import (
        build_engine, build_loss, frozen_rfd3_net, set_host_train_mode, spa_training_step,
    )

    cfg = _cfg()
    cfg.data.conditioning = "island"   # Run B
    device = torch.device(cfg.hardware.device)
    ds = CDDBPromptDataset(cfg, split="train")
    if len(ds) == 0:
        pytest.skip("no train-split structure has a cached ESM3 prompt")

    motif_ex = next((ex for _ in range(16) if (ex := ds[0])["prompt_mask"] is not None), None)
    assert motif_ex is not None, "island produced no motif in 16 draws"
    pm = motif_ex["prompt_mask"]
    N = pm.shape[1]
    I = int(motif_ex["feats"]["ref_motif_token_type"].shape[0])
    masked = int(pm[0].sum())
    assert N == I                                  # indexed motif -> prompt aligns to the token track
    assert 0 < masked < N                          # masks the motif, leaves a scaffold (not all/none)
    assert pm.dtype == torch.bool and tuple(pm.shape) == (motif_ex["t"].shape[0], N)

    engine = build_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = attach_spa(net, cfg).to(device)
    set_host_train_mode(net)
    loss_fn = build_loss(cfg).to(device)
    ex = move_to_device(motif_ex, device)

    host_p = net.diffusion_module.diffusion_transformer.blocks[0].attention_pair_bias.orig.to_q.weight
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    g = torch.Generator().manual_seed(0)
    losses, host_none = [], []
    for _ in range(12):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = spa_training_step(net, adapter, loss_fn, ex, cfg, g)
        loss.backward()
        host_none.append(host_p.grad is None)
        opt.step()
        losses.append(loss.item())
    spa_grad = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in adapter.parameters())
    assert all(map(math.isfinite, losses)) and min(losses[3:]) < losses[0]
    assert all(host_none) and spa_grad > 0
