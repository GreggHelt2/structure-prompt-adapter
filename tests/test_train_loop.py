"""Workstream-C training-loop tests: scheduler, length-stratified sampler, full-state checkpoint
round-trip (all CPU, no RFD3), plus an end-to-end save->resume on the real toy GPU path.

The CPU tests pin the deterministic guarantees (a resumed run restores adapter/opt/sched/RNG/step
*exactly*); the GPU test exercises the whole ``train()`` resume path (run -> checkpoint -> resume ->
finish at the right step). The toy-overfit *mechanism* itself is covered by ``test_harness.py``.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from spa.train.harness import (
    _length_stratified_sampler, build_scheduler, load_checkpoint, load_spa, save_checkpoint,
    save_spa,
)

# --------------------------------------------------------------------------------------------------
# CPU unit tests (always run)
# --------------------------------------------------------------------------------------------------


def test_length_stratified_sampler_balances():
    """A 900/90/10 skewed length population should be drawn ~balanced across its three bands, with
    the rare long band massively upsampled vs its natural 1% frequency (dev 04 §11)."""
    lengths = np.concatenate([np.full(900, 50), np.full(90, 150), np.full(10, 350)])
    rows = pd.DataFrame({"length": lengths})
    s = _length_stratified_sampler(rows, 20, torch.Generator().manual_seed(0))
    sel = lengths[np.array(list(s))]
    assert len(sel) == len(lengths)               # num_samples == population size
    assert (sel == 350).mean() > 0.2              # rare band pulled from 1% toward ~1/3 (>20x)
    for v in (50, 150, 350):                       # all three bands roughly balanced
        assert 0.2 < (sel == v).mean() < 0.45


def test_length_stratified_sampler_uniform_when_degenerate():
    """All-equal lengths -> uniform weights (no crash, no div-by-zero)."""
    rows = pd.DataFrame({"length": np.full(50, 100)})
    s = _length_stratified_sampler(rows, 20, torch.Generator().manual_seed(0))
    assert len(list(s)) == 50


def test_warmup_cosine_schedule():
    """Linear warmup to the base LR, then a cosine decay down to the min_lr_ratio floor."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    cfg = OmegaConf.create({"train": {"warmup_steps": 5, "max_steps": 20, "min_lr_ratio": 0.1}})
    sched = build_scheduler(opt, cfg)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])  # LR applied at this step (before advancing)
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.2, abs=1e-6)         # warmup start = 1/5 of base
    assert max(lrs) == pytest.approx(1.0, abs=1e-6)       # reaches the base LR at the warmup peak
    assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(5, 19))  # monotone decay after warmup
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1, abs=1e-6)  # ends at the floor (step 20)


def test_checkpoint_round_trip_full_state(tmp_path):
    """save_checkpoint -> perturb everything -> load_checkpoint restores adapter weights, optimizer
    moments, scheduler position, sampling-RNG, and the step/sample counters EXACTLY (deterministic)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = OmegaConf.create({"train": {"warmup_steps": 2, "max_steps": 10, "min_lr_ratio": 0.0},
                            "variant": {"name": "C"}})
    sched = build_scheduler(opt, cfg)
    gen = torch.Generator().manual_seed(123)
    for _ in range(3):  # give the optimizer state + advance the scheduler
        loss = model(torch.randn(2, 4)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    _ = torch.rand(5, generator=gen)  # advance the sampling generator

    path = os.path.join(tmp_path, "last.pt")
    save_checkpoint(path, model, opt, sched, step=2, samples_seen=24, gen=gen, cfg=cfg,
                    provenance={"git_commit": "abc1234", "variant": "C"})

    w_before = model.weight.detach().clone()
    gen_before = gen.get_state().clone()
    epoch_before = sched.last_epoch

    torch.nn.init.normal_(model.weight)          # perturb the model
    gen.manual_seed(999)                          # perturb the RNG
    fresh_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # NO optimizer state
    fresh_sched = build_scheduler(fresh_opt, cfg)
    step, samples_seen = load_checkpoint(path, model, fresh_opt, fresh_sched, gen, torch.device("cpu"))

    assert (step, samples_seen) == (2, 24)
    assert torch.equal(model.weight, w_before)                # adapter weights restored
    assert torch.equal(gen.get_state(), gen_before)           # sampling RNG restored
    assert fresh_sched.last_epoch == epoch_before             # scheduler position restored
    assert "exp_avg" in fresh_opt.state[model.weight]         # Adam moments restored
    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert raw["provenance"]["git_commit"] == "abc1234"       # provenance stamped


def test_load_spa_accepts_export_and_full_state(tmp_path):
    """load_spa must read BOTH checkpoint formats we write: the adapter-only `save_spa` export
    (`spa_*_final.pt`) AND the full-state `save_checkpoint` snapshots (`spa_*_step*.pt` / `last.pt`),
    extracting the adapter sub-dict from the latter — so an early snapshot can be evaluated directly
    (eval.ckpt=...) with no manual extraction."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = OmegaConf.create({"train": {"warmup_steps": 2, "max_steps": 10, "min_lr_ratio": 0.0},
                            "variant": {"name": "C"}})
    sched = build_scheduler(opt, cfg)
    gen = torch.Generator().manual_seed(123)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()                                   # give the optimizer real moment state
    w_before = model.weight.detach().clone()

    export_path = os.path.join(tmp_path, "spa_C_final.pt")        # adapter-only (save_spa)
    full_path = os.path.join(tmp_path, "spa_C_step1000.pt")       # full-state (save_checkpoint)
    save_spa(model, export_path)
    save_checkpoint(full_path, model, opt, sched, step=1, samples_seen=8, gen=gen, cfg=cfg,
                    provenance={"git_commit": "deadbee"})

    for path in (export_path, full_path):                        # both formats round-trip identically
        fresh = torch.nn.Linear(4, 4)
        torch.nn.init.normal_(fresh.weight)                      # perturb so a no-op "load" would fail
        load_spa(fresh, path)
        assert torch.equal(fresh.weight, w_before), f"load_spa failed for {os.path.basename(path)}"


# --------------------------------------------------------------------------------------------------
# GPU end-to-end resume (needs the real RFD3 ckpt + CUDA — mirrors test_harness gating)
# --------------------------------------------------------------------------------------------------

CKPT = os.environ.get("SPA_RFD3_CKPT", "/home/user1/projects/spa/models/rfdiffusion3/rfd3_latest.ckpt")
ESM3_CACHE = "/home/user1/projects/spa/training_data/processed/esm3_cache"

gpu = pytest.mark.skipif(
    not (os.path.exists(CKPT) and torch.cuda.is_available()),
    reason="real RFD3 ckpt and/or CUDA device not available",
)


def _toy_cfg(ckpt_dir, max_steps, ckpt_every):
    return OmegaConf.create(
        {
            "run_name": "c6_resume",
            "paths": {"rfd3_ckpt": CKPT, "esm3_cache_dir": ESM3_CACHE,
                      "data_root": "/home/user1/projects/spa/training_data"},
            "hardware": {"device": "cuda:0", "batch_size": 1, "grad_accum": 1},
            "data": {"name": "toy", "conditioning": None},
            "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                      "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
            "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                        "strip_bos_eos": True, "use_clss": False},
            "train": {"lr": 1.0e-3, "weight_decay": 0.0, "grad_clip": 1.0, "cfg_drop_rate": 0.0,
                      "se3_augment": False, "n_cycle": 1, "max_steps": max_steps, "seed": 0,
                      "warmup_steps": 2, "min_lr_ratio": 0.0, "matmul_precision": "high",
                      "ckpt_every_steps": ckpt_every, "resume": "auto", "val_every_steps": 0,
                      "log_every_steps": 5, "tracker": None, "capture_length": 24,
                      "synthetic_target_offset": 3.0, "ckpt_dir": str(ckpt_dir),
                      "loss": {"sigma_data": 16, "weight": 4.0, "lddt_weight": 0.25,
                               "alpha_ligand": 10.0, "unindexed_t_alpha": 0.75}},
        }
    )


@gpu
def test_train_resumes_end_to_end(tmp_path):
    """Run the real toy ``train()`` for 6 steps (checkpointing at 3/6), then re-invoke with a larger
    budget: it must auto-resume from last.pt at step 6 and finish at step 9, with the step/sample
    counters threaded correctly through the GCS-resume code path."""
    from spa.train.harness import train

    cfg = _toy_cfg(tmp_path, max_steps=6, ckpt_every=3)
    train(cfg)
    last = os.path.join(tmp_path, "spa_C_last.pt")
    ck = torch.load(last, map_location="cpu", weights_only=False)
    assert ck["step"] == 5 and ck["samples_seen"] == 6
    assert ck["provenance"]["git_commit"]               # provenance stamped from the real repo

    cfg.train.max_steps = 10                             # resume with a larger budget
    train(cfg)
    ck2 = torch.load(last, map_location="cpu", weights_only=False)
    assert ck2["step"] == 9 and ck2["samples_seen"] == 10
