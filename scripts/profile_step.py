"""Instrument one SPA training step: D, per-component timing, steady-state throughput.

A diagnostic entry point (same Hydra config as train.py) used to size the Phase-2 budget from real
hardware (dev 04 §11 / 08 §6). Run via run_train.sh with RUN_MODE=profile, or directly:

    conda run -n spa-dev python scripts/profile_step.py data=cddb hardware=cloud_h100 ...

Times each part of a step with CUDA syncs (GPU ops are async, so naive timing lies):
  fetch      DataLoader next() — live featurization + ESM3-cache .pt load (in workers)
  to_dev     H2D move of the example
  set_prompt SPA projector + K/V projection (+ RMSNorm)
  forward    frozen RFD3 single denoise step + SPA cross-attn
  loss       RFD3-native DiffusionLoss
  backward   autograd through RFD3's graph to the SPA params
  opt        grad clip + AdamW step
Plus a micro-bench separating ESM3-cache load vs featurization (both hide inside `fetch`).
Env: N_STEPS (default 40), WARMUP (default 10).
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import os
    import statistics
    import time

    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from spa.data.dataset import CDDBPromptDataset, load_structure, move_to_device
    from spa.model import attach_spa
    from spa.train.harness import build_engine, build_loss, frozen_rfd3_net, set_host_train_mode
    from spa.utils.device import resolve_device

    n_steps = int(os.environ.get("N_STEPS", "40"))
    warmup = int(os.environ.get("WARMUP", "10"))
    OmegaConf.set_struct(cfg, False)
    cfg.train.se3_augment = False  # cddb pipeline augments natively (mirror harness)

    device = resolve_device(cfg.hardware.device)
    print(f"device={device}  variant={cfg.variant.name}  length_cap={cfg.data.get('length_cap')}  "
          f"num_workers={cfg.train.num_workers}")

    engine = build_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = attach_spa(net, cfg).to(device)
    set_host_train_mode(net)
    loss_fn = build_loss(cfg).to(device)
    opt = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=cfg.train.lr)

    ds = CDDBPromptDataset(cfg, split="train")
    print(f"dataset(train) = {len(ds)} structures (cap={cfg.data.get('length_cap')}, "
          f"require_cached={cfg.data.require_cached_prompt})")
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0],
                        num_workers=cfg.train.num_workers, persistent_workers=cfg.train.num_workers > 0)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    keys = ["fetch", "to_dev", "set_prompt", "forward", "loss", "backward", "opt"]
    T = {k: [] for k in keys}
    Ds, Ls = [], []
    it = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(n_steps):
        t0 = time.perf_counter(); ex = next(it); t1 = time.perf_counter()
        ex = move_to_device(ex, device); sync(); t2 = time.perf_counter()
        prompt = ex["prompt"]; Ds.append(int(prompt.shape[0])); Ls.append(int(ex["X_noisy_L"].shape[1]))
        opt.zero_grad()
        adapter.set_prompt(prompt, key_padding_mask=ex.get("prompt_mask")); sync(); t3 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = net.forward(input={"X_noisy_L": ex["X_noisy_L"], "t": ex["t"], "f": ex["feats"]},
                              n_cycle=cfg.train.n_cycle)["X_L"].float()
            sync(); t4 = time.perf_counter()
            loss, _ = loss_fn(network_input={"f": ex["feats"], "t": ex["t"]},
                              network_output={"X_L": out},
                              loss_input={"crd_mask_L": ex["crd_mask_L"],
                                          "is_original_unindexed_token": ex["is_original_unindexed_token"],
                                          "X_gt_L_in_input_frame": ex["X_gt_L_in_input_frame"]})
        sync(); t5 = time.perf_counter()
        loss.backward(); sync(); t6 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.train.grad_clip); opt.step(); sync()
        t7 = time.perf_counter()
        for k, dt in zip(keys, [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6]):
            T[k].append(dt)
        print(f"step {step:3d} total={(t7-t0)*1e3:6.0f}ms  D={Ds[-1]:2d} L={Ls[-1]:4d}  "
              f"fetch={(t1-t0)*1e3:5.0f} fwd={(t4-t3)*1e3:5.0f} bwd={(t6-t5)*1e3:5.0f}")

    def ss(k):
        return statistics.mean(T[k][warmup:])

    total = sum(ss(k) for k in keys)
    print(f"\n===== STEADY-STATE (steps {warmup}-{n_steps-1}; n={n_steps-warmup}) =====")
    for k in keys:
        print(f"  {k:10s} {ss(k)*1e3:7.1f} ms  ({ss(k)/total*100:4.1f}%)")
    print(f"  {'TOTAL':10s} {total*1e3:7.1f} ms/step  ->  {1/total:5.2f} struct/s")
    print(f"  D (diffusion replicas): mode={statistics.mode(Ds)} min={min(Ds)} max={max(Ds)}   "
          f"L(atoms): mean={statistics.mean(Ls):.0f} min={min(Ls)} max={max(Ls)}")
    if device.type == "cuda":
        print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    n_epoch = int(364373 * 0.936)  # train split x (<=384 fraction)
    print(f"  EXTRAPOLATION: 1 epoch over ~{n_epoch} struct = {n_epoch*total/3600:.1f} h (this device+L)")

    print("\n===== DATALOADER MICRO-BENCH (single-thread, isolates fetch internals) =====")
    row = ds.rows.iloc[0]
    cache_pt = os.path.join(ds.cache_dir, f"{os.path.splitext(row['pdb_file'])[0]}.pt")
    tl = []
    for _ in range(20):
        a = time.perf_counter(); _ = torch.load(cache_pt, weights_only=True); tl.append(time.perf_counter()-a)
    print(f"  ESM3-cache .pt load:  {statistics.mean(tl)*1e3:6.1f} ms")
    tf = ds.transform
    ft = []
    for i in range(min(5, len(ds.rows))):
        r = ds.rows.iloc[i]
        a = time.perf_counter()
        _ = tf(load_structure(r["id"], os.path.join(ds.pdb_dir, r["pdb_file"]))); ft.append(time.perf_counter()-a)
    print(f"  featurization (1 struct, single-thread): {statistics.mean(ft)*1e3:6.0f} ms")
    print("DONE")


if __name__ == "__main__":
    main()
