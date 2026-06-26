"""Instrument SPA training steps: per-component timing, steady-state throughput, across a D-sweep.

Diagnostic entry point (same Hydra config as train.py) to size the Phase-2 budget from real hardware
and quantify the efficiency levers (dev 04 §11 / 08 §6). Run via run_train.sh with RUN_MODE=profile,
or directly:

    conda run -n spa-dev python scripts/profile_step.py data=cddb hardware=cloud_h100 ...

Times each part of a step with CUDA syncs (GPU ops are async, so naive timing lies):
  fetch / to_dev / set_prompt / forward / loss / backward / opt
Sweeps D (diffusion noise-replicas/structure) over DSWEEP and reports steady-state ms/step + struct/s
+ peak mem for each, so we can read the throughput-vs-D trade directly.

Env knobs:
  DSWEEP            comma list of D to sweep (default "4,8,32"; ascending so a likely-OOM big-D is last)
  N_STEPS / WARMUP  steps per D / warmup discarded (default 25 / 8)
  CKPT_OFF          "1" -> disable RFD3 activation checkpointing (faster backward, more memory)
  MATMUL_PRECISION  torch.set_float32_matmul_precision (default "high" = TF32; "highest" = true fp32)
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

    n_steps = int(os.environ.get("N_STEPS", "25"))
    warmup = int(os.environ.get("WARMUP", "8"))
    dsweep = [int(x) for x in os.environ.get("DSWEEP", "4,8,32").split(",")]
    ckpt_off = os.environ.get("CKPT_OFF", "0") == "1"
    matmul_prec = os.environ.get("MATMUL_PRECISION", "high")

    torch.set_float32_matmul_precision(matmul_prec)
    OmegaConf.set_struct(cfg, False)
    cfg.train.se3_augment = False  # cddb pipeline augments natively (mirror harness)
    device = resolve_device(cfg.hardware.device)
    print(f"device={device}  matmul_precision={matmul_prec}  ckpt_off={ckpt_off}  "
          f"DSWEEP={dsweep}  length_cap={cfg.data.get('length_cap')}  num_workers={cfg.train.num_workers}")

    engine = build_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = attach_spa(net, cfg).to(device)
    set_host_train_mode(net)
    loss_fn = build_loss(cfg).to(device)
    opt = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=cfg.train.lr)

    if ckpt_off:  # monkeypatch foundry's flag (read-only repo) + belt-and-suspenders on every module
        import rfd3.model.layers.blocks as _blocks
        _blocks.DISABLE_CHECKPOINTING = True
        for m in net.modules():
            if hasattr(m, "use_checkpointing"):
                m.use_checkpointing = False
        print("RFD3 activation checkpointing DISABLED (faster backward, more memory)")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    keys = ["fetch", "to_dev", "set_prompt", "forward", "loss", "backward", "opt"]

    def profile_one(D):
        cfg.data.diffusion_batch_size = D
        ds = CDDBPromptDataset(cfg, split="train")
        loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0],
                            num_workers=cfg.train.num_workers, persistent_workers=cfg.train.num_workers > 0)
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
        m = {k: statistics.mean(T[k][warmup:]) for k in keys}
        total = sum(m.values())
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        return {"per": m, "total": total, "peak": peak, "D": statistics.mode(Ds),
                "Lmean": statistics.mean(Ls), "n_struct": len(ds)}

    results = {}
    for D in dsweep:
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        print(f"\n----- profiling D={D} ({n_steps} steps, warmup {warmup}) -----")
        try:
            r = results[D] = profile_one(D)
            print(f"  D={D}: {r['total']*1e3:.0f} ms/step -> {1/r['total']:.2f} struct/s  peak {r['peak']:.1f} GB"
                  f"  (n_struct={r['n_struct']}, Lmean={r['Lmean']:.0f} atoms)")
        except torch.cuda.OutOfMemoryError:
            results[D] = None
            print(f"  D={D}: CUDA OOM (needs checkpointing / less memory) — skipping")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    n_epoch = int(364373 * 0.936)
    print(f"\n===== D-SWEEP SUMMARY (device={device}, ckpt_off={ckpt_off}, TF32={matmul_prec}) =====")
    print(f"  {'D':>4} {'ms/step':>9} {'struct/s':>9} {'peak GB':>8} {'epoch(h)':>9} {'fwd%':>5} {'bwd%':>5} {'fetch%':>7}")
    for D in dsweep:
        r = results.get(D)
        if r is None:
            print(f"  {D:>4} {'OOM':>9}")
            continue
        tot = r["total"]
        print(f"  {D:>4} {tot*1e3:>9.0f} {1/tot:>9.2f} {r['peak']:>8.1f} {n_epoch*tot/3600:>9.1f} "
              f"{r['per']['forward']/tot*100:>5.0f} {r['per']['backward']/tot*100:>5.0f} {r['per']['fetch']/tot*100:>7.1f}")

    print("\n===== DATALOADER MICRO-BENCH (single-thread) =====")
    cfg.data.diffusion_batch_size = dsweep[0]
    ds = CDDBPromptDataset(cfg, split="train")
    row = ds.rows.iloc[0]
    cache_pt = os.path.join(ds.cache_dir, f"{os.path.splitext(row['pdb_file'])[0]}.pt")
    tl = [(lambda a: (torch.load(cache_pt, weights_only=True), time.perf_counter() - a)[1])(time.perf_counter())
          for _ in range(20)]
    print(f"  ESM3-cache .pt load:  {statistics.mean(tl)*1e3:6.1f} ms")
    tf = ds.transform
    ft = []
    for i in range(min(5, len(ds.rows))):
        r = ds.rows.iloc[i]
        a = time.perf_counter()
        _ = tf(load_structure(r["id"], os.path.join(ds.pdb_dir, r["pdb_file"]))); ft.append(time.perf_counter() - a)
    print(f"  featurization (1 struct): {statistics.mean(ft)*1e3:6.0f} ms")
    print("DONE")


if __name__ == "__main__":
    main()
