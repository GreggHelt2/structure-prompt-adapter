"""SPA training harness — own loop, RFD3 + ESM3 frozen, gradient on SPA params only.

Kickoff step 6. Spec: dev ``04_training_strategy.md`` §5–§6; RFD3 path mapped in dev ``08`` §6
(step-6 map). RFD3's own inference engine runs a 200-step sampler under ``no_grad``; SPA needs its
own single-step training forward with grad on SPA. We reuse RFD3's pieces rather than reinvent them:

- **Forward:** ``RFD3.forward(input={X_noisy_L, t, f}, n_cycle=...)`` in *training* mode runs ONE
  denoising step (``RFD3.py:70``). Requires ``net.training=True``; we keep the frozen host's dropout
  in eval for a stable signal.
- **Loss:** ``rfd3.metrics.losses.DiffusionLoss`` (the RFD3-native EDM loss), instantiated from the
  checkpoint's loss config (``cfg.train.loss``).
- **Host:** SPA attaches to the net inference runs — the EMA ``shadow`` copy (dev ``08`` §6, step 4)
  — used frozen. The EMA machinery is irrelevant: RFD3 weights never update, only SPA does.

Per step: optionally SE(3)-augment the (input, target) together; CFG zero-prompt dropout; **recompute
the prompt K/V each step** (so gradients reach ``prompt_kv`` — the "compute once" optimization is
per-forward, reused across the 18 blocks, NOT across optimizer steps); ``RFD3.forward`` → loss →
backward → clip → step over ``adapter.parameters()`` only.

Real CDDB training data (featurization) is kickoff step 8; locally we overfit a synthetic example
(``spa.data.synthetic``) to validate the mechanism (loss falls, λ responds, grad on SPA only).
"""

from __future__ import annotations

import torch

from ..data.augment import apply_se3, random_rotation
from ..model import attach_spa
from ..utils.device import resolve_device


def build_engine(cfg):
    """Build + initialize the RFD3 inference engine from ``cfg.paths.rfd3_ckpt`` (loads weights)."""
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    spec = dict(cfg.get("specification") or {})
    spec.setdefault("length", cfg.train.capture_length)  # unconditional design length for capture
    engine = RFD3InferenceEngine(
        **RFD3InferenceConfig(
            ckpt_path=cfg.paths.rfd3_ckpt,
            diffusion_batch_size=cfg.hardware.batch_size,
            specification=spec,
            seed=cfg.train.seed,
        )
    )
    engine.initialize()
    return engine


def frozen_rfd3_net(engine):
    """The RFD3 net inference runs (the EMA ``shadow``), used frozen as the SPA host (dev 08 §6)."""
    return dict(engine.trainer.state["model"].named_modules())["_forward_module.shadow"]


def build_loss(cfg):
    """Instantiate the RFD3-native ``DiffusionLoss`` from ``cfg.train.loss``."""
    from rfd3.metrics.losses import DiffusionLoss

    lc = cfg.train.loss
    return DiffusionLoss(
        sigma_data=lc.sigma_data, weight=lc.weight, lddt_weight=lc.lddt_weight,
        alpha_ligand=lc.alpha_ligand, unindexed_t_alpha=lc.unindexed_t_alpha,
    )


def build_scheduler(opt, cfg):
    """Linear warmup -> cosine decay over ``cfg.train.max_steps`` optimizer steps (dev ``04`` §5).

    LR ramps linearly over ``warmup_steps`` then follows a cosine from the base LR down to
    ``min_lr_ratio × base``. Stepped once per OPTIMIZER step (post grad-accum); its ``state_dict`` is
    checkpointed so a resumed run continues the same schedule (workstream C).
    """
    import math

    warmup = int(cfg.train.get("warmup_steps", 0))
    total = int(cfg.train.max_steps)
    min_ratio = float(cfg.train.get("min_lr_ratio", 0.0))

    def lr_lambda(step):  # step = 0-based optimizer-step index
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if total <= warmup:
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _wo_norm(adapter) -> float:
    """Mean Frobenius norm of the per-block zero-init output projections ``Wo`` — a proxy for how far
    SPA has grown from identity-at-init (dev ``03`` §8); 0 at init, logged to W&B over training."""
    ws = [ca.to_out.weight for ca in adapter.cross_attn]
    return float(torch.stack([w.norm() for w in ws]).mean()) if ws else 0.0


class _Tracker:
    """Thin metric sink so the train loop never branches on the backend: ``tracker=wandb`` logs to
    Weights & Biases (run config = hyperparams + provenance, dev ``04`` §11), ``tracker=null`` is a
    no-op. ``resume='allow'`` + a stable ``wandb_id`` reattaches the same run after a Vertex restart."""

    def __init__(self, cfg, provenance):
        self.run = None
        kind = str(cfg.train.get("tracker", None) or "null").lower()
        if kind != "wandb":
            return
        import wandb

        run_cfg = {
            "variant": cfg.variant.name,
            "conditioning": cfg.data.get("conditioning", None),
            "lr": cfg.train.lr,
            "max_steps": cfg.train.max_steps,
            "grad_accum": cfg.hardware.get("grad_accum", 1),
            "diffusion_batch_size": cfg.data.get("diffusion_batch_size", None),
            "length_cap": cfg.data.get("length_cap", None),
            "cfg_drop_rate": cfg.train.cfg_drop_rate,
            "sampler": cfg.train.get("sampler", None),
            "warmup_steps": cfg.train.get("warmup_steps", None),
            **{f"prov/{k}": v for k, v in provenance.items() if k != "split_meta"},
        }
        wid = cfg.train.get("wandb_id", None)
        self.run = wandb.init(
            entity=cfg.train.get("wandb_entity", None),
            project=cfg.train.get("wandb_project", None),
            name=cfg.get("run_name", None),
            id=(str(wid) if wid else None),
            resume="allow",
            config=run_cfg,
            tags=[cfg.variant.name, str(cfg.data.get("conditioning", "na"))],
        )

    def log(self, metrics, step):
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()


def set_host_train_mode(net) -> None:
    """Enable RFD3's training-forward branch (``net.training=True``) but keep host dropout in eval."""
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def spa_training_step(net, adapter, loss_fn, example, cfg, generator=None, drop_override=None):
    """One SPA training forward+loss on an example dict. Returns ``(loss, loss_dict)``.

    Example keys: ``feats``, ``t``, ``X_noisy_L``, ``X_gt_L_in_input_frame``, ``crd_mask_L``,
    ``is_original_unindexed_token``, ``prompt`` (``[D,N,1536]`` or ``None``).
    ``drop_override``: force the CFG zero-prompt branch on/off (``None`` -> random per
    ``cfg_drop_rate``); validation passes ``False`` to always measure the *conditioned* loss.
    """
    f, t = example["feats"], example["t"]
    x_noisy, gt = example["X_noisy_L"], example["X_gt_L_in_input_frame"]

    if cfg.train.se3_augment:                      # rotate input + target together
        R = random_rotation(generator)
        x_noisy, gt = apply_se3(x_noisy, R), apply_se3(gt, R)

    prompt = example.get("prompt")
    drop = (bool(drop_override) if drop_override is not None
            else torch.rand(1, generator=generator).item() < cfg.train.cfg_drop_rate)
    if prompt is None:                             # no prompt available (e.g. toy) -> base only
        adapter.clear_prompt()
    elif drop:                                     # CFG zero-prompt dropout -> learned null token e∅
        adapter.set_null_prompt(prompt.shape[0])   # keeps SPA live so grad reaches the adapter (dev 11 §6)
    else:                                          # recompute K/V each step (grad -> prompt_kv);
        adapter.set_prompt(prompt, key_padding_mask=example.get("prompt_mask"))  # non-overlap mask

    out = net.forward(input={"X_noisy_L": x_noisy, "t": t, "f": f},
                      n_cycle=cfg.train.n_cycle)["X_L"].float()
    return loss_fn(
        network_input={"f": f, "t": t},
        network_output={"X_L": out},
        loss_input={"crd_mask_L": example["crd_mask_L"],
                    "is_original_unindexed_token": example["is_original_unindexed_token"],
                    "X_gt_L_in_input_frame": gt},
    )


def save_spa(adapter, path) -> None:
    """Checkpoint only the SPA parameters (the adapter ModuleList)."""
    torch.save(adapter.state_dict(), path)


def load_spa(adapter, path) -> None:
    adapter.load_state_dict(torch.load(path, weights_only=True))


def gather_provenance(cfg) -> dict:
    """Best-effort run provenance stamped into every checkpoint (and the W&B run config, workstream
    C): which split / ESM3 cache / RFD3 ckpt / git commit produced these weights (dev ``04`` §11
    reproducibility). Never raises — provenance is informational, not load-bearing."""
    import json
    import os
    import subprocess

    prov = {
        "variant": cfg.variant.name,
        "conditioning": cfg.data.get("conditioning", None),
        "esm3_cache_dir": cfg.paths.get("esm3_cache_dir", None),
        "rfd3_ckpt": cfg.paths.get("rfd3_ckpt", None),
        "split_id": None,
        "git_commit": None,
    }
    for cand in (os.path.join(str(cfg.data.get("splits_root", "")), "split_meta.json"),
                 os.path.join(str(cfg.paths.get("data_root", "")), "processed", "split_meta.json")):
        try:
            if cand and os.path.exists(cand):
                with open(cand) as fh:
                    meta = json.load(fh)
                prov["split_id"] = meta.get("split_id", meta.get("name"))
                prov["split_meta"] = meta
                break
        except Exception:
            pass
    try:  # git commit of THIS (public) repo — harness.py is at <repo>/src/spa/train/harness.py
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        prov["git_commit"] = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    return prov


def save_checkpoint(path, adapter, opt, sched, step, samples_seen, gen, cfg, provenance) -> None:
    """Atomic full-state checkpoint for resume: adapter + optimizer + scheduler + step + sample
    counter + RNG (torch / cuda / sampling generator) + provenance (dev ``04`` §8). Written via a
    ``.tmp`` rename so a crash mid-write never corrupts the rolling ``last.pt``."""
    import os

    ckpt = {
        "adapter": adapter.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": int(step),
        "samples_seen": int(samples_seen),
        "gen": gen.get_state(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "provenance": provenance,
        "variant": cfg.variant.name,
    }
    tmp = f"{path}.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(path, adapter, opt, sched, gen, device):
    """Restore a :func:`save_checkpoint` into the live adapter / opt / sched / gen + RNG. Returns
    ``(step, samples_seen)`` of the LAST completed optimizer step — the caller resumes at ``step+1``.

    Loads to CPU first (RNG byte-tensors must stay on CPU), then moves the Adam moments onto the
    param device so a resumed ``opt.step`` doesn't hit a device mismatch."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ck["adapter"])  # adapter params stay on their device (cross-device copy)
    opt.load_state_dict(ck["optimizer"])
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)
    sched.load_state_dict(ck["scheduler"])
    gen.set_state(ck["gen"])
    rng = ck.get("rng") or {}
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(ck["step"]), int(ck.get("samples_seen", 0))


def _single(batch):
    """DataLoader collate for batch_size=1: pass the one example dict through unstacked (each example
    is already a ``D``-wide diffusion batch; structures have different ``L`` so they can't be stacked)."""
    return batch[0]


def _length_stratified_sampler(rows, n_bins, generator):
    """``WeightedRandomSampler`` with inverse-bin-frequency weights over equal-width length bins, so
    each length band is drawn ~equally often (balanced size coverage for a sub-epoch budget; dev
    ``04`` §11). ``replacement=True`` upsamples rare (long) lengths."""
    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    lengths = rows["length"].to_numpy()
    n = len(lengths)
    lo, hi = float(lengths.min()), float(lengths.max())
    if hi <= lo or n_bins < 2:
        weights = np.ones(n, dtype=np.float64)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        b = np.clip(np.digitize(lengths, edges[1:-1]), 0, n_bins - 1)
        counts = np.bincount(b, minlength=n_bins).astype(np.float64)
        weights = 1.0 / counts[b]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=n, replacement=True, generator=generator)


def _example_stream(cfg, engine, net, device):
    """Infinite iterator of on-device example dicts. ``toy`` -> one synthetic example; else the real
    ``CDDBPromptDataset`` via a DataLoader (live featurization in workers, moved to device here)."""
    if cfg.data.name == "toy":
        from ..data.synthetic import capture_synthetic_example

        example = capture_synthetic_example(engine, net, cfg, device)
        while True:
            yield example
    else:
        from torch.utils.data import DataLoader

        from ..data.dataset import CDDBPromptDataset, move_to_device

        ds = CDDBPromptDataset(cfg, split="train")
        if len(ds) == 0:
            raise RuntimeError(
                "CDDBPromptDataset(train) is empty — check the split manifest + ESM3 cache "
                f"(require_cached_prompt={cfg.data.get('require_cached_prompt', False)})."
            )
        nw = int(cfg.train.get("num_workers", 2))
        kind = str(cfg.train.get("sampler", "stratified")).lower()
        if kind == "stratified":
            g = torch.Generator().manual_seed(int(cfg.train.get("seed", 0)))
            sampler = _length_stratified_sampler(ds.rows, int(cfg.train.get("sampler_bins", 20)), g)
            loader = DataLoader(ds, batch_size=1, sampler=sampler, collate_fn=_single,
                                num_workers=nw, persistent_workers=nw > 0)
            print(f"sampler=stratified over {len(ds)} structures, "
                  f"{int(cfg.train.get('sampler_bins', 20))} length bins")
        else:
            loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_single,
                                num_workers=nw, persistent_workers=nw > 0)
        while True:
            for ex in loader:
                yield move_to_device(ex, device)


def validate(cfg, net, adapter, loss_fn, device, n_batches=8):
    """Mean *conditioned* val-loss over ``n_batches`` from the ``validate`` split (no grad, no CFG
    drop) — the early-stopping signal (dev ``04`` §8). Returns ``None`` if the split is empty."""
    from torch.utils.data import DataLoader

    from ..data.dataset import CDDBPromptDataset, move_to_device

    ds = CDDBPromptDataset(cfg, split="validate")
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_single, num_workers=0)
    losses, it = [], iter(loader)
    for _ in range(n_batches):
        try:
            ex = move_to_device(next(it), device)
        except StopIteration:
            break
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = spa_training_step(net, adapter, loss_fn, ex, cfg, drop_override=False)
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else None


def train(cfg) -> None:
    """Run SPA training from a composed Hydra config. ``data=toy`` overfits a synthetic example;
    ``data=cddb`` streams real featurized CDDB examples (kickoff step 8 Part B)."""
    import os

    torch.set_float32_matmul_precision(str(cfg.train.get("matmul_precision", "high")))  # TF32 (dev 04 §11)
    device = resolve_device(cfg.hardware.device)
    engine = build_engine(cfg)
    net = frozen_rfd3_net(engine)

    # The real CDDB pipeline applies RFD3-native SE(3) augmentation; the harness must NOT also augment
    # (double aug). The toy/synthetic path has no native aug, so it keeps cfg's setting.
    if cfg.data.name != "toy" and cfg.train.se3_augment:
        from omegaconf import OmegaConf

        OmegaConf.set_struct(cfg, False)
        cfg.train.se3_augment = False
        print("note: data != toy -> forcing train.se3_augment=False (the pipeline augments natively)")

    adapter = attach_spa(net, cfg).to(device)
    set_host_train_mode(net)
    loss_fn = build_loss(cfg).to(device)
    # train only requires_grad params — a variant may include a FROZEN encoder in the adapter
    # (e.g. variant-A's CLSS structure_adapter), which must stay out of the optimizer.
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = build_scheduler(opt, cfg)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    stream = _example_stream(cfg, engine, net, device)

    # Resume = auto: if a rolling last.pt exists, restore full state and continue at the next step
    # (the cloud job rsyncs it DOWN from GCS at start — run_train.sh — so a preempted/restarted Vertex
    # job picks up where it left off; dev ``04`` §8). `clear_prompt`/identity-at-init are untouched.
    provenance = gather_provenance(cfg)
    samples_seen, start_step = 0, 0
    last_path = os.path.join(cfg.train.ckpt_dir, f"spa_{cfg.variant.name}_last.pt")
    if str(cfg.train.get("resume", "auto")).lower() in ("auto", "true", "1") and os.path.exists(last_path):
        start_step, samples_seen = load_checkpoint(last_path, adapter, opt, sched, gen, device)
        start_step += 1  # the saved step is the last COMPLETED one; resume on the next
        print(f"resumed from {last_path}: continuing at optimizer step {start_step} "
              f"(samples_seen={samples_seen})")
    if start_step >= cfg.train.max_steps:
        print(f"run already complete ({start_step} >= max_steps={cfg.train.max_steps}); nothing to do")
        return
    ckpt_every = int(cfg.train.get("ckpt_every_steps", 0))
    val_every = int(cfg.train.get("val_every_steps", 0))
    do_val = val_every > 0 and cfg.data.name != "toy"  # validate needs the CDDB validate split
    log_every = int(cfg.train.get("log_every_steps", 10))
    tracker = _Tracker(cfg, provenance)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    # Effective batch = ``grad_accum`` micro-steps (each one structure == a D-wide diffusion batch)
    # per OPTIMIZER step; ``max_steps`` counts optimizer steps and the run is driven by a step/time
    # budget, not epochs (sub-epoch reality, dev ``04`` §11). The toy path overfits ONE repeated
    # example, so accumulation there only rescales the effective LR -> force grad_accum=1 (keeps the
    # sanity run fast + behaviour-equivalent), mirroring the se3_augment special-case above.
    grad_accum = 1 if cfg.data.name == "toy" else int(cfg.hardware.get("grad_accum", 1))
    max_hours = cfg.train.get("max_hours", None)

    import time

    t0 = time.monotonic()
    for step in range(start_step, cfg.train.max_steps):
        opt.zero_grad()
        accum_loss, n_back = 0.0, 0
        for _ in range(grad_accum):
            example = next(stream)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss, _ = spa_training_step(net, adapter, loss_fn, example, cfg, gen)
            # With the learned null token a CFG-dropped step still has a gradient (dev ``11`` §6); the
            # only no-grad case left is a genuinely prompt-free forward (e.g. toy with no prompt) ==
            # vanilla frozen RFD3, which has nothing to teach SPA -> skip its backward (the B1 guard,
            # else backward raises "element 0 ... does not require grad").
            if not loss.requires_grad:
                continue
            (loss / grad_accum).backward()
            accum_loss += loss.item()
            n_back += 1
        lr_now = sched.get_last_lr()[0]  # the LR actually applied at this step (before advancing)
        gnorm = 0.0
        if n_back > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.train.grad_clip))
            opt.step()
        sched.step()
        samples_seen += grad_accum
        if step % log_every == 0 or step == cfg.train.max_steps - 1:
            avg = accum_loss / n_back if n_back else float("nan")
            print(f"step {step:5d} | loss {avg:.4f} | lr {lr_now:.2e}"
                  f"{'  (no-grad: skipped)' if n_back == 0 else ''}")
            tracker.log({"train/loss": avg, "lr": lr_now, "grad_norm": gnorm,
                         "spa/Wo_norm": _wo_norm(adapter), "samples_seen": samples_seen}, step=step)
        if ckpt_every and (step + 1) % ckpt_every == 0:
            save_checkpoint(last_path, adapter, opt, sched, step, samples_seen, gen, cfg, provenance)
            snap = os.path.join(cfg.train.ckpt_dir, f"spa_{cfg.variant.name}_step{step + 1}.pt")
            save_checkpoint(snap, adapter, opt, sched, step, samples_seen, gen, cfg, provenance)
            print(f"  checkpoint @ step {step + 1} -> last.pt + {os.path.basename(snap)}")
        if do_val and (step + 1) % val_every == 0:
            vloss = validate(cfg, net, adapter, loss_fn, device, int(cfg.train.get("val_batches", 8)))
            if vloss is not None:
                print(f"  val @ step {step + 1} | val_loss {vloss:.4f}")
                tracker.log({"val/loss": vloss}, step=step)
        if max_hours is not None and (time.monotonic() - t0) >= float(max_hours) * 3600:
            print(f"time budget {max_hours}h reached at optimizer step {step}")
            break

    save_checkpoint(last_path, adapter, opt, sched, step, samples_seen, gen, cfg, provenance)
    export = os.path.join(cfg.train.ckpt_dir, f"spa_{cfg.variant.name}_final.pt")
    save_spa(adapter, export)  # adapter-only export for inference/validation (no optimizer state)
    print(f"done at optimizer step {step} | resume={os.path.basename(last_path)} "
          f"export={os.path.basename(export)} | samples_seen={samples_seen}")
    tracker.finish()
