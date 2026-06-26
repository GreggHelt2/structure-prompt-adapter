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


def set_host_train_mode(net) -> None:
    """Enable RFD3's training-forward branch (``net.training=True``) but keep host dropout in eval."""
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def spa_training_step(net, adapter, loss_fn, example, cfg, generator=None):
    """One SPA training forward+loss on an example dict. Returns ``(loss, loss_dict)``.

    Example keys: ``feats``, ``t``, ``X_noisy_L``, ``X_gt_L_in_input_frame``, ``crd_mask_L``,
    ``is_original_unindexed_token``, ``prompt`` (``[D,N,1536]`` or ``None``).
    """
    f, t = example["feats"], example["t"]
    x_noisy, gt = example["X_noisy_L"], example["X_gt_L_in_input_frame"]

    if cfg.train.se3_augment:                      # rotate input + target together
        R = random_rotation(generator)
        x_noisy, gt = apply_se3(x_noisy, R), apply_se3(gt, R)

    prompt = example.get("prompt")
    drop = torch.rand(1, generator=generator).item() < cfg.train.cfg_drop_rate
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


def _single(batch):
    """DataLoader collate for batch_size=1: pass the one example dict through unstacked (each example
    is already a ``D``-wide diffusion batch; structures have different ``L`` so they can't be stacked)."""
    return batch[0]


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
        loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_single,
                            num_workers=nw, persistent_workers=nw > 0)
        while True:
            for ex in loader:
                yield move_to_device(ex, device)


def train(cfg) -> None:
    """Run SPA training from a composed Hydra config. ``data=toy`` overfits a synthetic example;
    ``data=cddb`` streams real featurized CDDB examples (kickoff step 8 Part B)."""
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

    # Effective batch = ``grad_accum`` micro-steps (each one structure == a D-wide diffusion batch)
    # per OPTIMIZER step; ``max_steps`` counts optimizer steps and the run is driven by a step/time
    # budget, not epochs (sub-epoch reality, dev ``04`` §11). The toy path overfits ONE repeated
    # example, so accumulation there only rescales the effective LR -> force grad_accum=1 (keeps the
    # sanity run fast + behaviour-equivalent), mirroring the se3_augment special-case above.
    grad_accum = 1 if cfg.data.name == "toy" else int(cfg.hardware.get("grad_accum", 1))
    max_hours = cfg.train.get("max_hours", None)

    import time

    t0 = time.monotonic()
    for step in range(cfg.train.max_steps):
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
        if n_back > 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.train.grad_clip)
            opt.step()
        sched.step()
        if step % 10 == 0 or step == cfg.train.max_steps - 1:
            avg = accum_loss / n_back if n_back else float("nan")
            print(f"step {step:5d} | loss {avg:.4f} | lr {lr_now:.2e}"
                  f"{'  (no-grad: skipped)' if n_back == 0 else ''}")
        if max_hours is not None and (time.monotonic() - t0) >= float(max_hours) * 3600:
            print(f"time budget {max_hours}h reached at optimizer step {step}")
            break

    import os
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    out = os.path.join(cfg.train.ckpt_dir, f"spa_{cfg.variant.name}_last.pt")
    save_spa(adapter, out)
    print(f"saved SPA checkpoint -> {out}")
