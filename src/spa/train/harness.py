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

    drop = torch.rand(1, generator=generator).item() < cfg.train.cfg_drop_rate
    if drop or example.get("prompt") is None:      # CFG zero-prompt dropout
        adapter.clear_prompt()
    else:
        adapter.set_prompt(example["prompt"])      # recompute K/V each step (grad -> prompt_kv)

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


def train(cfg) -> None:
    """Run SPA training from a composed Hydra config (local toy-overfit; real data is step 8)."""
    device = resolve_device(cfg.hardware.device)
    engine = build_engine(cfg)
    net = frozen_rfd3_net(engine)

    if cfg.data.name == "toy":
        from ..data.synthetic import capture_synthetic_example
        examples = [capture_synthetic_example(engine, net, cfg, device)]
    else:
        raise NotImplementedError(
            "real CDDB dataset/featurization is kickoff step 8; use data=toy for the local overfit."
        )

    adapter = attach_spa(net, cfg).to(device)
    set_host_train_mode(net)
    loss_fn = build_loss(cfg).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    gen = torch.Generator().manual_seed(cfg.train.seed)

    for step in range(cfg.train.max_steps):
        opt.zero_grad()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = spa_training_step(net, adapter, loss_fn, examples[step % len(examples)], cfg, gen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.train.grad_clip)
        opt.step()
        if step % 10 == 0 or step == cfg.train.max_steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    import os
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    out = os.path.join(cfg.train.ckpt_dir, f"spa_{cfg.variant.name}_last.pt")
    save_spa(adapter, out)
    print(f"saved SPA checkpoint -> {out}")
