"""LOCAL-TESTING-ONLY synthetic training example for the SPA harness (kickoff step 6/7).

Real CDDB featurization is kickoff step 8. To validate the harness mechanism locally without the
heavy training data pipeline, we capture a REAL RFD3 batch (the actual ``f``/``t``/noisy coords the
network consumes — via a forward-pre-hook on ``diffusion_module``, the proven step-4 technique) and
pair it with an **artificial but learnable** target: the vanilla denoised output plus a fixed
per-atom offset. SPA then has a real conditioning gap to learn, so the overfit shows the loss fall
while RFD3 stays frozen — exercising the real ``RFD3.forward`` + real ``DiffusionLoss`` end-to-end.
"""

from __future__ import annotations

import glob
import os

import torch


class _Stop(Exception):
    """Abort the rollout once the first real diffusion_module input is captured."""


def _load_prompt(cfg, device, D: int) -> torch.Tensor:
    """Load a cached ESM3 prompt (or a random stand-in), batched to ``[D, N, 1536]``."""
    files = sorted(glob.glob(os.path.join(cfg.paths.esm3_cache_dir, "*.pt")))
    if files:
        p = torch.load(files[0], weights_only=True).float().to(device)
    else:
        p = torch.randn(32, cfg.model.c_kv, device=device)
    return p[None].expand(D, -1, -1).contiguous()


def capture_synthetic_example(engine, net, cfg, device) -> dict:
    """Capture a real RFD3 batch + build a learnable artificial target. Returns a harness example."""
    dm = net.diffusion_module
    box = {}

    def hook(_m, args, kwargs):
        box["kw"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        raise _Stop

    handle = dm.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        engine.run(inputs=None, out_dir=None)
    except _Stop:
        pass
    finally:
        handle.remove()

    kw = box["kw"]
    x_noisy, t, f = kw["X_noisy_L"], kw["t"], kw["f"]
    D, L = x_noisy.shape[0], x_noisy.shape[1]
    n_tokens = int(f["atom_to_token_map"].max()) + 1

    # Vanilla (frozen-RFD3) output -> artificial target = vanilla + fixed per-atom offset, so SPA has
    # a genuine conditioning gap to close. Host dropout off for a deterministic target.
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        out0 = net.forward(input={"X_noisy_L": x_noisy, "t": t, "f": f}, n_cycle=cfg.train.n_cycle)["X_L"].float()
    g = torch.Generator().manual_seed(cfg.train.seed)
    offset = torch.randn(out0.shape, generator=g).to(device) * cfg.train.synthetic_target_offset
    gt = (out0 + offset).float()

    return {
        "feats": f,
        "t": t,
        "X_noisy_L": x_noisy,
        "X_gt_L_in_input_frame": gt,
        "crd_mask_L": torch.ones(L, dtype=torch.bool, device=device),
        "is_original_unindexed_token": torch.zeros(n_tokens, dtype=torch.bool, device=device),
        "prompt": _load_prompt(cfg, device, D),
    }
