"""Phase-0 CFG probe: how much of each coordinate step does the SPA prompt account for?

Dev spec: ``docs/plan/56_cfg_guidance_for_spa.md`` §2.6a. This is the ~1-hour, $0 go/no-go check
that strictly precedes building a real classifier-free-guidance path for SPA, and it needs **no CFG
machinery at all**.

WHY THIS QUANTITY. If SPA ever gets a guidance scale ω, RFD3's CFG update is

    delta_new = delta_cond + (ω - 1)(delta_cond - delta_ref)        [inference_sampler.py:307]

so ``(ω - 1)`` multiplies exactly ``delta_cond - delta_ref``, where ``delta_cond`` is the coordinate
delta with the real prompt and ``delta_ref`` the delta with SPA's learned null token. Measure that
difference and you know, before writing a line of sampler code, whether there is anything for ω to
amplify. Around 1% of the step and even ω=3 adds ~2%: the lever is inert (`56` §5 Gain 3's
saturation branch), reached in an hour instead of two days.

THE ALGEBRA IS KINDER THAN THE SPEC ASSUMED. With ``delta = (X_noisy - X_denoised) / t_hat``,

    delta_cond - delta_ref = (X_ref - X_cond) / t_hat

so the ratio to the conditional step is

    ||delta_cond - delta_ref|| / ||delta_cond|| = ||X_ref - X_cond|| / ||X_noisy - X_cond||

and ``t_hat`` cancels entirely. We therefore need no assumption about the sampler's t convention,
which matters because Supp Eq. 5 and the code disagree by one on the scale (`07` I.11). Both forms
are computed and asserted equal as a self-check.

HOW IT WORKS. Rather than reconstructing featurized inputs, this wraps ``net.diffusion_module`` and
rides a REAL trajectory: on every sampler step it lets the conditional forward through UNCHANGED
(so the trajectory stays a normal SPA trajectory) and runs one extra forward with the null prompt
purely to record the diagnostic. The extra pass is bracketed by RNG save/restore so it cannot shift
the sampler's noise draws.

CONTROL. ``--control`` makes the "real" prompt the null token as well, so both passes are identical
and every ratio must be exactly 0. That validates the probe rather than the hypothesis; run it once.

Usage (A5000, ~minutes):
    conda run -n spa-dev python scripts/eval/probe_cfg_delta.py \
        --prompt-pdb <prompt.pdb> --ckpt models/spa-Nx1536-uncond.pt --length 100 --out probe.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _rfd3_ckpt() -> str:
    import os

    root = os.environ.get("SPA_PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
    return str(Path(root) / "models" / "rfdiffusion3" / "rfd3_latest.ckpt")


def _seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_probe_class():
    """Build the probe class lazily, since it must subclass ``nn.Module``.

    ``net.diffusion_module`` is a *registered submodule*, and torch's ``__setattr__`` refuses to
    accept a plain object in a submodule slot, so the wrapper has to be an ``nn.Module`` itself.
    Registering the wrapped module under a NON-submodule name (``object.__setattr__``) keeps it out
    of the parent's child list, so no parameter is double-counted and nothing in the host's state
    dict changes.
    """
    import torch.nn as nn

    class DeltaProbe(nn.Module):
        """Wraps ``diffusion_module`` to record ||X_ref - X_cond|| / ||X_noisy - X_cond|| per step.

        The conditional output is returned untouched, so the sampler proceeds exactly as it would
        without the probe. Only the *extra* null-prompt forward is added, and the RNG state is saved
        and restored around it so the sampler's subsequent noise draws are unaffected.
        """

        def __init__(self, module, adapter, batch: int, control: bool = False):
            super().__init__()
            object.__setattr__(self, "_module", module)     # not a child: avoid re-registering
            object.__setattr__(self, "_adapter", adapter)
            self._batch = batch
            self._control = control
            self.rows: list[dict] = []

        def forward(self, *args, **kwargs):
            return self._call(*args, **kwargs)

        def _call(self, *args, **kwargs):
            import torch

            outs = self._module(*args, **kwargs)      # the real conditional pass, untouched

            X_noisy = kwargs.get("X_noisy_L")
            t = kwargs.get("t")
            if X_noisy is None or not isinstance(outs, dict) or "X_L" not in outs:
                return outs                                # not a shape we can measure; pass through

            ctx = self._adapter._context
            saved = (ctx.k, ctx.v, ctx.key_padding_mask, ctx.prompts)
            rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                self._adapter.set_null_prompt(self._batch)
                with torch.no_grad():
                    outs_ref = self._module(*args, **kwargs)
            finally:
                ctx.k, ctx.v, ctx.key_padding_mask, ctx.prompts = saved
                torch.set_rng_state(rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)

            # A THIRD pass with the SAME (unchanged) inputs and prompt. RFD3's fast reduced-precision
            # GPU path is not bitwise reproducible (dev plan/RFD3_irreproducibility), so two identical
            # forwards differ. This pass measures that numerical floor per step, in the SAME
            # trajectory, perfectly paired, which is the only fair reference for the prompt effect.
            rng2 = torch.get_rng_state()
            cuda_rng2 = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                with torch.no_grad():
                    outs_rep = self._module(*args, **kwargs)
            finally:
                torch.set_rng_state(rng2)
                if cuda_rng2 is not None:
                    torch.cuda.set_rng_state_all(cuda_rng2)

            Xc = outs["X_L"].detach().float()
            Xr = outs_ref["X_L"].detach().float()
            X2 = outs_rep["X_L"].detach().float()
            Xn = X_noisy.detach().float()

            step = Xn - Xc                                 # proportional to delta_cond
            diff = Xr - Xc                                 # proportional to delta_cond - delta_ref
            noise = X2 - Xc                                # the same-input numerical floor
            n_step = torch.linalg.vector_norm(step).item()
            n_diff = torch.linalg.vector_norm(diff).item()
            n_noise = torch.linalg.vector_norm(noise).item()
            ratio = n_diff / n_step if n_step > 0 else float("nan")
            ratio_noise = n_noise / n_step if n_step > 0 else float("nan")
            snr = (n_diff / n_noise) if n_noise > 0 else float("inf")

            # Same ratio via the explicit deltas, to confirm t_hat really does cancel.
            t_scalar = float(t.flatten()[0].item()) if t is not None else float("nan")
            ratio_delta = float("nan")
            if t is not None and math.isfinite(t_scalar) and t_scalar != 0:
                d_cond = step / t_scalar
                d_ref = (Xn - Xr) / t_scalar
                nc = torch.linalg.vector_norm(d_cond).item()
                ratio_delta = torch.linalg.vector_norm(d_cond - d_ref).item() / nc if nc > 0 else float("nan")

            per_atom = torch.linalg.vector_norm(diff, dim=-1)        # [D, L] displacement magnitudes, Å
            self.rows.append({
                "step": len(self.rows),
                "t": t_scalar,
                "ratio": ratio,                     # the headline number
                "ratio_via_delta": ratio_delta,     # self-check, must match `ratio`
                "ratio_noise": ratio_noise,         # same-input GPU-nondeterminism floor
                "snr": snr,                         # prompt effect / numerical floor
                "step_norm": n_step,
                "diff_norm": n_diff,
                "noise_norm": n_noise,
                "prompt_shift_rms_A": float(per_atom.pow(2).mean().sqrt().item()),
                "prompt_shift_max_A": float(per_atom.max().item()),
                "n_atoms": int(per_atom.shape[-1]),
            })
            return outs

    return DeltaProbe


def _force_determinism():
    """Try to make repeat forwards bit-identical, so the noise floor collapses to zero.

    dev ``plan/RFD3_irreproducibility.md`` attributes RFD3's non-reproducibility to the fast
    reduced-precision GPU path (``matmul_precision("medium")`` + bf16 autocast) making FP reductions
    non-associative in scheduler-dependent order, and notes that ``use_deterministic_algorithms``
    throws. We try anyway with ``warn_only=True``, because for THIS probe a zero noise floor would
    remove the whole interpretive burden. Returns a dict of what actually took effect.
    """
    import os

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    applied = {}
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        applied["use_deterministic_algorithms"] = True
    except Exception as e:                                   # noqa: BLE001
        applied["use_deterministic_algorithms"] = f"FAILED: {e}"
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True
    except Exception as e:                                   # noqa: BLE001
        applied["cudnn_deterministic"] = f"FAILED: {e}"
    try:
        torch.set_float32_matmul_precision("highest")         # undo foundry's "medium"
        applied["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception as e:                                   # noqa: BLE001
        applied["float32_matmul_precision"] = f"FAILED: {e}"
    applied["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return applied


def run(prompt_pdb, ckpt, length, timesteps, seed, K, device, out_json, control,
        deterministic=False):
    import torch
    from omegaconf import OmegaConf

    det_info = None
    if deterministic:
        det_info = _force_determinism()
        print(f"[probe] determinism attempt: {det_info}")

    from spa.eval.generate import build_eval_engine, load_adapter, resolve_prompt
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    dev = resolve_device(device)
    cfg = OmegaConf.create({
        "paths": {"rfd3_ckpt": _rfd3_ckpt()},
        "hardware": {"device": device},
        "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
        "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False},
        "eval": {"num_designs": K, "length": length, "specification": None,
                 "num_timesteps": timesteps, "seed": seed, "ckpt": ckpt,
                 "prompt_pdb": str(prompt_pdb), "prompt_cache": None, "use_sequence": False},
    })

    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, dev)
    adapter.eval()
    adtype = next(adapter.parameters()).dtype

    p = resolve_prompt(cfg, dev)                                   # [N, 1536]
    print(f"[probe] prompt {Path(prompt_pdb).name}: N={p.shape[0]}, design length L={length}")

    if control:
        adapter.set_null_prompt(K)
        print("[probe] CONTROL: both passes use the null token; every ratio must be exactly 0.")
    else:
        adapter.set_prompt(p[None].expand(K, -1, -1).to(device=dev, dtype=adtype).contiguous())
    adapter.set_scale(1.0)

    DeltaProbe = _make_probe_class()
    probe = DeltaProbe(net.diffusion_module, adapter, batch=K, control=control)
    net.diffusion_module = probe                                   # ride the real trajectory
    try:
        _seed_all(seed)
        with torch.no_grad():
            engine.run(inputs=None, out_dir=None)
    finally:
        net.diffusion_module = probe._module

    rows = probe.rows
    if not rows:
        raise SystemExit("[probe] no steps recorded — the wrapper never saw a measurable call.")

    def _fin(key):
        return [r[key] for r in rows if math.isfinite(r[key])]

    ratios, noises, snrs = _fin("ratio"), _fin("ratio_noise"), _fin("snr")
    mismatch = max(abs(r["ratio"] - r["ratio_via_delta"])
                   for r in rows if math.isfinite(r["ratio_via_delta"]))
    med = lambda v: sorted(v)[len(v) // 2]
    summary = {
        "n_steps": len(rows),
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_median": med(ratios),
        "ratio_max": max(ratios),
        "ratio_min": min(ratios),
        "ratio_first": rows[0]["ratio"],
        "ratio_last": rows[-1]["ratio"],
        "noise_floor_mean": sum(noises) / len(noises),
        "noise_floor_median": med(noises),
        "snr_mean": sum(snrs) / len(snrs),
        "snr_median": med(snrs),
        "snr_min": min(snrs),
        "prompt_shift_rms_A_max": max(r["prompt_shift_rms_A"] for r in rows),
        "identity_check_max_abs_diff": mismatch,
        "control": bool(control),
    }

    print("\n[probe] === per-step (every 5th) ===")
    print(f"{'step':>5} {'t':>10} {'ratio':>9} {'noise':>9} {'SNR':>8} {'rms Å':>9} {'max Å':>9}")
    for r in rows[::5]:
        print(f"{r['step']:>5} {r['t']:>10.3f} {r['ratio']:>9.4f} {r['ratio_noise']:>9.4f} "
              f"{r['snr']:>8.2f} {r['prompt_shift_rms_A']:>9.3f} {r['prompt_shift_max_A']:>9.3f}")

    print("\n[probe] === summary ===")
    for k, v in summary.items():
        print(f"  {k:34s} {v}")
    print(f"\n[probe] t_hat-cancellation self-check: max |ratio - ratio_via_delta| = {mismatch:.3e}")
    print("[probe] READ THE RATIO AGAINST THE NOISE FLOOR, not against zero: RFD3's GPU path is not")
    print("[probe] bitwise reproducible, so two identical forwards differ (plan/RFD3_irreproducibility).")
    print(f"[probe] median SNR = {summary['snr_median']:.2f}x the numerical floor.")
    print("[probe] what (omega-1) x ratio adds to each step, at the median ratio:")
    for w in (1.5, 2.0, 3.0):
        print(f"    omega={w:<4} -> +{(w - 1) * summary['ratio_median'] * 100:6.2f}% of the step")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(
            {"summary": summary, "per_step": rows,
             "config": {"prompt_pdb": str(prompt_pdb), "ckpt": ckpt, "length": length,
                        "timesteps": timesteps, "seed": seed, "K": K, "control": control, "deterministic": deterministic,
                        "determinism_applied": det_info}},
            indent=2))
        print(f"\n[probe] wrote {out_json}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-pdb", required=True)
    ap.add_argument("--ckpt", required=True, help="trained SPA adapter .pt")
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=None, help="None -> checkpoint default (100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-K", "--num-designs", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--control", action="store_true",
                    help="null-vs-null: validates the probe, every ratio must be exactly 0")
    ap.add_argument("--deterministic", action="store_true",
                    help="try to force bitwise-reproducible forwards so the noise floor is 0")
    a = ap.parse_args()
    run(a.prompt_pdb, a.ckpt, a.length, a.timesteps, a.seed, a.num_designs,
        a.device, a.out, a.control, a.deterministic)


if __name__ == "__main__":
    main()
