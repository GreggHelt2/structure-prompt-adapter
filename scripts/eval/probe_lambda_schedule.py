"""Sweep λ schedules over diffusion steps at matched mean λ (dev ``docs/plan/57_lambda_schedule.md``).

Runs one arm per shape from ``spa.model.lambda_schedule``, all at the same **mean** λ so the comparison
tests *distribution* rather than *total conditioning strength*, plus a λ=0 baseline. Adherence-only by
default: no ProteinMPNN, no OpenFold3, so Phase A costs nothing but A5000 time.

SCORING IS IN THIS DRIVER, DELIBERATELY
Phase A's two metrics are both CPU-only and need no pipeline: **adherence** is TM-score of each design
against the prompt's own structure (:func:`spa.eval.score.adherence`) and **diversity** is the mean
pairwise TM within an arm (:func:`spa.eval.score.pairwise_tm_diversity`). Both are imported from the
shared scorer rather than reimplemented, so a Phase-A number is computed by the same code path as every
other adherence number in the project. ``run_flywheel`` is NOT usable here: its Stage 1 calls
``generate(cfg)`` unconditionally and has no per-step λ hook, so it cannot be pointed at these arm
directories. Its Stages 2 to 4 *can* be (``inverse_fold`` and ``score_design`` both take injected
designs, which is how ``score_threeway_designability.py`` scores pre-existing backbones), and that is
the route Phase B takes.

⭐ THE MATCHED-MEAN GUARANTEE IS ENFORCED AT RUNTIME, NOT JUST IN THE ARRAY
A schedule array with the right mean is worthless if the sampler consumes a different number of entries
than it has. `results/28` measured **99** ``diffusion_module`` calls for a ``num_timesteps=100`` run, i.e.
``num_timesteps - 1``, so the schedule is built to that length. Because that rule is measured rather than
documented, every arm **asserts** that the realized mean matches the target and **aborts the arm** if it
does not, rather than reporting a number whose comparability has quietly lapsed. An asymmetric shape is
where this bites: dropping ``linear_decay``'s final zero would *raise* its realized mean.

Usage (Phase A, adherence + diversity, no OF3):
    conda run -n spa-dev python scripts/eval/probe_lambda_schedule.py \\
        --prompt-pdb <prompt.pdb> --ckpt models/spa-Nx1536-uncond.pt \\
        --length 150 -K 8 --target-lambda 1.0 --out-dir <dir> --json <results.json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
_PROJECT = _HERE.parents[3]


def _seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-pdb", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--length", type=int, default=150)
    ap.add_argument("-K", "--num-designs", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=None, help="None -> checkpoint default (100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-lambda", type=float, default=1.0)
    ap.add_argument("--shapes", default=None, help="comma-separated; default = all")
    ap.add_argument("--support", default="0.35,0.85", help="trajectory-fraction support a,b")
    ap.add_argument("--teeth", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(_PROJECT / "outputs" / "_incoming" / "lambda_schedule"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |realized mean - target| before an arm is aborted as non-comparable")
    ap.add_argument("--no-score", action="store_true",
                    help="write PDBs only, skip the CPU adherence+diversity scoring (default: score)")
    ap.add_argument("--hook-const", default="0",
                    help="comma-separated λ values run as EXTRA flat arms THROUGH the hook, as controls "
                         "rather than comparison arms (they are exempt from the matched-mean gate). "
                         "λ=0 is the plumbing positive control: with the prompt set but every step "
                         "driven to 0, adherence MUST fall back to the baseline. If it instead matches "
                         "constant λ=1, the hook is not transmitting λ and no arm means anything. "
                         "Empty string disables.")
    a = ap.parse_args()

    import torch
    from omegaconf import OmegaConf

    from spa.eval.generate import (_run_once, build_eval_engine, load_adapter,
                                   resolve_prompt, write_pdb)
    from spa.eval.score import _as_struct, adherence, pairwise_tm_diversity
    from spa.model.lambda_schedule import ScheduledLambda, build, shape_names
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    support = tuple(float(x) for x in a.support.split(","))
    dev = resolve_device(a.device)

    cfg = OmegaConf.create({
        "paths": {"rfd3_ckpt": str(_PROJECT / "models" / "rfdiffusion3" / "rfd3_latest.ckpt")},
        "hardware": {"device": a.device},
        "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
        "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False},
        "eval": {"num_designs": a.num_designs, "length": a.length, "specification": None,
                 "num_timesteps": a.timesteps, "seed": a.seed, "ckpt": a.ckpt,
                 "prompt_pdb": str(a.prompt_pdb), "prompt_cache": None, "use_sequence": False},
    })

    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, dev); adapter.eval()
    adtype = next(adapter.parameters()).dtype

    # the live sampler is the authority on step count, not our config default
    sampler = net.inference_sampler
    n_timesteps = int(getattr(sampler, "num_timesteps", a.timesteps or 100))
    n_sched = n_timesteps - 1          # measured rule (results/28); the per-arm assert protects it
    print(f"[sched] sampler num_timesteps={n_timesteps} -> schedule length {n_sched} "
          f"(measured: the module is called num_timesteps-1 times)")

    p = resolve_prompt(cfg, dev)
    prompt = p[None].expand(a.num_designs, -1, -1).to(device=dev, dtype=adtype).contiguous()
    pid = Path(a.prompt_pdb).stem
    print(f"[sched] prompt {pid}: N={p.shape[0]}, L={a.length}, K={a.num_designs}, "
          f"target mean λ={a.target_lambda}\n")

    # The adherence reference is the prompt's OWN structure: "how much of the prompt fold did this
    # design reproduce". Loaded once; TM-score is CPU-only so scoring adds seconds, not minutes.
    prompt_struct = None if a.no_score else _as_struct(str(a.prompt_pdb))

    names = a.shapes.split(",") if a.shapes else shape_names(support=support, teeth=a.teeth)
    # Control arms are flat λ pushed through the SAME hook. They are NOT matched-mean, on purpose, so
    # they are gated separately below and never enter the shape comparison.
    controls = {f"ctrl_lam{float(v):g}": float(v)
                for v in (a.hook_const.split(",") if a.hook_const.strip() else [])}
    arms, rows = ["baseline"] + names + list(controls), []

    for arm in arms:
        t0 = time.time()
        hook = original = None
        is_control = arm in controls
        if arm == "baseline":
            adapter.clear_prompt()                      # λ=0, vanilla RFD3, the unsteered reference
            values = []
        else:
            if is_control:
                values = [controls[arm]] * n_sched      # flat, unnormalized: a control, not a comparison arm
            else:
                values = build(arm, n_sched, a.target_lambda, support=support, teeth=a.teeth)
            adapter.set_prompt(prompt)
            hook = ScheduledLambda(net.diffusion_module, adapter, values)
            original = hook.install(net)

        try:
            _seed_all(a.seed)                           # paired noise across arms
            with torch.no_grad():
                # _run_once unwraps engine.run's {example_id: [RFD3Output, ...]} mapping;
                # calling engine.run directly hands back the raw dict, not design objects.
                outs = _run_once(engine)
        finally:
            if hook is not None:
                ScheduledLambda.uninstall(net, original)

        realized = hook.realized_mean() if hook else 0.0
        calls = hook.calls if hook else 0
        ok = (arm == "baseline") or is_control or abs(realized - a.target_lambda) <= a.tol
        adir = out_dir / arm; adir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, o in enumerate(outs):
            p = adir / f"{pid}_{arm}_{i}.pdb"
            write_pdb(o.atom_array, p)
            paths.append(p)
        if hook is not None:
            (adir / "lambda_trace.json").write_text(json.dumps(hook.applied, indent=2))

        # --- Phase-A scoring: adherence (TM to the prompt) + within-arm diversity -------------------
        designs = []
        if prompt_struct is not None:
            for i, o in enumerate(outs):
                adh = adherence(o.atom_array, prompt_struct, tm_norm="prompt")
                designs.append({"idx": i, "pdb": str(paths[i]),
                                "tm_prompt": adh.tm_norm_prompt, "tm_design": adh.tm_norm_design,
                                "n_design": adh.n_design, "n_prompt": adh.n_prompt})
            # lower diversity_tm = the arm's designs are LESS alike = MORE diverse (score.py contract)
            div = pairwise_tm_diversity([o.atom_array for o in outs])
            tms = [d["tm_prompt"] for d in designs]
            tm_mean = sum(tms) / len(tms) if tms else None
            tm_med = sorted(tms)[len(tms) // 2] if tms else None
        else:
            div = tm_mean = tm_med = None

        rows.append({"arm": arm, "is_control": is_control, "n_designs": len(outs), "calls": calls,
                     "schedule_len": len(values), "target_lambda": a.target_lambda,
                     "realized_mean_lambda": realized, "matched_mean_ok": ok,
                     "lambda_min": min(values) if values else 0.0,
                     "lambda_max": max(values) if values else 0.0,
                     "tm_prompt_mean": tm_mean, "tm_prompt_median": tm_med,
                     "tm_prompt_min": min(tms) if designs else None,
                     "tm_prompt_max": max(tms) if designs else None,
                     "diversity_tm": div, "designs": designs,
                     "seconds": round(time.time() - t0, 1), "out_dir": str(adir)})
        flag = "OK " if ok else "⛔ MEAN MISMATCH"
        _f = lambda v: "  n/a" if v is None else f"{v:.3f}"
        print(f"[sched] {arm:<13} {flag} calls={calls:>3}/{len(values) or '-':<3} "
              f"realized mean λ={realized:.6f}  λ∈[{rows[-1]['lambda_min']:.3f},"
              f"{rows[-1]['lambda_max']:.3f}]  TM={_f(tm_mean)}  div={_f(div)}  "
              f"{rows[-1]['seconds']:>5.1f}s")
        if not ok:
            print(f"[sched]   ⛔ realized mean deviates from target by "
                  f"{abs(realized - a.target_lambda):.2e} > tol {a.tol:.0e}. This arm is NOT comparable "
                  f"to the others; the sampler consumed {calls} of {len(values)} entries. "
                  f"Fix the schedule length before using this run.")

    bad = [r["arm"] for r in rows if not r["matched_mean_ok"]]
    print(f"\n[sched] {len(rows)} arm(s) -> {out_dir}")
    print("[sched] matched-mean check: " + ("ALL OK" if not bad else f"⛔ FAILED for {bad}"))
    if bad:
        print("[sched] ⛔ DO NOT COMPARE ARMS until this is fixed: a differing realized mean means the "
              "sweep is measuring total conditioning strength, not its distribution.")

    # --- per-fold summary: dTM against the λ=0 baseline, and against constant λ (today's behaviour) ---
    if prompt_struct is not None:
        base = next((r for r in rows if r["arm"] == "baseline"), None)
        const = next((r for r in rows if r["arm"] == "constant"), None)
        b_tm = base["tm_prompt_mean"] if base else None
        c_tm = const["tm_prompt_mean"] if const else None
        print(f"\n[sched] {pid}: adherence (TM normalized by the PROMPT) and within-arm diversity")
        print(f"[sched] {'arm':<14}{'meanTM':>8}{'dTM_base':>10}{'dTM_const':>11}{'div_TM':>8}{'ok':>4}")
        for r in rows:
            d_b = r["tm_prompt_mean"] - b_tm if (b_tm is not None and r["tm_prompt_mean"] is not None) else None
            d_c = r["tm_prompt_mean"] - c_tm if (c_tm is not None and r["tm_prompt_mean"] is not None) else None
            r["dtm_vs_baseline"], r["dtm_vs_constant"] = d_b, d_c
            g = lambda v: "   n/a" if v is None else f"{v:+.3f}"
            h = lambda v: "  n/a" if v is None else f"{v:.3f}"
            print(f"[sched] {r['arm']:<14}{h(r['tm_prompt_mean']):>8}{g(d_b):>10}{g(d_c):>11}"
                  f"{h(r['diversity_tm']):>8}{('ok' if r['matched_mean_ok'] else 'BAD'):>4}")
        print("[sched] read: higher meanTM = more adherent; LOWER div_TM = more diverse designs. "
              "Report both, always (plan/57 §8).")
        # The plumbing control: λ driven to 0 at every step, through the hook, with the prompt SET.
        # It must land on the baseline. If it lands on `constant` instead, set_scale is not reaching the
        # forward pass and every arm above is really the same arm.
        z = next((r for r in rows if r["arm"] == "ctrl_lam0"), None)
        if z is not None and b_tm is not None and c_tm is not None:
            d_b, d_c = abs(z["tm_prompt_mean"] - b_tm), abs(z["tm_prompt_mean"] - c_tm)
            verdict = ("✅ hook TRANSMITS λ (zero-through-hook tracks the λ=0 baseline)" if d_b < d_c
                       else "⛔ HOOK MAY NOT BE TRANSMITTING λ: zero-through-hook is closer to constant "
                            "λ=1 than to the λ=0 baseline. Do not interpret the shape arms.")
            print(f"[sched] plumbing control ctrl_lam0: TM={z['tm_prompt_mean']:.3f} vs baseline "
                  f"{b_tm:.3f} (|Δ|={d_b:.3f}) vs constant {c_tm:.3f} (|Δ|={d_c:.3f}) -> {verdict}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({
            "arms": rows,
            "config": {"prompt_pdb": str(a.prompt_pdb), "ckpt": a.ckpt, "length": a.length,
                       "K": a.num_designs, "seed": a.seed, "target_lambda": a.target_lambda,
                       "num_timesteps": n_timesteps, "schedule_len": n_sched,
                       "support": list(support), "teeth": a.teeth, "tm_norm": "prompt",
                       "scored": prompt_struct is not None},
        }, indent=2))
        print(f"[sched] wrote {a.json}")


if __name__ == "__main__":
    main()
