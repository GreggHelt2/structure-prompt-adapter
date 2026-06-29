"""Checkpoint-step adherence ladder — does more training help? (dev ``13_longer_training_decision.md``).

Plots **prompt-adherence vs. training step** by evaluating a ladder of *existing* SPA snapshots on a
held-out prompt — **no new training, no OF3**. The shape of the curve is the decision (dev ``13`` §2):
flat after ~3–10k ⇒ a longer same-recipe run won't help; still rising at 30k ⇒ we stopped on the
upslope (extend).

**Equivalent to running the flywheel driver once per checkpoint, by construction** (dev ``13``, the
"same results?" discussion): generation is the *same* :func:`spa.eval.generate.generate` path, which
re-seeds the global RNG to ``eval.seed`` immediately before every rollout — so each design depends
only on ``(checkpoint, prompt, condition, λ, seed)``, never on process history. This module just
reuses that path efficiently — **build the RFD3 engine once, attach the adapter once, compute the
ESM3 prompt once (then feed it via the cache so the loop never reloads ESM3), generate the
checkpoint-independent baseline once, and loop the checkpoints swapping only the adapter weights**
(``load_spa``, ~1 s each). Results match the driver distributionally (RFD3's GPU sampler is
kernel-nondeterministic, so neither is bit-reproducible — dev ``08`` §6); the ladder reads a trend
across checkpoints, robust to per-rollout jitter.

Anchor check (dev ``13`` §"de-risk"): the step-30k point should reproduce the driver's known result
(``A0A7S1B8G4`` λ=1 prompt-TM ≈ 0.373, dev ``08`` §6) within sampler noise — printed for comparison.

All knobs are config/CLI (``eval.ladder`` = list of snapshot paths; reuses ``eval.lambda_scale`` /
``eval.num_designs`` / ``eval.length`` / ``eval.prompt_pdb`` / ``eval.seed`` / ``eval.out_dir``) —
nothing hardware- or cost-specific is hardcoded (dev root ``CLAUDE.md`` portability rule).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev


@dataclass
class LadderPoint:
    """Adherence summary for one ``(step, condition, λ)`` group over K designs (dev ``13`` §2)."""

    step: int
    label: str            # snapshot label (e.g. "step10000" / "final"); "baseline" for the reference
    condition: str        # "baseline" | "spa"
    lambda_scale: float
    n: int
    tm_mean: float | None
    tm_sd: float | None
    tm_max: float | None
    prmsd_mean: float | None
    d_tm: float | None    # tm_mean - baseline tm_mean (None for the baseline row)


def _parse_step(path: str) -> tuple[int, str]:
    """``(numeric_step, label)`` from a snapshot filename: ``...step1000.pt`` -> (1000, "step1000");
    ``..._final.pt`` -> (30000, "final") [the adapter export == the last step]; else (0, stem)."""
    stem = Path(str(path)).stem
    m = re.search(r"step(\d+)", stem)
    if m:
        return int(m.group(1)), f"step{m.group(1)}"
    if "final" in stem:
        return 30000, "final"
    return 0, stem


def _dist(values):
    clean = [float(v) for v in values if v is not None and float(v) == float(v)]
    if not clean:
        return None, None, None
    return (float(mean(clean)),
            float(pstdev(clean)) if len(clean) > 1 else 0.0,
            float(max(clean)))


def run_ladder(cfg) -> dict:
    """Run the checkpoint-step adherence ladder for one prompt; return + persist the per-point table.

    Reuses :func:`spa.eval.generate.generate` for sampling (so it is the same experiment as the driver,
    dev ``13``). Returns ``{points, results_path, anchor}``.
    """
    import torch

    from ..model import attach_spa
    from ..train.harness import frozen_rfd3_net, load_spa
    from ..utils.device import resolve_device
    from .generate import (
        _resolve_out_dir,
        build_eval_engine,
        generate,
        resolve_prompt,
    )
    from .score import adherence

    ev = cfg.eval
    ladder = list(ev.get("ladder") or [])
    if not ladder:
        raise ValueError("ladder sweep needs eval.ladder=[<ckpt1>,<ckpt2>,...] (snapshot paths).")
    lambdas = [float(x) for x in (list(ev.get("lambda_scale", [0.5, 1.0, 2.0]))
                                  if not isinstance(ev.get("lambda_scale"), (int, float))
                                  else [float(ev.lambda_scale)])]
    prompt_struct = ev.get("prompt_pdb")
    if not prompt_struct:
        raise ValueError("ladder sweep needs eval.prompt_pdb (the held-out prompt structure).")
    tm_norm = str((ev.get("score") or {}).get("tm_norm", "prompt"))

    device = resolve_device(cfg.hardware.device)
    base_out = _resolve_out_dir(ev.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # --- one-time setup: engine + adapter + prompt (ESM3 once -> cache, so the loop never reloads it) ---
    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    ev.ckpt = None                                    # attach zero-init; we swap weights per snapshot
    from .generate import load_adapter
    adapter = load_adapter(net, cfg, device)
    adapter.eval()

    p = resolve_prompt(cfg, device)                   # loads ESM3 once (from prompt_pdb), then frees it
    cache_path = base_out / "prompt_cache.pt"
    torch.save(p.detach().cpu(), cache_path)
    ev.prompt_cache = str(cache_path)
    ev.prompt_pdb = None                              # subsequent generate() reads the cache (no ESM3)

    def _adh_group(designs):
        adhs = [adherence(d, prompt_struct, tm_norm=tm_norm) for d in designs]
        tm_mean, tm_sd, tm_max = _dist([a.tm_score for a in adhs])
        prmsd_mean, _, _ = _dist([a.prompt_rmsd for a in adhs])
        return tm_mean, tm_sd, tm_max, prmsd_mean

    # --- baseline once (vanilla RFD3, checkpoint-independent) ---
    ev.out_dir = str(base_out / "baseline")
    ev.conditions = ["baseline"]
    ev.lambda_scale = [0.0]
    base_designs = generate(cfg, engine=engine, adapter=adapter)
    b_tm, b_sd, b_max, b_prmsd = _adh_group(base_designs)
    points = [LadderPoint(0, "baseline", "baseline", 0.0, len(base_designs),
                          b_tm, b_sd, b_max, b_prmsd, None)]
    print(f"[ladder] baseline: n={len(base_designs)} TM={b_tm:.3f} pRMSD={b_prmsd:.2f}")

    # --- sweep the checkpoints (swap adapter weights only) ---
    for ckpt in ladder:
        step, label = _parse_step(ckpt)
        load_spa(adapter, ckpt)
        ev.out_dir = str(base_out / label)
        ev.conditions = ["spa"]
        ev.lambda_scale = lambdas
        spa_designs = generate(cfg, engine=engine, adapter=adapter)
        for lam in lambdas:
            grp = [d for d in spa_designs if float(d.lambda_scale) == float(lam)]
            tm_mean, tm_sd, tm_max, prmsd = _adh_group(grp)
            d_tm = (tm_mean - b_tm) if (tm_mean is not None and b_tm is not None) else None
            points.append(LadderPoint(step, label, "spa", float(lam), len(grp),
                                      tm_mean, tm_sd, tm_max, prmsd, d_tm))
            print(f"[ladder] {label} λ={lam:g}: n={len(grp)} TM={tm_mean:.3f} "
                  f"dTM={d_tm:+.3f} pRMSD={prmsd:.2f}")

    results_path = base_out / "ladder_results.json"
    with open(results_path, "w") as fh:
        json.dump({"prompt": str(prompt_struct), "tm_norm": tm_norm,
                   "points": [asdict(pt) for pt in points]}, fh, indent=2, default=str)

    anchor = _print_table(points, prompt_struct)
    print(f"[ladder] wrote results -> {results_path}")
    return {"points": points, "results_path": results_path, "anchor": anchor}


def _print_table(points, prompt_struct) -> dict:
    """Print the adherence-vs-step table; return the step-30k λ=1 anchor (vs driver's ≈0.373)."""
    print(f"\n=== adherence-vs-training-step ladder (dev 13 §2) — prompt {Path(str(prompt_struct)).stem} ===")
    print(f"{'step':>8}{'cond':>10}{'lambda':>8}{'n':>4}{'TM':>9}{'(sd)':>8}{'TMmax':>8}{'dTM':>9}{'pRMSD':>8}")
    print("-" * 74)
    anchor = {}
    for pt in points:
        d_tm = "—" if pt.d_tm is None else f"{pt.d_tm:+.3f}"
        sd = "—" if pt.tm_sd is None else f"{pt.tm_sd:.3f}"
        tmx = "—" if pt.tm_max is None else f"{pt.tm_max:.3f}"
        print(f"{pt.step:>8}{pt.condition:>10}{pt.lambda_scale:>8.2f}{pt.n:>4}"
              f"{(pt.tm_mean or 0):>9.3f}{sd:>8}{tmx:>8}{d_tm:>9}{(pt.prmsd_mean or 0):>8.2f}")
        if pt.label == "final" and abs(pt.lambda_scale - 1.0) < 1e-9:
            anchor = {"final_lambda1_tm": pt.tm_mean}
    if anchor:
        # The step-30k λ=1 value, for an equivalence check vs a same-prompt driver/known ref. (The ref
        # is PER PROMPT — e.g. A0A7S1B8G4 ≈ 0.373, dev 08 §6 — so we print the value, not a hardcoded
        # number that would be meaningless for other prompts.)
        print(f"\n[anchor] step-30k λ=1 TM={anchor['final_lambda1_tm']:.3f} "
              f"(compare to a same-prompt driver/known ref for the equivalence check)")
    return anchor
