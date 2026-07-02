"""Specificity-controls poster panel (dev 16 §11): baseline vs spa vs nullprompt vs wrong-prompt.

Reads the two control-run flywheel_results.json — Run A (baseline / spa / nullprompt on the target
prompt) and Run B (spa on the DECOY prompt = wrong-prompt, scored vs target). Shows that SPA raises
adherence ONLY with the real prompt: nullprompt and wrong-prompt collapse to ~baseline, while
designability is preserved across all conditions.

    conda run -n spa-dev python scripts/eval/plot_controls.py \
        --run-a outputs/eval/control_A_nullprompt_A0A522W419 \
        --run-b outputs/eval/control_B_wrongprompt_A0A522W419_decoy_A0A2X2KHU0 \
        --lam 2 --out outputs/eval/figures/specificity_controls.png
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

COLOR = {"baseline": "#7f7f7f", "spa": "#1f77b4", "nullprompt": "#9467bd", "wrong": "#d62728"}


def _scores(run_dir):
    return json.load(open(Path(run_dir) / "flywheel_results.json")).get("scores", [])


def _sel(scores, cond, lam):
    return [s for s in scores if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6]


def _mean_tm(rows):
    v = [r.get("tm_norm_prompt", r.get("tm_score")) for r in rows]
    v = [x for x in v if x is not None]
    return st.mean(v) if v else math.nan


def _dsucc(rows):
    v = [r.get("designable") for r in rows if r.get("designable") is not None]
    return (sum(1 for x in v if x) / len(v)) if v else math.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--out", default="outputs/eval/figures/specificity_controls.png")
    args = ap.parse_args()

    A, B = _scores(args.run_a), _scores(args.run_b)
    lg = f"λ={args.lam:g}"
    # (label, rows, colorkey)
    bars = [
        ("baseline", _sel(A, "baseline", 0.0), "baseline"),
        (f"spa\n{lg}", _sel(A, "spa", args.lam), "spa"),
        (f"nullprompt\n{lg}", _sel(A, "nullprompt", args.lam), "nullprompt"),
        (f"wrong-prompt\n{lg}", _sel(B, "spa", args.lam), "wrong"),
    ]
    labels = [b[0] for b in bars]
    colors = [COLOR[b[2]] for b in bars]
    tms = [_mean_tm(b[1]) for b in bars]
    ds = [_dsucc(b[1]) for b in bars]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = range(len(bars))

    base_tm = tms[0]
    ax1.bar(x, tms, color=colors, width=0.66, zorder=2)
    ax1.axhline(base_tm, color="#7f7f7f", ls="--", lw=1, zorder=1)
    for i, t in enumerate(tms):
        lab = f"{t:.2f}" if i == 0 else f"{t:.2f}\nΔ{t - base_tm:+.2f}"
        ax1.annotate(lab, (i, t), ha="center", va="bottom", fontsize=9, color="#222")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("adherence  TM → target")
    ax1.set_ylim(0, (max([t for t in tms if not math.isnan(t)] or [1]) * 1.28))
    ax1.set_title("SPA steers only with the real prompt")
    ax1.grid(axis="y", color="#eee", zorder=0); ax1.set_axisbelow(True)

    ax2.bar(x, ds, color=colors, width=0.66, zorder=2)
    for i, d in enumerate(ds):
        ax2.annotate(f"{d:.2f}", (i, d), ha="center", va="bottom", fontsize=9, color="#222")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("designable fraction  d_succ")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Designability")
    ax2.grid(axis="y", color="#eee", zorder=0); ax2.set_axisbelow(True)

    fig.suptitle("Specificity controls — A0A522W419 (soft-only, K=8): SPA needs the *right* prompt",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}  (TM: " + ", ".join(f"{l.splitlines()[0]}={t:.2f}"
          for l, t in zip(labels, tms)) + ")")


if __name__ == "__main__":
    main()
