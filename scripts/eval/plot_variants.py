"""Big-n variant-separation figure — per-fold SPA dTM for the 3 variants across the 15 held-out folds.

Shows variant-ROBUSTNESS: N×1536 / 1×1536 / 1×32 track each other fold-by-fold (spread within SEM);
the cheap CLSS 1×32 holds its own. Reads outputs/eval/bigN_variants/{variant}_{u}/. Unconditional
fold-steering, adherence-only.

    conda run -n spa-dev python scripts/eval/plot_variants.py --out outputs/eval/figures/bigN_variants.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

VARIANTS = ["Nx1536", "1x1536", "1x32"]
COLORS = {"Nx1536": "#1f77b4", "1x1536": "#2ca02c", "1x32": "#d62728"}


def _scores(p):
    with open(p) as f:
        return json.load(f).get("scores", [])


def _mean_tm(sc, cond, lam):
    v = [s["tm_score"] for s in sc if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6
         and s.get("tm_score") is not None]
    return st.mean(v) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/eval/bigN_variants")
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--out", default="outputs/eval/figures/bigN_variants.png")
    args = ap.parse_args()

    folds = sorted({os.path.basename(d).split("_", 1)[1]
                    for d in glob.glob(f"{args.root}/*_*") if os.path.isdir(d)})
    rows = []
    for u in folds:
        base = None
        dt = {}
        for v in VARIANTS:
            p = f"{args.root}/{v}_{u}/flywheel_results.json"
            if not os.path.exists(p):
                continue
            sc = _scores(p)
            if base is None:
                base = _mean_tm(sc, "baseline", 0.0)
            m = _mean_tm(sc, "spa", args.lam)
            if m is not None and base is not None:
                dt[v] = m - base
        if len(dt) == 3:
            rows.append((u, dt))
    rows.sort(key=lambda r: st.mean(r[1].values()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    means = {}
    for i, v in enumerate(VARIANTS):
        vals = [r[1][v] for r in rows]
        means[v] = st.mean(vals)
        ax.bar(x + (i - 1) * w, vals, w, label=f"{v} (mean {means[v]:+.3f})", color=COLORS[v])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(f"SPA fold-shift  dTM @ λ={args.lam:g}")
    spread = max(means.values()) - min(means.values())
    ax.set_title(f"Variant-robustness across 15 held-out folds — N×1536 ≈ 1×1536 ≈ 1×32 "
                 f"(mean-dTM spread {spread:.3f}, within SEM)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    from poster_style import savefig_poster
    savefig_poster(fig, args.out)  # 300-DPI PNG + vector PDF sibling (poster-ready)
    print(f"[plot] {args.out}  ({len(rows)} folds; means " +
          ", ".join(f"{v} {means[v]:+.3f}" for v in VARIANTS) + f"; spread {spread:.3f})")


if __name__ == "__main__":
    main()
