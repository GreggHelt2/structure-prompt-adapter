"""Poster panel for the §5 variant-designability result (cloud H100, soft-only, 15 folds, K=4/N=4).

The claim this panel earns: the three SPA variants — per-residue **N×1536**, global **1×1536**,
CLSS-compressed **1×32** — are **near-equivalent on both axes**. Each preserves designability
(Δd_succ ≥ 0) while steering the fold (ΔTM +0.13…+0.15); the cheapest 1×32 holds its own (and even
edges ahead on designability). Fold-steering is variant-robust → the abstract's ≥2-variants + CLSS
encoder commitments, on the designability axis.

Data are the aggregates of record transcribed in
    docs/results/02_eval_results.md  §5  ("Big-n variant designability", job 5530144373582331904)
(per-design JSONs in gs://genomancer-spa-cache/eval/variant_desig/ are gitignored locally; the
markdown table is the authoritative aggregate — mirrored here, not recomputed).

Two dumbbell panels (matching plot_b1_full_designability.py), one row per variant:
  left  — designability d_succ: baseline ○ → SPA ● (best-of-K scRMSD < 2 Å).
  right — adherence TM (prompt-normalized): baseline ○ → SPA ●.

    conda run -n spa-dev python scripts/eval/plot_variant_designability.py \
        --out outputs/eval/figures/variant_designability.png
"""

from __future__ import annotations

import argparse
import os

# key, label, baseline d_succ, SPA d_succ, baseline TM, SPA TM  — docs/results/02 §5
VARIANTS = [
    ("Nx1536", "N×1536\n(per-residue)", 0.800, 0.867, 0.311, 0.441),
    ("1x1536", "1×1536\n(mean-pool)", 0.850, 0.850, 0.311, 0.458),
    ("1x32", "1×32\n(CLSS)", 0.817, 0.883, 0.312, 0.441),
]
# Variant palette matches scripts/eval/plot_variants.py (the poster "brand").
COLOR = {"Nx1536": "#1f77b4", "1x1536": "#2ca02c", "1x32": "#d62728"}
SPREAD_DSUCC = 0.033  # cross-variant spread, SPA d_succ (§5)
SPREAD_TM = 0.017     # cross-variant spread, SPA adherence-TM (§5)


def _dumbbell(ax, idx_base, idx_spa, xlim, xlabel, title):
    ys = list(range(len(VARIANTS)))[::-1]  # first variant on top
    for v, y in zip(VARIANTS, ys):
        c = COLOR[v[0]]
        base, spa = v[idx_base], v[idx_spa]
        ax.plot([base, spa], [y, y], color=c, lw=2.5, alpha=0.45, zorder=1, solid_capstyle="round")
        ax.scatter([base], [y], facecolors="white", edgecolors="#8a8a8a", linewidths=1.6, s=90, zorder=2)
        ax.scatter([spa], [y], facecolors=c, edgecolors="white", linewidths=1.4, s=110, zorder=3)
        d = spa - base
        # Δ label anchored just right of the rightmost dot so it never overlaps the marks.
        gap = 0.03 * (xlim[1] - xlim[0])
        ax.annotate(f"Δ{d:+.3f}", (max(base, spa) + gap, y), va="center", ha="left",
                    fontsize=9, color="#222")
    ax.set_yticks(ys)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11)
    return ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval/figures/variant_designability.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 3.9))

    ys = _dumbbell(axL, 2, 3, (0.5, 1.0),
                   "designable fraction  d_succ  (best-of-K scRMSD < 2 Å)",
                   f"Designability preserved  (spread {SPREAD_DSUCC:.3f})")
    axL.set_yticklabels([v[1] for v in VARIANTS], fontsize=9)
    axL.legend(handles=[
        Line2D([0], [0], marker="o", lw=0, mfc="white", mec="#8a8a8a", mew=1.6, ms=9, label="baseline"),
        Line2D([0], [0], marker="o", lw=0, mfc="#555", mec="white", mew=1.2, ms=10, label="SPA"),
    ], fontsize=9, loc="lower left", framealpha=0.9)

    _dumbbell(axR, 4, 5, (0.2, 0.6),
              "adherence  TM  (prompt-normalized)",
              f"Fold-steering equivalent  (spread {SPREAD_TM:.3f})")
    axR.set_yticklabels([])

    fig.suptitle("Three SPA variants — near-equivalent on both axes; the cheap 1×32 holds its own "
                 "(soft-only, 15 held-out folds, K=4)", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
