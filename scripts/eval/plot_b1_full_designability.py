"""Poster panel for the §4 B1-full hard⊕soft designability result (cloud H100, K=8/N=8, 25 prompts).

The claim this panel earns: a native **hard motif** stays satisfied through the whole
ProteinMPNN→OF3 pipeline (refold motif-RMSD ≤ ~1 Å on every fold class) while the **soft SPA**
fold-prompt composes at a *bounded, fold-structured* designability cost — essentially free with
RFD3's grain (all-α +0.02) and largest against it (all-β -0.15).

Data are the aggregates of record transcribed in
    docs/results/02_eval_results.md  §4  ("Big-n designability, B1-full", job 9109859564504743936)
(the per-design JSONs live in gs://genomancer-spa-cache/eval/b1_full/results/ and are gitignored
locally; the markdown table is the authoritative aggregate — mirrored here, not recomputed).

Two panels, shared fold-class rows:
  left  — designability dumbbell: baseline d_succ ● → SPA d_succ ● (best-of-K scRMSD < 2 Å).
  right — hard motif survives: refold-side motif-RMSD per group, all under the 1.0 Å line.

    conda run -n spa-dev python scripts/eval/plot_b1_full_designability.py \
        --out outputs/eval/figures/b1_full_designability.png
"""

from __future__ import annotations

import argparse
import os

# key, label, n, baseline d_succ, SPA d_succ, refold motif-RMSD (Å)  — docs/results/02 §4
GROUPS = [
    ("overall", "overall  (n=25)", 25, 0.695, 0.625, 0.69),
    ("a", "all-α  (n=7)", 7, 0.714, 0.732, 0.62),
    ("ab", "α/β  (n=10)", 10, 0.775, 0.675, 0.63),
    ("b", "all-β  (n=6)", 6, 0.479, 0.333, 0.96),
    ("irr", "irregular  (n=2)", 2, 0.875, 0.875, 0.36),
]
# Fold-class palette matches scripts/eval/plot_bigN_h5.py (the poster "brand"); overall = ink.
COLOR = {"overall": "#333333", "a": "#d62728", "ab": "#9467bd", "b": "#1f77b4", "irr": "#7f7f7f"}
# Secondary read (length bands), for the caption line rather than the plot.
BANDS = "length bands: ≤256 res Δ−0.091 (motif 0.74 Å)   ·   256–384 res Δ−0.038 (motif 0.60 Å)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval/figures/b1_full_designability.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ys = list(range(len(GROUPS)))[::-1]  # first group (overall) on top
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})

    # ---- left: designability dumbbell (baseline -> SPA) ----
    for (k, _lab, _n, base, spa, _m), y in zip(GROUPS, ys):
        c = COLOR[k]
        axL.plot([base, spa], [y, y], color=c, lw=2.5, alpha=0.45, zorder=1, solid_capstyle="round")
        axL.scatter([base], [y], facecolors="white", edgecolors="#8a8a8a", linewidths=1.6, s=90, zorder=2)
        axL.scatter([spa], [y], facecolors=c, edgecolors="white", linewidths=1.4, s=110, zorder=3)
        d = spa - base
        # Δ label anchored just right of the rightmost dot (in ink) so it never overlaps the marks.
        axL.annotate(f"Δ{d:+.2f}", (max(base, spa) + 0.02, y), va="center", ha="left",
                     fontsize=9, color="#222")
    axL.set_yticks(ys)
    axL.set_yticklabels([g[1] for g in GROUPS], fontsize=10)
    axL.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    axL.set_xlim(0.25, 1.0)
    axL.set_xlabel("designable fraction  d_succ  (best-of-K scRMSD < 2 Å)", fontsize=10)
    axL.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)
    axL.set_axisbelow(True)
    axL.set_title("Soft SPA cost is bounded & fold-structured", fontsize=11)
    axL.legend(handles=[
        Line2D([0], [0], marker="o", lw=0, mfc="white", mec="#8a8a8a", mew=1.6, ms=9, label="baseline (motif-only)"),
        Line2D([0], [0], marker="o", lw=0, mfc="#555", mec="white", mew=1.2, ms=10, label="hard ⊕ soft SPA"),
    ], fontsize=9, loc="lower left", framealpha=0.9)

    # ---- right: hard motif survives the refold ----
    for (k, _lab, _n, _b, _s, m), y in zip(GROUPS, ys):
        axR.barh(y, m, color=COLOR[k], height=0.55, zorder=2)
        axR.annotate(f"{m:.2f} Å", (m, y), xytext=(0.3, 0), textcoords="offset fontsize",
                     va="center", ha="left", fontsize=9, color="#222")
    axR.axvline(1.0, color="#c0392b", ls="--", lw=1.2, zorder=3, label="1.0 Å — motif satisfied")
    axR.set_yticks(ys)
    axR.set_yticklabels([])
    axR.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    axR.set_xlim(0, 1.35)
    axR.set_xlabel("refold-side motif-RMSD (Å)", fontsize=10)
    axR.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)
    axR.set_axisbelow(True)
    axR.set_title("Hard motif survives the pipeline", fontsize=11)
    axR.legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.suptitle("Hard ⊕ soft designability across 25 held-out folds (K=8, cloud H100) — "
                 "hard always satisfied, soft composes at a fold-structured cost",
                 fontsize=12.5, y=0.99)
    fig.text(0.5, 0.005, BANDS, ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
