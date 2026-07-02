"""Poster panel 5 — the in-silico validation pipeline schematic (no data; a hand-drawn flow diagram).

Panel spec (docs/results/03_poster_narrative.md §8 panel 5):
    RFD3 ± SPA → ProteinMPNN → OpenFold3 → score (TM / scRMSD / motif-RMSD).

Pipeline facts drawn (docs/plan/05_validation_pipeline.md; all legs verified locally on the A5000):
  - Generation: RFdiffusion3 ± SPA → backbone (mmCIF, converted to PDB for the next leg).
  - Inverse folding: ProteinMPNN → N sequences per design (FASTA).
  - Structure prediction: OpenFold3 (MSA-free) → refolded structure (CIF).
  - Scoring: scRMSD<2 Å = designability (refold vs design), motif-RMSD = hard-motif adherence,
    TM (prompt-normalized) = soft fold adherence. Baseline = wrapped-no-prompt ≡ vanilla RFD3.

    conda run -n spa-dev python scripts/eval/plot_validation_schematic.py \
        --out outputs/eval/figures/validation_schematic.png
"""

from __future__ import annotations

import argparse
import os

INK, MUTED, ARROW = "#222222", "#5b6b73", "#455a64"
# categorical stage palette (reuses the poster "brand" hues; SPA=orange ties to panel 1).
STAGES = [
    ("RFdiffusion3\n± SPA", "design generation", "#ffe6cc", "#e67e22"),
    ("ProteinMPNN", "inverse folding", "#dbe9f6", "#1f77b4"),
    ("OpenFold3", "structure prediction", "#dcefdc", "#2ca02c"),
    ("Score", "designability + adherence", "#e9e1f3", "#9467bd"),
]
# arrow I/O labels between consecutive stages
IO = ["backbone\n(mmCIF→PDB)", "sequences\n(FASTA, ×N)", "refold\n(CIF)"]


def rbox(ax, x, y, w, h, fc, ec, lw=1.8, z=3):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, p0, p1, color=ARROW, lw=2.4, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=2, shrinkB=2, mutation_scale=17))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval/figures/validation_schematic.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7.0, 5.62, "In-silico validation flywheel — RFdiffusion3 ± SPA → ProteinMPNN → "
            "OpenFold3 → score", ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)

    # ---- the four stage boxes ----
    xs = [0.5, 3.9, 7.3, 10.7]
    ws = [2.6, 2.4, 2.4, 3.0]
    yb, hb = 2.9, 1.15
    cy = yb + hb / 2
    centers = []
    for (title, sub, fc, ec), x, w in zip(STAGES, xs, ws):
        rbox(ax, x, yb, w, hb, fc, ec)
        ax.text(x + w / 2, yb + hb * 0.63, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=ec)
        ax.text(x + w / 2, yb + hb * 0.24, sub, ha="center", va="center",
                fontsize=8.8, style="italic", color=MUTED)
        centers.append((x + w / 2, x, x + w))

    # ---- inter-stage arrows + I/O format labels ----
    for i, io in enumerate(IO):
        x_from = centers[i][2]
        x_to = centers[i + 1][1]
        arrow(ax, (x_from + 0.05, cy), (x_to - 0.05, cy))
        ax.text((x_from + x_to) / 2, 2.5, io, ha="center", va="center",
                fontsize=8.6, color=MUTED)

    # ---- two conditions feeding stage 1 ----
    rbox(ax, 0.35, 4.9, 3.9, 0.5, "#ffe6cc", "#e67e22", lw=1.4, z=3)
    ax.text(2.3, 5.15, "SPA:  + ESM3 fold-prompt  ⊕  hard native motif", ha="center", va="center",
            fontsize=9, color="#b3560f")
    rbox(ax, 0.35, 4.28, 3.9, 0.5, "#eef1f3", "#5f7d8c", lw=1.4, z=3)
    ax.text(2.3, 4.53, "baseline:  wrapped, no prompt  ≡  vanilla RFD3", ha="center", va="center",
            fontsize=9, color=MUTED)
    arrow(ax, (1.8, 4.26), (1.8, 4.07), lw=1.8)

    # ---- metrics panel under the Score box ----
    mx, my, mw, mh = 10.7, 1.15, 3.0, 1.5  # aligned under the Score box (xs[3], ws[3])
    rbox(ax, mx, my, mw, mh, "#f4f0fa", "#9467bd", lw=1.4, z=2)
    ax.text(mx + mw / 2, my + mh - 0.24, "metrics", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#6f4a9c")
    lines = [
        ("scRMSD < 2 Å", "designability (refold vs design)"),
        ("motif-RMSD", "hard-motif adherence"),
        ("TM (prompt-norm)", "soft fold adherence"),
    ]
    for j, (m, desc) in enumerate(lines):
        yy = my + mh - 0.6 - j * 0.36
        ax.text(mx + 0.2, yy, m, ha="left", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(mx + 0.2, yy - 0.15, desc, ha="left", va="center", fontsize=7.6, color=MUTED)
    arrow(ax, (centers[3][0], yb - 0.02), (centers[3][0], my + mh + 0.02), lw=1.8)

    fig.text(0.5, 0.055,
             "Self-consistency: does the ProteinMPNN→OpenFold3 refold reproduce the SPA design (scRMSD), "
             "and match the fold-prompt (TM) + pinned motif (motif-RMSD)?",
             ha="center", fontsize=10, color=INK)
    fig.text(0.5, 0.018, "Baseline (wrapped, no prompt) is bit-identical to vanilla RFdiffusion3 — a clean "
             "same-sampler A/B.", ha="center", fontsize=10, style="italic", color=MUTED)

    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
