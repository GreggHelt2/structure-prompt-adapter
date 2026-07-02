"""Poster panel 1 — the SPA mechanism schematic (no data; a hand-drawn architecture diagram).

The claim this panel earns (docs/results/03_poster_narrative.md §8 panel 1):
    "A parameter-efficient, non-destructive sidecar — identity at rest, tunable by λ."

Architecture facts drawn (code-confirmed in docs/plan/02_attachment_points.md + 03_spa_architecture.md):
  - SPA wraps each of RFD3's 18 token-track blocks (LocalAttentionPairBias); RFD3 + ESM3 are FROZEN.
  - Decoupled cross-attention: query from the token hidden Z (768-d), K/V from the ESM3 structural
    fold-prompt ([N×1536] → c_model), softmax over the N prompt tokens.
  - Zero-init output projection ⇒ at λ=0 the added term is exactly 0 ⇒ bit-identical to vanilla RFD3.
  - λ is an inference-time strength knob. ~24M trainable params (the sidecar), everything else frozen.

    conda run -n spa-dev python scripts/eval/plot_mechanism_schematic.py \
        --out outputs/eval/figures/mechanism_schematic.png
"""

from __future__ import annotations

import argparse
import os

FROZEN_FC, FROZEN_EC = "#e9eef1", "#5f7d8c"   # frozen components (RFD3, ESM3)
SPA_FC, SPA_EC = "#ffe6cc", "#e67e22"          # trainable SPA sidecar (the hero element)
INK, MUTED = "#222222", "#5b6b73"
ARROW = "#455a64"


def rbox(ax, x, y, w, h, text, fc, ec, tc=INK, fs=10, lw=1.6, bold=False, z=3):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=z + 1, fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color=ARROW, lw=2.2, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=1, shrinkB=1, mutation_scale=15))


def node(ax, xy, sym="+", ec=ARROW, r=0.24):
    from matplotlib.patches import Circle
    ax.add_patch(Circle(xy, r, fc="white", ec=ec, lw=2, zorder=5))
    ax.text(*xy, sym, ha="center", va="center", fontsize=15, color=ec, zorder=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval/figures/mechanism_schematic.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.14)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6.5, 6.6, "SPA — a decoupled cross-attention sidecar on frozen RFdiffusion3",
            ha="center", va="center", fontsize=15, fontweight="bold", color=INK)

    # ---- representative RFD3 block, with a 2-rect "×18" stack hint behind it ----
    for dx, dy in ((0.30, 0.30), (0.15, 0.15)):
        ax.add_patch(FancyBboxPatch((3.2 + dx, 1.15 + dy), 7.3, 3.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc="#f2f5f7", ec="#c2ced4", lw=1.2, zorder=1))
    ax.add_patch(FancyBboxPatch((3.2, 1.15), 7.3, 3.5, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc="#fbfdfe", ec=FROZEN_EC, lw=1.8, zorder=2))
    ax.text(6.85, 4.32, "1 of 18 token-track blocks  ·  FROZEN", ha="center", va="center",
            fontsize=11, style="italic", color=MUTED, zorder=4)

    # ---- residual backbone (horizontal): Z_i -> ⊕ -> ⊕ -> Z_{i+1} ----
    yres = 3.5
    arrow(ax, (3.55, yres), (4.76, yres))       # Z_i -> ⊕1
    arrow(ax, (5.24, yres), (7.36, yres))       # ⊕1 -> ⊕2
    arrow(ax, (7.84, yres), (9.95, yres))       # ⊕2 -> Z_{i+1}
    node(ax, (5.0, yres), "+")
    node(ax, (7.6, yres), "+")
    ax.text(3.5, yres + 0.32, r"$Z_i$  (768-d)", ha="left", va="bottom", fontsize=10, color=INK)
    ax.text(9.95, yres + 0.32, r"$Z_{i+1}$", ha="right", va="bottom", fontsize=10, color=INK)

    # ---- native sublayer (adds at ⊕1) ----
    rbox(ax, 4.15, 1.55, 1.7, 0.85, "Self-attention\n+ pair bias", FROZEN_FC, FROZEN_EC, fs=10)
    arrow(ax, (5.0, 2.40), (5.0, yres - 0.24))

    # ---- SPA sidecar (adds at ⊕2 through the ×λ gate) ----
    rbox(ax, 6.5, 1.4, 2.2, 0.95, "SPA\ndecoupled cross-attn", SPA_FC, SPA_EC, fs=10.5, bold=True)
    node(ax, (7.85, 2.85), r"$\times\lambda$", ec=SPA_EC, r=0.2)
    arrow(ax, (7.85, 2.35), (7.85, 2.65), color=SPA_EC)      # SPA -> gate
    arrow(ax, (7.85, 3.05), (7.6, yres - 0.24), color=SPA_EC)  # gate -> ⊕2
    # query taps the frozen track (left); K/V come from the prompt (right) — the "decoupled" part.
    arrow(ax, (6.95, yres - 0.28), (6.95, 2.36), color=SPA_EC, lw=1.8)
    ax.text(6.72, 2.72, "Q", ha="right", va="center", fontsize=9, color=SPA_EC)

    # ---- prompt pathway (right column) feeding K/V into SPA ----
    rbox(ax, 10.75, 4.35, 2.2, 0.9, "structural\nfold-prompt", "#eef3f6", FROZEN_EC, fs=10)
    arrow(ax, (11.85, 4.35), (11.85, 4.0))
    rbox(ax, 10.75, 3.05, 2.2, 0.9, "ESM3 encoder\nFROZEN", FROZEN_FC, FROZEN_EC, fs=10)
    arrow(ax, (11.85, 3.05), (11.85, 2.7))
    rbox(ax, 10.75, 1.7, 2.2, 0.95, r"$[N \times 1536]$  per-residue", "#eef3f6", FROZEN_EC, fs=9.5)
    ax.text(11.85, 1.55, "(1×1536 pooled · 1×32 CLSS variants)", ha="center", va="top",
            fontsize=7.8, color=MUTED)
    arrow(ax, (10.72, 2.0), (8.72, 1.9), color=SPA_EC)     # prompt -> SPA (K,V)
    ax.text(9.7, 2.28, "K, V\n(1536→c_model)", ha="center", va="center", fontsize=8.3, color=SPA_EC)

    # ---- legend + captions ----
    ax.legend(handles=[
        Patch(fc=FROZEN_FC, ec=FROZEN_EC, label="frozen  (RFdiffusion3, ESM3)"),
        Patch(fc=SPA_FC, ec=SPA_EC, label="trainable  (SPA sidecar, ~24M params)"),
    ], loc="lower left", fontsize=9.5, framealpha=0.95, bbox_to_anchor=(0.015, 0.02))

    fig.text(0.5, 0.075,
             r"Zero-init output ⇒ at $\lambda=0$ the added term is exactly 0 ⇒ "
             r"bit-identical to vanilla RFdiffusion3;   $\lambda$ tunes conditioning strength at inference.",
             ha="center", fontsize=10.5, color=INK)
    fig.text(0.5, 0.032, "A parameter-efficient, non-destructive sidecar — identity at rest, tunable by λ.",
             ha="center", fontsize=10.5, style="italic", color=MUTED)

    from poster_style import savefig_poster
    savefig_poster(fig, args.out)  # 300-DPI PNG + vector PDF sibling (poster-ready)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
