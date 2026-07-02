"""Poster panel — specificity controls: the fold-steering *requires* a prompt (nullprompt) AND *follows
the specific prompt* (wrong-prompt → the design goes to the decoy), across fold classes.

Two panels (aggregates of record from the control + breadth runs; docs/results/02 §4). All at λ=2:
  A — A0A522W419 (α/β), K=8 + OF3: baseline vs spa vs nullprompt vs wrong-prompt, adherence TM→target.
  B — wrong-prompt breadth (3 fold classes): TM→target vs TM→decoy — the design abandons the scored
      target and adopts the decoy fold *where SPA has steering power* (α/β, β); inconclusive on the
      low-headroom all-α (SPA barely steers there — nothing to misdirect).

    conda run -n spa-dev python scripts/eval/plot_specificity_poster.py \
        --out outputs/eval/figures/specificity_poster.png
"""

from __future__ import annotations

import argparse

# Panel A — A0A522W419 controls, λ=2, adherence TM→target (docs/results/02 §4).
A_LABELS = ["baseline", "spa", "nullprompt", "wrong-prompt"]
A_TM = [0.326, 0.644, 0.341, 0.203]
A_COLOR = ["#7f7f7f", "#1f77b4", "#9467bd", "#d62728"]

# Panel B — wrong-prompt breadth, λ=2: (target, class, TM→target, TM→decoy).
B_ROWS = [
    ("A0A6J8EPQ1", "α/β", 0.23, 0.62),
    ("A0A7S3EB45", "all-β", 0.23, 0.40),
    ("A0A7C9GW19", "all-α", 0.31, 0.30),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval/figures/specificity_poster.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from poster_style import savefig_poster

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})

    # ---- Panel A: needs a prompt ----
    xa = np.arange(len(A_LABELS))
    axA.bar(xa, A_TM, color=A_COLOR, width=0.66, zorder=2)
    axA.axhline(A_TM[0], color="#7f7f7f", ls="--", lw=1, zorder=1)
    for i, t in enumerate(A_TM):
        lab = f"{t:.2f}" if i == 0 else f"{t:.2f}\nΔ{t - A_TM[0]:+.2f}"
        axA.annotate(lab, (i, t), ha="center", va="bottom", fontsize=10, color="#222")
    axA.set_xticks(xa); axA.set_xticklabels(A_LABELS, fontsize=10)
    axA.set_ylabel("adherence  TM → target"); axA.set_ylim(0, 0.80)
    axA.set_title("SPA needs a prompt\n(A0A522W419 · α/β · λ=2)", fontsize=12)
    axA.grid(axis="y", color="#eee", zorder=0); axA.set_axisbelow(True)

    # ---- Panel B: follows the given prompt ----
    xb = np.arange(len(B_ROWS)); w = 0.36
    tgt = [r[2] for r in B_ROWS]; dec = [r[3] for r in B_ROWS]
    axB.bar(xb - w / 2, tgt, w, label="TM → scored target", color="#7f7f7f", zorder=2)
    axB.bar(xb + w / 2, dec, w, label="TM → decoy (the prompt given)", color="#e67e22", zorder=2)
    for i in range(len(B_ROWS)):
        axB.annotate(f"{tgt[i]:.2f}", (i - w / 2, tgt[i]), ha="center", va="bottom", fontsize=9)
        axB.annotate(f"{dec[i]:.2f}", (i + w / 2, dec[i]), ha="center", va="bottom", fontsize=9)
    axB.annotate("inconclusive\n(SPA near-inert)", (2, 0.34), ha="center", va="bottom",
                 fontsize=8.5, color="#777", style="italic")
    axB.set_xticks(xb); axB.set_xticklabels([f"{r[0]}\n{r[1]}" for r in B_ROWS], fontsize=9)
    axB.set_ylabel("adherence  TM  (λ=2)"); axB.set_ylim(0, 0.80)
    axB.set_title("…and follows the *given* prompt\n(wrong-prompt → the design goes to the decoy)", fontsize=12)
    axB.legend(fontsize=9, loc="upper left"); axB.grid(axis="y", color="#eee", zorder=0); axB.set_axisbelow(True)

    fig.suptitle("Specificity — the fold-steering requires, and follows, the actual structural prompt",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig_poster(fig, args.out)


if __name__ == "__main__":
    main()
