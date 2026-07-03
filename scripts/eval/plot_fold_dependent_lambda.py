"""Fold-dependent λ×designability figure (300-dpi PNG + vector PDF) — the panel-2 "mechanism" money figure.

Twin-axis per fold class: adherence (TM→fold) and designable rate (scRMSD<2 Å) vs λ, for the two clean
sweeps in docs/results/02 §2 (all-β A0A7S1B8G4 "against the grain"; α/β A0A522W419 "with the grain").
Data is embedded (the sweeps are fixed results); regenerates a stable figure independent of data churn.

    conda run -n spa-dev python scripts/eval/plot_fold_dependent_lambda.py \
        --out outputs/eval/figures/fold_dependent_lambda_300
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAM = [0, 0.5, 1, 2]
# (title, tm-by-λ, designable-rate-by-λ, sweet-λ, italic note, annotation offset+align)
DATA = [
    ("all-β · A0A7S1B8G4  —  against the grain",
     [0.239, 0.328, 0.366, 0.350], [1.00, 0.625, 0.75, 0.625], 1,
     "interior optimum:\nTM turns over, cost bounded", (10, -30), "left"),
    ("α/β · A0A522W419  —  with the grain",
     [0.323, 0.534, 0.621, 0.660], [0.875, 0.875, 0.875, 1.00], 2,
     "no tradeoff:\nboth axes rise, monotone", (-12, 20), "right"),
]
TM_C, D_C = "#0e6b78", "#b07818"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval/figures/fold_dependent_lambda_300",
                    help="output basename (writes .png and .pdf)")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35))
    x = list(range(len(LAM)))
    handles = None
    for ax, (title, tm, d, sweet, note, off, ha) in zip(axes, DATA):
        ax2 = ax.twinx()
        l1, = ax.plot(x, tm, "-o", color=TM_C, lw=2.6, ms=8, label="adherence (TM → fold)", zorder=3)
        l2, = ax2.plot(x, d, "--s", color=D_C, lw=2.3, ms=7, label="designable rate", zorder=3)
        si = LAM.index(sweet)
        ax.axvline(si, color="#9aa7ad", ls=":", lw=1.4, zorder=1)
        ax.scatter([si], [tm[si]], s=210, facecolors="none", edgecolors=TM_C, lw=2.2, zorder=4)
        ax.annotate(f"λ={sweet:g} sweet spot", (si, tm[si]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=9.5, fontweight="bold", color="#33454e", zorder=5)
        ax.text(0.03, 0.06, note, transform=ax.transAxes, fontsize=8.6, color="#5a6b73", style="italic")
        ax.set_xticks(x); ax.set_xticklabels([f"λ={l:g}" for l in LAM], fontsize=10)
        ax.set_ylim(0.15, 0.72); ax2.set_ylim(0.50, 1.06)
        ax.set_ylabel("TM adherence", color=TM_C, fontsize=10.5, fontweight="bold")
        ax2.set_ylabel("designable rate (scRMSD<2 Å)", color=D_C, fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=TM_C); ax2.tick_params(axis="y", labelcolor=D_C)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        ax.grid(alpha=0.22, zorder=0); ax.set_axisbelow(True)
        handles = [l1, l2]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Fold-dependent λ control — the designability cost is structured by RFdiffusion3's prior",
                 fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
