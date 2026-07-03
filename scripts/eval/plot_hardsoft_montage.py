"""Composite native-vs-SPA hard⊕soft pair renders into a poster montage (300-dpi PNG + vector PDF).

Consumes the per-(id, condition) PNGs from render_hardsoft_pairs.py plus the same pick JSON (for TM /
motif-RMSD labels), one row per fold (native | + SPA), ordered by ΔTM. native=red / +SPA=blue / motif=green.

    conda run -n spa-dev python scripts/eval/plot_hardsoft_montage.py \
        --pick <pick.json> --renders-dir <dir> --out outputs/eval/figures/<name> \
        --title "Hard ⊕ soft ..." --subtitle "..."
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NAT_C, SPA_C = "#b23a2e", "#1f6fd0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", required=True)
    ap.add_argument("--renders-dir", required=True)
    ap.add_argument("--out", required=True, help="output basename (writes .png and .pdf)")
    ap.add_argument("--title", default="Hard ⊕ soft: native RFdiffusion3 vs SPA")
    ap.add_argument("--subtitle", default="pinned motif in green (held in both); native scaffold drifts, SPA recovers the target fold (grey)")
    a = ap.parse_args()
    pick = json.load(open(a.pick))

    rows = sorted(pick, key=lambda p: pick[p]["spa"]["tm"] - pick[p]["native"]["tm"], reverse=True)
    nr = len(rows)
    fig, axes = plt.subplots(nr, 2, figsize=(8.2, 3.55 * nr), squeeze=False)
    for i, pid in enumerate(rows):
        c = pick[pid]
        fn = (c.get("fold_name") or pid).replace("_", " ")
        label = f"{fn}\n{pid}" + (f" · motif {c['motif_rmsd']:.2f} Å" if c.get("motif_rmsd") is not None else "")
        for j, (tag, who, col) in enumerate([("native", "native RFdiffusion3", NAT_C), ("spa", "+ SPA", SPA_C)]):
            ax = axes[i][j]
            ax.imshow(mpimg.imread(os.path.join(a.renders_dir, f"{pid}_{tag}.png"))); ax.axis("off")
            ax.set_title(f"{who}    TM {c[tag]['tm']:.2f}", fontsize=11, color=col, fontweight="bold", pad=3)
        axes[i][0].text(-0.03, 0.5, label, transform=axes[i][0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10.5, fontweight="bold", color="#1a2a33")
    fig.suptitle(a.title, fontsize=15.5, fontweight="bold", y=0.995)
    fig.text(0.5, 1.0 - 0.4 / (3.55 * nr), a.subtitle, ha="center", fontsize=9.3, color="#3a5563")
    plt.tight_layout(rect=[0.02, 0, 1, 1.0 - 0.9 / (3.55 * nr)])
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", a.out + ".png/.pdf")


if __name__ == "__main__":
    main()
