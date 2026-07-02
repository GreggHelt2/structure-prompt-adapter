"""Poster panel 3b — a montage of the hard⊕soft structural money-shots (baseline vs SPA on the target).

Composites the two rendered PyMOL cartoons into one labeled figure. Each input cartoon is already a
2-panel image — baseline (red) | SPA (blue), both superposed on the gray target fold — so this stacks
the α/β and all-β examples with per-row captions. Run after the cartoons have been rendered.

    conda run -n spa-dev python scripts/eval/plot_moneyshot_montage.py \
        --out outputs/eval/figures/moneyshot_montage.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_FIGDIR = "outputs/eval/figures"
# (cartoon filename, row caption) — the two locked hard⊕soft money-shots.
ROWS = [
    ("cartoon_hardsoft_A0A522W419_helixstrand.png", "α/β · A0A522W419 · helixstrand motif · TM 0.38 → 0.74"),
    ("cartoon_hardsoft_A0A7S3EB45_beta.png", "all-β · A0A7S3EB45 · 2-strand motif · TM 0.27 → 0.80"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=DEFAULT_FIGDIR)
    ap.add_argument("--out", default=f"{DEFAULT_FIGDIR}/moneyshot_montage.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    from poster_style import savefig_poster

    missing = [f for f, _ in ROWS if not (Path(args.figdir) / f).exists()]
    if missing:
        raise SystemExit(f"missing cartoons in {args.figdir}: {missing} (render them first)")

    fig, axes = plt.subplots(len(ROWS), 1, figsize=(11, 9.2))
    for ax, (fname, caption) in zip(axes, ROWS):
        ax.imshow(mpimg.imread(str(Path(args.figdir) / fname)))
        ax.set_title(caption, fontsize=13.5, fontweight="bold", color="#222")
        ax.axis("off")

    fig.text(0.5, 0.945,
             "baseline (no SPA) —— with SPA —— superposed on the target fold (gray);  the hard motif stays pinned",
             ha="center", fontsize=10.5, color="#555")
    fig.suptitle("Hard ⊕ soft money-shots: SPA recovers the target fold",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    savefig_poster(fig, args.out)


if __name__ == "__main__":
    main()
