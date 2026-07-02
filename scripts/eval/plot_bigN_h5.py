"""Poster figure for the big-n H5 hard⊕soft result (reads outputs/eval/bigN_h5/run{A,B}_<u>/).

Three panels (the characterized headline):
  1. SOFT — per-prompt Run A·λ2 dTM, sorted, colored by fold class (composition strongest on β).
  2. HARD — per-prompt motif-RMSD (design-side), flat near 0 + the 100%-satisfied annotation.
  3. HEADROOM — base TM vs dTM scatter (the fold-structured failure mode: high base -> no room to add).

Designability (when the bigN_h5_design run lands) can be overlaid later; this is the adherence headline.

    conda run -n spa-dev python scripts/eval/plot_bigN_h5.py --root outputs/eval/bigN_h5 \
        --out outputs/eval/figures/bigN_h5_headline.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

FOLD = {  # from configs/eval/manifest_curated15.yaml
    "A0A1X7NTP0": "a", "A0A6A0D1E8": "b", "A0A3P5VTL4": "b", "A0A090ME36": "b", "A0A7S1B8G4": "b",
    "A0A820JRM2": "irr", "A0A6P1U7L6": "b", "A0A7S3EB45": "b", "A0A522W419": "ab", "A0A2X2KHU0": "a",
    "R7VVY2": "irr", "A0A6J8EPQ1": "ab", "A0A7C9GW19": "a", "H1SDK8": "ab", "A0A1X0IID6": "ab",
}
COLOR = {"a": "#d62728", "b": "#1f77b4", "ab": "#9467bd", "irr": "#7f7f7f"}
LABEL = {"a": "all-α", "b": "all-β", "ab": "α/β", "irr": "irregular"}


def _scores(path):
    with open(path) as fh:
        return json.load(fh).get("scores", [])


def _mean(scores, cond, lam, key="tm_score"):
    v = [s[key] for s in scores if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6
         and s.get(key) is not None]
    return st.mean(v) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/eval/bigN_h5")
    ap.add_argument("--out", default="outputs/eval/figures/bigN_h5_headline.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = []
    motif_all = []
    for da in sorted(glob.glob(f"{args.root}/runA_*")):
        u = os.path.basename(da)[len("runA_"):]
        a = _scores(f"{da}/flywheel_results.json")
        base = _mean(a, "baseline", 0.0)
        a_l2 = _mean(a, "spa", 2.0)
        mrmsd = _mean(a, "spa", 2.0, key="motif_rmsd") or _mean(a, "baseline", 0.0, key="motif_rmsd")
        if base is None or a_l2 is None:
            continue
        bpath = f"{args.root}/runB_{u}/flywheel_results.json"
        b = _scores(bpath) if os.path.exists(bpath) else []
        motif_all += [s["motif_rmsd"] for s in (a + b) if s.get("motif_rmsd") is not None]
        rows.append({"u": u, "fold": FOLD.get(u, "irr"), "base": base, "dtm": a_l2 - base, "mrmsd": mrmsd})

    rows.sort(key=lambda r: r["dtm"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 1) SOFT — sorted dTM by fold
    ys = range(len(rows))
    ax[0].barh(list(ys), [r["dtm"] for r in rows], color=[COLOR[r["fold"]] for r in rows])
    ax[0].set_yticks(list(ys)); ax[0].set_yticklabels([r["u"] for r in rows], fontsize=7)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("soft fold-shift  dTM (Run A·λ2 vs motif-only)")
    ax[0].set_title("SOFT — composes on most folds (strongest on β)")
    ax[0].legend(handles=[Patch(color=COLOR[k], label=LABEL[k]) for k in ("b", "ab", "a", "irr")],
                 fontsize=8, loc="lower right")

    # 2) HARD — motif-RMSD flat near 0
    ax[1].scatter([r["mrmsd"] for r in rows], list(ys), color=[COLOR[r["fold"]] for r in rows], s=30)
    ax[1].set_yticks(list(ys)); ax[1].set_yticklabels([r["u"] for r in rows], fontsize=7)
    ax[1].axvline(1.0, color="r", ls="--", lw=0.8, label="1.0 Å cutoff")
    ax[1].set_xlim(0, 1.1)
    ax[1].set_xlabel("motif-RMSD (Å), design-side")
    sat = sum(1 for m in motif_all if m < 1.0)
    ax[1].set_title(f"HARD — satisfied {sat}/{len(motif_all)} (100%), mean {st.mean(motif_all):.3f} Å")
    ax[1].legend(fontsize=8, loc="lower right")

    # 3) HEADROOM — base TM vs dTM
    for r in rows:
        ax[2].scatter(r["base"], r["dtm"], color=COLOR[r["fold"]], s=40)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("baseline TM (motif-only RFD3)")
    ax[2].set_ylabel("soft fold-shift  dTM")
    ax[2].set_title("HEADROOM — high base ⇒ less to add")

    fig.suptitle("Big-n H5: hard motif ⊕ soft SPA across 15 held-out folds (adherence)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    from poster_style import savefig_poster
    savefig_poster(fig, args.out)  # 300-DPI PNG + vector PDF sibling (poster-ready)
    print(f"[plot] wrote {args.out}  ({len(rows)} prompts)")


if __name__ == "__main__":
    main()
