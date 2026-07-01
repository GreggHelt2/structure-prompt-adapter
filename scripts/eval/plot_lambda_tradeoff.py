"""Big-n H5 λ-tradeoff panel — aggregate adherence + designability vs λ across the 15 held-out folds.

Shows the sweet spot: **adherence plateaus by λ=1** while **designability holds at λ=1 and drops at λ=2**.
Adherence from outputs/eval/bigN_h5/runA_<u> (mean prompt-TM per λ, averaged over prompts); designability
from bigN_h5_design_l1 (λ=1) + bigN_h5_design (λ=2) (mean designable/K, averaged over prompts).

    conda run -n spa-dev python scripts/eval/plot_lambda_tradeoff.py \
        --out outputs/eval/figures/bigN_h5_lambda_tradeoff.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def _scores(p):
    with open(p) as f:
        return json.load(f).get("scores", [])


def _mean_tm(sc, cond, lam):
    v = [s["tm_score"] for s in sc if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6
         and s.get("tm_score") is not None]
    return st.mean(v) if v else None


def _des_rate(sc, cond, lam):
    g = [s for s in sc if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6
         and s.get("designable") is not None]
    return sum(1 for s in g if s["designable"]) / len(g) if g else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adh", default="outputs/eval/bigN_h5")
    ap.add_argument("--des-l1", default="outputs/eval/bigN_h5_design_l1")
    ap.add_argument("--des-l2", default="outputs/eval/bigN_h5_design")
    ap.add_argument("--out", default="outputs/eval/figures/bigN_h5_lambda_tradeoff.png")
    args = ap.parse_args()

    # adherence: mean over prompts of the per-prompt mean prompt-TM, at baseline / λ1 / λ2
    adh = {0.0: [], 1.0: [], 2.0: []}
    for d in glob.glob(f"{args.adh}/runA_*"):
        sc = _scores(f"{d}/flywheel_results.json")
        for lam, cond in ((0.0, "baseline"), (1.0, "spa"), (2.0, "spa")):
            m = _mean_tm(sc, cond, lam)
            if m is not None:
                adh[lam].append(m)
    adh_m = {k: st.mean(v) for k, v in adh.items() if v}

    # designability: mean designable rate over prompts, baseline (both runs) / λ1 / λ2
    des = {0.0: [], 1.0: [], 2.0: []}
    for lam, root in ((1.0, args.des_l1), (2.0, args.des_l2)):
        for d in glob.glob(f"{root}/runD_*"):
            sc = _scores(f"{d}/flywheel_results.json")
            b, s = _des_rate(sc, "baseline", 0.0), _des_rate(sc, "spa", lam)
            if b is not None:
                des[0.0].append(b)
            if s is not None:
                des[lam].append(s)
    des_m = {k: st.mean(v) for k, v in des.items() if v}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, xl = [0, 1, 2], ["baseline", "λ=1", "λ=2"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(6.2, 7), sharex=True)

    ax[0].plot(x, [adh_m[l] for l in (0.0, 1.0, 2.0)], "o-", color="#1f77b4", lw=2.2)
    ax[0].set_ylabel("mean prompt-TM (adherence)")
    ax[0].set_title("Big-n H5 λ-tradeoff — 15 folds, motif ⊕ SPA (Run A)")
    ax[0].annotate("adherence plateaus by λ=1", (1, adh_m[1.0]), textcoords="offset points",
                   xytext=(8, -14), fontsize=9)

    ax[1].plot(x, [des_m[l] for l in (0.0, 1.0, 2.0)], "s--", color="#d62728", lw=2.2)
    ax[1].set_ylabel("mean designable rate (scRMSD < 2 Å)")
    ax[1].set_ylim(0, 1)
    ax[1].annotate("designability holds at λ=1,\nfalls at λ=2", (2, des_m[2.0]),
                   textcoords="offset points", xytext=(-95, 6), fontsize=9)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(xl)
    ax[1].set_xlabel("SPA scale λ")

    for a in ax:
        a.axvline(1, color="green", ls=":", alpha=0.6)
    ax[0].text(1.02, ax[0].get_ylim()[0] + 0.01, "sweet spot", color="green", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot] {args.out}")
    print(f"  adherence (mean TM): baseline {adh_m[0.0]:.2f}  λ1 {adh_m[1.0]:.2f}  λ2 {adh_m[2.0]:.2f}")
    print(f"  designable rate:     baseline {des_m[0.0]:.2f}  λ1 {des_m[1.0]:.2f}  λ2 {des_m[2.0]:.2f}")


if __name__ == "__main__":
    main()
