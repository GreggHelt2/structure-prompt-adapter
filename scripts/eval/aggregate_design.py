"""Aggregate the big-n H5 DESIGNABILITY subset (outputs/eval/bigN_h5_design/runD_<u>/).

Per prompt: motif-baseline vs Run A·λ2 — designable/K, mean best-of-K scrmsd, and refold-side
motif-RMSD (does the *designed sequence* still realize the pinned motif geometry?). Across prompts:
mean designable rate (baseline vs SPA → does SPA preserve foldability?) and refold motif satisfaction.

    conda run -n spa-dev python scripts/eval/aggregate_design.py --root outputs/eval/bigN_h5_design
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


def _agg(scores, cond, lam):
    g = [s for s in scores if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6]
    if not g:
        return None
    des = [1 if s.get("designable") else 0 for s in g if s.get("designable") is not None]
    scr = [s["scrmsd"] for s in g if s.get("scrmsd") is not None]
    mrr = [s["motif_rmsd_refold"] for s in g if s.get("motif_rmsd_refold") is not None]
    return {"k": len(g), "des": sum(des), "ndes": len(des),
            "scrmsd": st.mean(scr) if scr else None,
            "mrr": st.mean(mrr) if mrr else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/eval/bigN_h5_design")
    args = ap.parse_args()

    rows, base_rate, spa_rate, spa_mrr = [], [], [], []
    for d in sorted(glob.glob(f"{args.root}/runD_*")):
        u = os.path.basename(d)[len("runD_"):]
        sc = _scores(f"{d}/flywheel_results.json")
        base, spa = _agg(sc, "baseline", 0.0), _agg(sc, "spa", 2.0)
        if not base or not spa:
            continue
        rows.append((u, base, spa))
        if base["ndes"]:
            base_rate.append(base["des"] / base["ndes"])
        if spa["ndes"]:
            spa_rate.append(spa["des"] / spa["ndes"])
        if spa["mrr"] is not None:
            spa_mrr.append(spa["mrr"])

    print(f"=== big-n H5 designability ({len(rows)} prompts) — motif⊕SPA, Run A·λ2 vs motif-only ===")
    print(f"{'prompt':<12}{'base des':>9}{'spa des':>9}{'base scR':>9}{'spa scR':>9}{'refold mRMSD':>13}")
    for u, b, s in rows:
        bd, sd = f"{b['des']}/{b['ndes']}", f"{s['des']}/{s['ndes']}"
        bscr = f"{b['scrmsd']:.2f}" if b["scrmsd"] is not None else "--"
        sscr = f"{s['scrmsd']:.2f}" if s["scrmsd"] is not None else "--"
        mrr = f"{s['mrr']:.3f}" if s["mrr"] is not None else "--"
        print(f"{u:<12}{bd:>9}{sd:>9}{bscr:>9}{sscr:>9}{mrr:>13}")

    print("\n=== across-prompt summary ===")
    if base_rate and spa_rate:
        print(f"  mean designable rate: baseline {st.mean(base_rate):.2f}  vs  SPA(Run A·λ2) {st.mean(spa_rate):.2f}"
              f"  (d_succ {st.mean(spa_rate) - st.mean(base_rate):+.2f})")
        print(f"  prompts where SPA designable rate >= baseline: "
              f"{sum(1 for b, s in zip(base_rate, spa_rate) if s >= b)}/{len(spa_rate)}")
    if spa_mrr:
        print(f"  refold-side motif-RMSD (SPA): mean {st.mean(spa_mrr):.3f} Å (max {max(spa_mrr):.3f}) "
              f"— does the designed sequence still realize the motif?")


if __name__ == "__main__":
    main()
