"""Aggregate the big-n variant-separation run (outputs/eval/bigN_variants/{variant}_{u}/).

Per fold: baseline prompt-TM + each variant's SPA dTM at λ{1,2}. Across folds: mean dTM per variant —
do the cheap pooled variants (1×1536, 1×32) keep up with the per-residue N×1536 on the against-the-grain
β folds (where §5's single α/β prompt may not have separated them)? Unconditional fold-steering,
adherence-only. Tolerant of partial data (dirs still being written while the run is in flight).

    conda run -n spa-dev python scripts/eval/aggregate_variants.py --root outputs/eval/bigN_variants
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

VARIANTS = ["Nx1536", "1x1536", "1x32"]


def _scores(p):
    with open(p) as f:
        return json.load(f).get("scores", [])


def _mean_tm(sc, cond, lam):
    v = [s["tm_score"] for s in sc if s["condition"] == cond and abs(s["lambda_scale"] - lam) < 1e-6
         and s.get("tm_score") is not None]
    return st.mean(v) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/eval/bigN_variants")
    ap.add_argument("--lam", type=float, default=2.0, help="λ to show in the per-fold table")
    args = ap.parse_args()

    folds = sorted({os.path.basename(d).split("_", 1)[1]
                    for d in glob.glob(f"{args.root}/*_*") if os.path.isdir(d)})
    rows = []
    agg = {(v, lam): [] for v in VARIANTS for lam in (1.0, 2.0)}
    for u in folds:
        base = None
        for v in VARIANTS:
            p = f"{args.root}/{v}_{u}/flywheel_results.json"
            if os.path.exists(p):
                b = _mean_tm(_scores(p), "baseline", 0.0)
                if b is not None:
                    base = b
                    break
        row = {"u": u, "base": base}
        for v in VARIANTS:
            p = f"{args.root}/{v}_{u}/flywheel_results.json"
            if not (os.path.exists(p) and base is not None):
                continue
            sc = _scores(p)
            for lam in (1.0, 2.0):
                m = _mean_tm(sc, "spa", lam)
                if m is not None:
                    row[f"{v}_{lam}"] = (m, m - base)
                    agg[(v, lam)].append(m - base)
        rows.append(row)

    lam = args.lam
    print(f"=== big-n variant separation ({len(rows)} folds) — SPA dTM @ λ={lam:g} (TM vs baseline) ===")
    print(f"{'fold':<12}{'base':>7}{'Nx1536':>13}{'1x1536':>13}{'1x32':>13}")
    for r in rows:
        cells = []
        for v in VARIANTS:
            c = r.get(f"{v}_{lam}")
            cells.append(f"{c[0]:.2f}({c[1]:+.2f})" if c else "     --      ")
        print(f"{r['u']:<12}{(r['base'] or 0):>7.2f}{cells[0]:>13}{cells[1]:>13}{cells[2]:>13}")

    print("\n=== across-fold mean dTM per variant ===")
    for l in (1.0, 2.0):
        parts = []
        for v in VARIANTS:
            ds = agg[(v, l)]
            parts.append(f"{v} {st.mean(ds):+.3f} (n={len(ds)})" if ds else f"{v} --")
        print(f"  λ={l:g}: " + "   ".join(parts))
    a2 = {v: st.mean(agg[(v, 2.0)]) for v in VARIANTS if agg[(v, 2.0)]}
    if len(a2) == 3:
        spread = max(a2.values()) - min(a2.values())
        print(f"\n  spread @λ2 (max−min mean dTM): {spread:.3f}  -> "
              f"{'NEAR-EQUIVALENT (variant-robust)' if spread < 0.04 else 'SEPARATION (variants differ)'}")


if __name__ == "__main__":
    main()
