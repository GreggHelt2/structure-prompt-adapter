"""Aggregate the big-n H5 adherence-only matrix into a per-prompt table + across-prompt summary.

Reads <root>/runA_<uniprot>/flywheel_results.json (baseline + Run A spa λ{1,2}) and
<root>/runB_<uniprot>/flywheel_results.json (Run B spa λ{1,2}). For each prompt: baseline adherence
(TM-to-prompt), Run A / Run B dTM per λ, and motif satisfaction. Across prompts: mean dTM per
(checkpoint, λ), motif-satisfied rate, mean motif-RMSD (the hard side), and the emergent zero-shot
contrast (Run A vs Run B). Adherence-only -> designability columns are absent (separate subset).

    conda run -n spa-dev python scripts/eval/aggregate_h5.py --root outputs/eval/bigN_h5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def _load(path: str) -> list:
    with open(path) as fh:
        return json.load(fh).get("scores", [])


def _mean_tm(scores, condition: str, lam: float):
    vals = [s["tm_score"] for s in scores
            if s["condition"] == condition and abs(s["lambda_scale"] - lam) < 1e-6
            and s.get("tm_score") is not None]
    if not vals:
        return None, None, 0
    sem = st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    return st.mean(vals), sem, len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/eval/bigN_h5")
    args = ap.parse_args()

    rows, agg = [], {("A", 1.0): [], ("A", 2.0): [], ("B", 1.0): [], ("B", 2.0): []}
    motif_rmsd, motif_sat = [], []

    for da in sorted(glob.glob(f"{args.root}/runA_*")):
        u = os.path.basename(da)[len("runA_"):]
        a = _load(f"{da}/flywheel_results.json")
        bpath = f"{args.root}/runB_{u}/flywheel_results.json"
        b = _load(bpath) if os.path.exists(bpath) else []
        base, _, _ = _mean_tm(a, "baseline", 0.0)
        for s in a + b:
            if s.get("motif_rmsd") is not None:
                motif_rmsd.append(s["motif_rmsd"])
            if s.get("motif_satisfied") is not None:
                motif_sat.append(1 if s["motif_satisfied"] else 0)
        row = {"u": u, "base": base}
        for ck, sc in (("A", a), ("B", b)):
            for lam in (1.0, 2.0):
                m, _sem, _n = _mean_tm(sc, "spa", lam)
                if m is not None and base is not None:
                    row[f"{ck}{int(lam)}"] = (m, m - base)
                    agg[(ck, lam)].append(m - base)
        rows.append(row)

    def fmt(cell):
        return f"{cell[0]:.2f}({cell[1]:+.2f})" if cell else "   --    "

    print(f"\n=== big-n H5 (adherence-only) — per prompt: TM(dTM vs baseline) ===  [{len(rows)} prompts]")
    print(f"{'prompt':<12}{'base':>6}{'A·λ1':>11}{'A·λ2':>11}{'B·λ1':>11}{'B·λ2':>11}")
    for r in rows:
        print(f"{r['u']:<12}{(r['base'] or 0):>6.2f}"
              f"{fmt(r.get('A1')):>11}{fmt(r.get('A2')):>11}{fmt(r.get('B1')):>11}{fmt(r.get('B2')):>11}")

    print("\n=== across-prompt summary ===")
    for ck in ("A", "B"):
        for lam in (1.0, 2.0):
            ds = agg[(ck, lam)]
            if ds:
                pos = sum(1 for d in ds if d > 0)
                print(f"  Run {ck} · λ={lam:.0f}: mean dTM {st.mean(ds):+.3f}  "
                      f"(median {st.median(ds):+.3f}, {pos}/{len(ds)} prompts steer up)")
    if motif_rmsd:
        print(f"\n  HARD: motif-RMSD mean {st.mean(motif_rmsd):.3f} Å (max {max(motif_rmsd):.3f}); "
              f"satisfied {sum(motif_sat)}/{len(motif_sat)} designs ({100*sum(motif_sat)/len(motif_sat):.0f}%)")
    a2, b2 = agg[("A", 2.0)], agg[("B", 2.0)]
    if a2 and b2:
        print(f"\n  EMERGENT (λ=2): Run A mean dTM {st.mean(a2):+.3f} vs Run B {st.mean(b2):+.3f} "
              f"-> {'A≥B (zero-shot matches/beats trained)' if st.mean(a2) >= st.mean(b2) else 'B>A'}")


if __name__ == "__main__":
    main()
