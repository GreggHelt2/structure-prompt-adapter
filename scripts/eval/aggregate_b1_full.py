"""Aggregate the B1-full per-prompt flywheel results into the designability table.

Hard⊕soft, K=8/N=8, λ=1: baseline vs SPA — designable-rate (``d_succ`` = fraction scRMSD < cutoff),
median scRMSD, and motif-survival (refold-side motif-RMSD) — overall + by length band (le256/gt256)
+ by fold class. Reads ``<id>.json`` (flywheel_results) from ``--results-dir`` and the resolved
manifest (``--manifest`` = b1_full_resolved.json, for fold/band/len labels).

Usage: python aggregate_b1_full.py --results-dir <dir of <id>.json> --manifest b1_full_resolved.json
       [--scrmsd-cutoff 2.0] [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


def _rate(flags):
    flags = [f for f in flags if f is not None]
    return round(sum(bool(f) for f in flags) / len(flags), 3) if flags else None


def per_prompt(scores, cond, cutoff):
    """Baseline/SPA designability summary for one prompt's DesignScore list."""
    g = [s for s in scores if s.get("condition") == cond]
    scr = [s.get("scrmsd") for s in g]
    des = [(s["scrmsd"] < cutoff) for s in g if s.get("scrmsd") is not None]
    mot = [s.get("motif_rmsd_refold") for s in g]
    return {
        "n": len(g),
        "d_succ": round(sum(des) / len(des), 3) if des else None,
        "scrmsd_med": _med(scr),
        "motif_surv_med": _med(mot),
    }


def _grp(rows, keyfn):
    """Mean baseline/SPA d_succ (+ Δ) over a group of per-prompt rows."""
    out = {}
    for r in rows:
        out.setdefault(keyfn(r), []).append(r)
    table = {}
    for k, rs in out.items():
        bd = [r["baseline"]["d_succ"] for r in rs if r["baseline"]["d_succ"] is not None]
        sd = [r["spa"]["d_succ"] for r in rs if r["spa"]["d_succ"] is not None]
        ms = [r["spa"]["motif_surv_med"] for r in rs if r["spa"]["motif_surv_med"] is not None]
        b = round(st.mean(bd), 3) if bd else None
        s = round(st.mean(sd), 3) if sd else None
        table[k] = {
            "n_prompts": len(rs),
            "baseline_d_succ": b,
            "spa_d_succ": s,
            "delta_d_succ": round(s - b, 3) if (b is not None and s is not None) else None,
            "spa_motif_surv_med": round(st.mean(ms), 3) if ms else None,
        }
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scrmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = {p["id"]: p for p in json.load(open(a.manifest))["prompts"]}
    rows = []
    missing = []
    for pid, meta in man.items():
        f = Path(a.results_dir) / f"{pid}.json"
        if not f.exists():
            missing.append(pid)
            continue
        scores = json.load(open(f)).get("scores", [])
        rows.append({
            "id": pid, "fold": meta["fold"], "band": meta["band"], "len": meta["len"],
            "baseline": per_prompt(scores, "baseline", a.scrmsd_cutoff),
            "spa": per_prompt(scores, "spa", a.scrmsd_cutoff),
        })

    print(f"\n=== B1-full designability (hard⊕soft, K=8/N=8, λ=1; scRMSD<{a.scrmsd_cutoff}Å) ===")
    print(f"{'id':<12}{'fold':<10}{'len':>4}  {'base_dsucc':>10}{'spa_dsucc':>10}{'d':>7}  {'base_scr':>9}{'spa_scr':>9}  {'motif_surv':>10}")
    for r in sorted(rows, key=lambda x: x["len"]):
        b, s = r["baseline"], r["spa"]
        bd, sd = b["d_succ"], s["d_succ"]
        delta = "" if (bd is None or sd is None) else f"{sd - bd:+.2f}"
        print(f"{r['id']:<12}{r['fold']:<10}{r['len']:>4}  "
              f"{str(bd):>10}{str(sd):>10}{delta:>7}  "
              f"{str(b['scrmsd_med']):>9}{str(s['scrmsd_med']):>9}  {str(s['motif_surv_med']):>10}")

    overall = _grp(rows, lambda r: "ALL")
    by_band = _grp(rows, lambda r: r["band"])
    by_fold = _grp(rows, lambda r: r["fold"])
    for title, tbl in [("OVERALL", overall), ("BY BAND", by_band), ("BY FOLD", by_fold)]:
        print(f"\n--- {title} (mean d_succ over prompts) ---")
        for k, v in sorted(tbl.items()):
            print(f"  {k:<12} n={v['n_prompts']:<3} baseline {v['baseline_d_succ']}  SPA {v['spa_d_succ']}  "
                  f"Δ {v['delta_d_succ']}  motif_surv {v['spa_motif_surv_med']}")

    if missing:
        print(f"\n[agg] MISSING {len(missing)} prompt result(s): {', '.join(missing)}")
    summary = {"scrmsd_cutoff": a.scrmsd_cutoff, "n_prompts": len(rows), "missing": missing,
               "per_prompt": rows, "overall": overall, "by_band": by_band, "by_fold": by_fold}
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[agg] wrote {a.out}")


if __name__ == "__main__":
    main()
