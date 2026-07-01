"""Aggregate the variant soft-designability results into the C/B/A comparison table.

For each SPA variant (C=N×1536 / B=1×1536 / A=1×32-CLSS) on the curated-15: baseline vs SPA, both
**designability** (d_succ = fraction scRMSD < cutoff, median scRMSD) AND **adherence** (median prompt-TM,
ΔTM vs baseline). Answers the completeness question: do the cheaper pooled variants — especially the
1×32 CLSS encoder the abstract names — hold their own on designability, as they already do on adherence?
Reads ``<variant>/<id>.json`` under ``--results-dir`` + the manifest (fold/len labels, band filter).

Usage: python aggregate_variant_desig.py --results-dir <dir> --manifest b1_full_resolved.json
       [--scrmsd-cutoff 2.0] [--band le256] [--out summary.json]
(Download first, e.g.: gcloud storage rsync -r gs://…/eval/variant_desig <dir>)
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

VARIANTS = ["C_n_by_1536", "B_1_by_1536", "A_1_by_32"]
LABEL = {"C_n_by_1536": "C (N×1536)", "B_1_by_1536": "B (1×1536)", "A_1_by_32": "A (1×32 CLSS)"}


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def per_prompt(scores, cond, cutoff):
    """Baseline/SPA designability + adherence summary for one prompt's DesignScore list."""
    g = [s for s in scores if s.get("condition") == cond]
    des = [(s["scrmsd"] < cutoff) for s in g if s.get("scrmsd") is not None]
    return {
        "n": len(g),
        "d_succ": round(sum(des) / len(des), 3) if des else None,
        "scrmsd_med": _med([s.get("scrmsd") for s in g]),
        "tm_med": _med([s.get("tm_norm_prompt") for s in g]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scrmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--band", default="le256")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = {p["id"]: p for p in json.load(open(a.manifest))["prompts"] if p["band"] == a.band}
    per_variant_rows: dict = {}
    for v in VARIANTS:
        vdir = Path(a.results_dir) / v
        rows = []
        for pid, meta in man.items():
            f = vdir / f"{pid}.json"
            if not f.exists():
                continue
            scores = json.load(open(f)).get("scores", [])
            rows.append({
                "id": pid, "fold": meta["fold"], "len": meta["len"],
                "baseline": per_prompt(scores, "baseline", a.scrmsd_cutoff),
                "spa": per_prompt(scores, "spa", a.scrmsd_cutoff),
            })
        per_variant_rows[v] = rows

    print(f"\n=== Variant soft-designability + adherence (curated-{len(man)}, band={a.band}; scRMSD<{a.scrmsd_cutoff}Å) ===")
    print(f"{'variant':<16}{'n':>3}  {'base_dsucc':>10}{'spa_dsucc':>10}{'Ddsucc':>8}   {'base_TM':>8}{'spa_TM':>8}{'DTM':>7}")
    agg: dict = {}
    for v in VARIANTS:
        rows = per_variant_rows[v]
        if not rows:
            print(f"{LABEL[v]:<16}  (no results staged yet)")
            continue
        bd = _mean([r["baseline"]["d_succ"] for r in rows])
        sd = _mean([r["spa"]["d_succ"] for r in rows])
        btm = _mean([r["baseline"]["tm_med"] for r in rows])
        stm = _mean([r["spa"]["tm_med"] for r in rows])
        dd = round(sd - bd, 3) if (bd is not None and sd is not None) else None
        dtm = round(stm - btm, 3) if (btm is not None and stm is not None) else None
        agg[v] = {"n": len(rows), "baseline_d_succ": bd, "spa_d_succ": sd, "delta_d_succ": dd,
                  "baseline_tm": btm, "spa_tm": stm, "delta_tm": dtm}
        print(f"{LABEL[v]:<16}{len(rows):>3}  {str(bd):>10}{str(sd):>10}{str(dd):>8}   {str(btm):>8}{str(stm):>8}{str(dtm):>7}")

    # near-equivalence check: spread of SPA metrics across variants
    sds = [agg[v]["spa_d_succ"] for v in agg if agg[v]["spa_d_succ"] is not None]
    stms = [agg[v]["spa_tm"] for v in agg if agg[v]["spa_tm"] is not None]
    if len(sds) > 1:
        print(f"\n  SPA d_succ spread across variants: {max(sds) - min(sds):.3f}  (min {min(sds)}, max {max(sds)})")
    if len(stms) > 1:
        print(f"  SPA adherence-TM spread across variants: {max(stms) - min(stms):.3f}  (min {min(stms)}, max {max(stms)})")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"scrmsd_cutoff": a.scrmsd_cutoff, "band": a.band, "per_variant": agg,
             "per_prompt": per_variant_rows}, indent=2))
        print(f"\n[agg] wrote {a.out}")


if __name__ == "__main__":
    main()
