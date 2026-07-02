"""Aggregate the scaffolding big-run results into the multigran-vs-base tables (dev 17 §7 / 16 §9.5).

Consumes a LOCAL copy of the staged results (gs://.../eval/scaffold/results pulled to --results-dir) +
the prep dir (for the prompt PDBs + keep_range). Per granularity × ckpt-group it reports:
  - sub-region motif-RMSD  (design[S] vs prompt[S], strict) — from each flywheel_results.json summary,
  - designability d_succ / scRMSD                            — from the summaries,
  - sub-region TM          (design[S] vs prompt[S], the FAIR soft metric) — recomputed from staged PDBs,
and the headline Δ(spa-multigran − spa-base).

Usage: conda run -n spa-dev python scripts/eval/aggregate_scaffold.py \
         --results-dir /tmp/scaffold_results --prep-dir /tmp/scaffold_prep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _region_tm(design_pdb, prompt_ca, keep):
    import tmtools
    from spa.eval.score import _as_struct, _ca_array, _coords64, _seq_of
    d = _ca_array(_as_struct(str(design_pdb)))[keep]
    p = prompt_ca[keep]
    r = tmtools.tm_align(_coords64(d), _coords64(p), _seq_of(d), _seq_of(p))
    return max(float(r.tm_norm_chain1), float(r.tm_norm_chain2))


def _summ(js, cond):
    """(mean motif_rmsd, success_rate) for a condition from a flywheel_results.json payload."""
    for s in js.get("summaries", []):
        if s["condition"] == cond:
            mr = (s.get("motif_rmsd") or {}).get("mean")
            return mr, s.get("success_rate")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--tags", default="multigran,base")
    args = ap.parse_args()

    from spa.eval.score import _as_struct, _ca_array

    prep = Path(args.prep_dir)
    resolved = json.loads((prep / "scaffold_resolved.json").read_text())
    keep_of = {p["id"]: {g: list(range(s, e)) for g, (s, e) in p["grans"].items()} for p in resolved["prompts"]}
    grans = resolved["grans"]
    rdir = Path(args.results_dir)

    # group -> gran -> lists
    acc = {}
    for tag in args.tags.split(","):
        for gran in grans:
            for jf in sorted((rdir / tag / gran).glob("*.json")):
                pid = jf.stem
                js = json.loads(jf.read_text())
                prompt_ca = _ca_array(_as_struct(str(prep / f"{pid}.pdb")))
                keep = keep_of[pid][gran]
                # spa-<tag> (+ baseline from the multigran group only)
                conds = [("spa", f"spa-{tag}")] + ([("baseline", "baseline")] if tag == "multigran" else [])
                for cond, label in conds:
                    mr, dsucc = _summ(js, cond)
                    tms = []
                    for pdb in (rdir / tag / gran / pid).glob(f"*_{cond}_*.pdb"):
                        try:
                            tms.append(_region_tm(pdb, prompt_ca, keep))
                        except Exception:
                            pass
                    acc.setdefault((label, gran), {"motif": [], "dsucc": [], "tm": []})
                    d = acc[(label, gran)]
                    if mr is not None: d["motif"].append(mr)
                    if dsucc is not None: d["dsucc"].append(dsucc)
                    d["tm"] += tms

    m = lambda xs: (mean(xs) if xs else None)
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    for gran in grans:
        print(f"\n=== granularity: {gran} ===")
        print(f"{'group':<16}{'motifRMSD↓':>12}{'subregTM↑':>11}{'d_succ↑':>9}")
        print("-" * 48)
        for label in ["baseline", "spa-base", "spa-multigran"]:
            d = acc.get((label, gran))
            if not d:
                continue
            print(f"{label:<16}{f(m(d['motif'])):>12}{f(m(d['tm'])):>11}{f(m(d['dsucc'])):>9}")
        b, mg = acc.get(("spa-base", gran)), acc.get(("spa-multigran", gran))
        if b and mg:
            dtm = (m(mg["tm"]) - m(b["tm"])) if (m(mg["tm"]) and m(b["tm"])) else None
            print(f"  Δ(mg−base) sub-region TM: {('%+.3f' % dtm) if dtm is not None else 'n/a'} "
                  f"(claim: multigran > base ⇒ positive)")


if __name__ == "__main__":
    main()
