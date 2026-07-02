"""Auto-pick a structurally-distinct decoy fold for the wrong-prompt specificity control (dev 16 §11).

Given a target protein, choose a decoy from the curated-15 held-out pool that is (a) of comparable
length and (b) maximally structurally distinct. The decoy's ESM3 prompt then feeds `eval.prompt_pdb`
for a `spa` run scored against the *target*: if SPA steering is prompt-specific, adherence-to-target
collapses to ~baseline (the design followed the decoy, not the target).

Selection rule (deterministic — same target always yields the same decoy):
  1. candidates = pool \\ {target}, keep those with |len - L| <= len_tol * L  (comparable length).
  2. TM-align target vs each candidate; distinctness score = max(tm_norm_chain1, tm_norm_chain2)
     (conservative: "similar" if EITHER length-normalization is high).
  3. pick the LOWEST score (most distinct); tie-break by id. Warn if none is below max_tm (the pool
     has no clearly-different fold at this length → the decoy would only be weakly distinct).

Reuses spa.eval.score.tm_score so parsing + normalization match the scorer. Pure CPU (tmtools).

    conda run -n spa-dev python scripts/eval/pick_decoy.py --target A0A522W419
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/eval/manifest_curated15.yaml"
DEFAULT_PDB_DIR = ("/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/"
                   "atomistica_data_release/pdb")


def _load_pool(manifest_path, pdb_dir):
    import yaml
    m = yaml.safe_load(open(manifest_path))
    pat = m["pdb_pattern"]
    return [{"id": str(p["id"]), "len": int(p["len"]), "fold": p.get("fold", "?"),
             "pdb": str(Path(pdb_dir) / pat.format(id=str(p["id"])))} for p in m["prompts"]]


def _target_record(target, pool_by_id, pdb_dir):
    if target in pool_by_id:
        return pool_by_id[target]
    # target given as a PDB path: compute its length from the structure
    from spa.eval.score import _as_struct, _ca_array
    L = int((_ca_array(_as_struct(str(target))).array_length()))
    return {"id": Path(target).stem, "len": L, "fold": "?", "pdb": str(target)}


def pick_decoy(target, manifest=DEFAULT_MANIFEST, pdb_dir=DEFAULT_PDB_DIR, len_tol=0.25, max_tm=0.4):
    from spa.eval.score import tm_score
    pool = _load_pool(manifest, pdb_dir)
    by_id = {p["id"]: p for p in pool}
    tgt = _target_record(target, by_id, pdb_dir)
    L = tgt["len"]
    cands = [p for p in pool if p["pdb"] != tgt["pdb"] and abs(p["len"] - L) <= len_tol * L]
    if not cands:
        raise SystemExit(f"no candidate within +/-{len_tol:.0%} of L={L}; widen --len-tol")
    for p in cands:
        tm1, tm2 = tm_score(tgt["pdb"], p["pdb"])
        p["tm"] = float(max(tm1, tm2))
    cands.sort(key=lambda p: (p["tm"], p["id"]))
    return tgt, cands[0], cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target UniProt id (in the manifest) or a PDB path")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--len-tol", type=float, default=0.25)
    ap.add_argument("--max-tm", type=float, default=0.4)
    ap.add_argument("--json", help="optional path to write the chosen decoy record")
    args = ap.parse_args()

    tgt, decoy, cands = pick_decoy(args.target, args.manifest, args.pdb_dir, args.len_tol, args.max_tm)
    print(f"target: {tgt['id']}  len={tgt['len']}  fold={tgt['fold']}   "
          f"(len window ±{args.len_tol:.0%} -> {round((1 - args.len_tol) * tgt['len'])}-{round((1 + args.len_tol) * tgt['len'])})")
    print(f"{'candidate':16}{'len':>5}{'fold':>11}{'TMmax':>8}")
    for p in cands:
        mark = "   <-- DECOY" if p["id"] == decoy["id"] else ""
        print(f"{p['id']:16}{p['len']:>5}{p['fold']:>11}{p['tm']:>8.3f}{mark}")
    if decoy["tm"] >= args.max_tm:
        print(f"\nWARNING: most-distinct TMmax {decoy['tm']:.3f} >= max_tm {args.max_tm} — no clearly "
              f"different fold at this length; decoy is only weakly distinct (widen --len-tol or enlarge the pool).")
    print(f"\nDECOY -> {decoy['id']}  len={decoy['len']}  fold={decoy['fold']}  TMmax={decoy['tm']:.3f}")
    print(f"pdb: {decoy['pdb']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"target": tgt, "decoy": decoy}, fh, indent=2)
        print(f"[pick_decoy] wrote {args.json}")


if __name__ == "__main__":
    main()
