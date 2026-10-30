#!/usr/bin/env python3
"""Prep for the plan/153 long-prompt steering experiment (soft-only self-prompt), GPU-FREE.

For each of the 28 CDDB test-split folds in configs/eval/manifest_longfold_steer.json:
  1) EXTRACT its ESM3 [L,1536] soft-prompt `.pt` from the full ESM3 cache tar (esm3_cache.tar) by name
     (AF-<id>-F1-model_v4_esmfold_v1.pt) -- these are the exact tensors every prior run used, so there
     is NO regeneration and NO GPU/ESM3 needed. (All 28 were verified present in the cache audit
     listing, plan/153 section 10.1.)
  2) copy the source CDDB `.pdb` (the cloud needs it for +eval.flywheel.prompt_struct, the adherence
     reference),
  3) emit longfold_steer_resolved.json (the manifest run_longfold_steer.sh reads from the staged PREP).

NO motif carve (soft-only), unlike prep_b1_full.py. Streaming a 234 GB tar over a laptop link is
impractical, so run this on a GCS-connected box (a cheap cloud CPU step, or the instance that reads
GCS at ~1 GB/s). Extraction needs no GPU.

Usage:
  # On a box with the cache tar hydrated locally (fastest):
  python prep_longfold_steer.py --manifest configs/eval/manifest_longfold_steer.json \
      --cache-tar /path/to/esm3_cache.tar --pdb-dir <CDDB pdb dir> --out-dir <prep>
  # Then stage: gcloud storage cp <prep>/* gs://genomancer-spa-cache/eval/longfold_steer/prep/
"""
import argparse, json, shutil, tarfile
from pathlib import Path

PT_PATTERN = "AF-{id}-F1-model_v4_esmfold_v1.pt"
PDB_PATTERN = "AF-{id}-F1-model_v4_esmfold_v1.pdb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/eval/manifest_longfold_steer.json")
    ap.add_argument("--cache-tar", required=True,
                    help="esm3_cache.tar (hydrated locally; entries keyed AF-<id>-...pt)")
    ap.add_argument("--pdb-dir", required=True,
                    help="CDDB pdb dir (AF-<id>-F1-model_v4_esmfold_v1.pdb)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", default="longfold_steer_resolved.json")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    want = {PT_PATTERN.format(id=p["id"]): p["id"] for p in man["prompts"]}
    # tar member names may be bare or prefixed with "./"; match on basename.
    found = {}
    with tarfile.open(a.cache_tar, "r") as tf:
        for m in tf:
            base = Path(m.name).name
            if base in want:
                uid = want[base]
                src = tf.extractfile(m)
                if src is None:
                    continue
                (out / f"{uid}.pt").write_bytes(src.read())
                found[uid] = True
                if len(found) == len(want):
                    break
    missing_pt = [p["id"] for p in man["prompts"] if p["id"] not in found]
    if missing_pt:
        raise SystemExit(f"[prep] FATAL: {len(missing_pt)} .pt not found in {a.cache_tar}: {missing_pt}")

    resolved = []
    for p in man["prompts"]:
        uid = p["id"]
        pdb = Path(a.pdb_dir) / PDB_PATTERN.format(id=uid)
        if not pdb.exists():
            raise SystemExit(f"[prep] FATAL: PDB missing for {uid}: {pdb}")
        shutil.copyfile(pdb, out / f"{uid}.pdb")
        resolved.append({"id": uid, "length": p["length"], "fold_class": p["fold_class"],
                         "pt": f"{uid}.pt", "pdb": f"{uid}.pdb"})

    payload = {k: man[k] for k in ("spa_ckpt", "lambda_scale", "num_designs", "num_seqs", "seeds")}
    payload["prompts"] = resolved
    (out / a.out_name).write_text(json.dumps(payload, indent=2))
    print(f"[prep] extracted {len(found)} .pt + {len(resolved)} .pdb -> {out}/{a.out_name}")


if __name__ == "__main__":
    main()
