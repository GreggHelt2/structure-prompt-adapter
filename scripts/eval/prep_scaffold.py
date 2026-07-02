"""Local (A5000) prep for the scaffolding-eval cloud big-run (dev 17 §7 / 16 §9.5).

For each held-out prompt (a manifest, default configs/eval/manifest_lambda_sweep.yaml = 17 folds):
generate its ESM3 ``[N,1536]`` soft-prompt ``.pt``, copy the source PDB (the cloud needs it for
adherence + sub-region motif-RMSD/TM scoring), and compute a **deterministic contiguous sub-region S**
per granularity — emitted as a ``keep_range`` ``[start, end)`` so the cloud ``run_scaffold_eval.sh``
passes it compactly (``+eval.subregion.keep_range=[s,e]``). Embeddings are byte-identical to the cloud
(same ESM3 weights; memory esm3-weights-byte-identical-local-cloud), so producing them locally avoids a
250 GB cache pull / a cloud HF token — same rationale as prep_b1_full.py.

Granularities (dev discussion 2026-07-02):
  - ``domain``        — one domain of the clean 2-domain split (≈ the local A5000 pass; breadth + designability).
  - ``segment_small`` — a SMALL contiguous window (len ~ U(12, 30)): the SHARP test — sparse prompts are where a
                        sub-region-trained curriculum could finally beat the base full-prompt SPA (16 §9.5).

Usage:
  conda run -n spa-dev python scripts/eval/prep_scaffold.py \
    --manifest configs/eval/manifest_lambda_sweep.yaml \
    --pdb-dir /home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb \
    --out-dir /tmp/.../scaffold_prep --grans domain,segment_small \
    --gcs-uri gs://genomancer-spa-cache/eval/scaffold/prep
Outputs under --out-dir: <id>.pt + <id>.pdb (one each) + scaffold_resolved.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def _prompt_seed(base_seed: int, pid: str) -> int:
    import hashlib
    h = int(hashlib.sha1(pid.encode()).hexdigest(), 16)
    return (base_seed ^ (h & 0xFFFFFFFF)) & 0xFFFFFFFF


def _domain_range(n, pdb, rng):
    """Contiguous [start, end) of one domain (or a segment fallback) via subregion_pad_mask."""
    from spa.data.granularity import subregion_pad_mask
    _, pad = subregion_pad_mask(n, weights={"global": 0, "segment": 0, "domain": 1.0},
                                min_seg=12, pdb_path=pdb, rng=rng)
    if pad is None:                       # degenerate -> whole structure; fall back to a small window
        return _small_range(n, rng)
    idx = np.nonzero(~pad)[0]
    return int(idx[0]), int(idx[-1]) + 1


def _small_range(n, rng, lo=12, hi=30):
    """A SMALL contiguous window [start, end), len ~ U(lo, min(hi, n-1))."""
    hi = min(hi, max(lo + 1, n - 1))
    seg = int(rng.randint(lo, hi + 1)) if n > lo else max(1, n // 2)
    seg = min(seg, n)
    start = int(rng.randint(0, n - seg + 1))
    return start, start + seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/eval/manifest_lambda_sweep.yaml")
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grans", default="domain,segment_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gcs-uri", default="", help="if set, gcloud storage cp the prep dir here after writing")
    args = ap.parse_args()

    import torch

    from spa.eval.score import _as_struct, _ca_array
    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3

    grans = [g.strip() for g in args.grans.split(",") if g.strip()]
    man = yaml.safe_load(open(args.manifest))
    pattern = man["pdb_pattern"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = load_esm3(torch.device("cuda"))

    resolved = []
    for p in man["prompts"]:
        uid = p["id"]
        pdb = Path(args.pdb_dir) / pattern.format(id=uid)
        if not pdb.exists():
            print(f"[prep] MISSING PDB {uid}: {pdb}"); sys.exit(1)
        emb = esm3_prompt(pdb, model, strip_bos_eos=True).to("cpu", torch.float32).contiguous()
        torch.save(emb, out / f"{uid}.pt")
        shutil.copyfile(pdb, out / f"{uid}.pdb")
        n = len(_ca_array(_as_struct(str(pdb))))
        assert emb.shape[0] == n, f"{uid}: ESM3 N={emb.shape[0]} != CA {n} (strip_bos_eos?)"
        rng = np.random.RandomState(_prompt_seed(args.seed, uid))
        gmap = {}
        for g in grans:
            s, e = (_domain_range(n, str(pdb), rng) if g == "domain" else _small_range(n, rng))
            gmap[g] = [int(s), int(e)]
        resolved.append({"id": uid, "len": int(n), "fold": p.get("fold"), "grans": gmap})
        print(f"[prep] {uid}: emb [{emb.shape[0]},{emb.shape[1]}]  " +
              "  ".join(f"{g}={gmap[g]}({gmap[g][1]-gmap[g][0]}res)" for g in grans))

    payload = {"manifest": args.manifest, "seed": args.seed, "grans": grans, "prompts": resolved}
    (out / "scaffold_resolved.json").write_text(json.dumps(payload, indent=2))
    print(f"[prep] wrote {len(resolved)} prompts × {len(grans)} gran -> {out}/scaffold_resolved.json")

    if args.gcs_uri:
        import subprocess
        gc = "/home/user1/google-cloud-sdk/bin/gcloud"
        dst = args.gcs_uri.rstrip("/") + "/"
        for f in sorted(out.iterdir()):
            subprocess.run([gc, "storage", "cp", str(f), dst], check=True)
        print(f"[prep] staged {len(list(out.iterdir()))} files -> {args.gcs_uri}")


if __name__ == "__main__":
    main()
