"""Local (A5000) prep for the B1-full cloud designability run.

For each pinned prompt (configs/eval/manifest_b1_full.yaml): generate its ESM3 ``[L,1536]`` soft-prompt
``.pt`` + DSSP-carve its full-length hard self-motif (scripts/eval/carve_motif.py), copy the source PDB
(the cloud needs it for ``eval.motif.source_pdb`` + ``motif_rmsd``), and emit a resolved manifest the
cloud ``run_eval.sh`` consumes. Embeddings are byte-identical to the cloud (same ESM3 weights; memory
esm3-weights-byte-identical-local-cloud), so producing them locally avoids a 250 GB cache pull / a cloud
HF token. Asserts prompt length == contig length (generate.build_motif requires it).

Usage: conda run -n spa-dev python scripts/eval/prep_b1_full.py \
         --manifest configs/eval/manifest_b1_full.yaml \
         --pdb-dir /home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb \
         --out-dir /tmp/.../b1_full_prep
Outputs under --out-dir: <id>.pt + <id>.pdb (one each) + b1_full_resolved.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-seg", type=int, default=2, help="motif segments to carve (H5 used 2)")
    args = ap.parse_args()

    import torch

    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from carve_motif import carve

    man = yaml.safe_load(open(args.manifest))
    pattern = man["pdb_pattern"]
    prompts = man["prompts"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_esm3(torch.device("cuda"))  # A5000 via CUDA_VISIBLE_DEVICES baked into spa-dev

    resolved = []
    for p in prompts:
        uid = p["id"]
        pdb = Path(args.pdb_dir) / pattern.format(id=uid)
        if not pdb.exists():
            print(f"[prep] MISSING PDB {uid}: {pdb}")
            sys.exit(1)
        # 1) ESM3 soft prompt [L,1536]
        emb = esm3_prompt(pdb, model, strip_bos_eos=True).to("cpu", torch.float32).contiguous()
        torch.save(emb, out / f"{uid}.pt")
        # 2) DSSP-carve full-length self-motif (hard)
        c = carve(str(pdb), n_seg=args.n_seg)
        # 3) copy source PDB (cloud motif source_pdb + motif_rmsd reference)
        shutil.copyfile(pdb, out / f"{uid}.pdb")
        # sanity: SPA prompt length must equal the motif contig length (generate.build_motif asserts it)
        assert emb.shape[0] == c["len"], f"{uid}: emb {emb.shape[0]} != contig len {c['len']}"
        resolved.append({
            "id": uid, "len": c["len"], "fold": p["fold"], "band": p["band"],
            "pt": f"{uid}.pt", "pdb": f"{uid}.pdb", "contig": c["contig"], "n_motif": c["n_motif"],
        })
        print(f"[prep] {uid}: emb [{emb.shape[0]},{emb.shape[1]}]  motif {c['n_motif']}res  contig {c['contig']}")

    payload = {
        "spa_ckpt": man["spa_ckpt"], "lambda_scale": man["lambda_scale"],
        "num_designs": man["num_designs"], "num_seqs": man["num_seqs"], "prompts": resolved,
    }
    (out / "b1_full_resolved.json").write_text(json.dumps(payload, indent=2))
    print(f"[prep] wrote {len(resolved)} prompts -> {out}/b1_full_resolved.json")


if __name__ == "__main__":
    main()
