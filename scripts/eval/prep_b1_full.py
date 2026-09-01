"""Local (A5000) prep for the B1-full cloud designability run.

For each pinned prompt (configs/eval/manifest_b1_full.yaml): generate its ESM3 ``[L,1536]`` soft-prompt
``.pt`` + DSSP-carve its full-length hard self-motif (scripts/eval/carve_motif.py), copy the source PDB
(the cloud needs it for ``eval.motif.source_pdb`` + ``motif_rmsd``), and emit a resolved manifest the
cloud ``run_eval.sh`` consumes. Embeddings are byte-identical to the cloud (same ESM3 weights; memory
esm3-weights-byte-identical-local-cloud), so producing them locally avoids a 250 GB cache pull / a cloud
HF token. Asserts prompt length == contig length (generate.build_motif requires it).

Usage: conda run -n spa-dev python scripts/eval/prep_b1_full.py \
         --manifest configs/eval/manifest_b1_full.yaml \
         --pdb-dir "$SPA_PROJECT_ROOT/training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb" \
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
    ap.add_argument("--out-name", default="b1_full_resolved.json",
                    help="resolved-manifest filename (keep the default unless prepping a second set "
                         "into the same directory)")
    # Fallbacks for manifests that do not pin the run config (curated15, lambda_sweep). Ignored when
    # the manifest declares its own, which manifest_b1_full.yaml does.
    ap.add_argument("--spa-ckpt", default="spa-Nx1536-uncond/spa_C_final.pt")
    ap.add_argument("--lambda-scale", default=1)
    ap.add_argument("--num-designs", type=int, default=8)
    ap.add_argument("--num-seqs", type=int, default=8)
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
        # `band` is optional: manifest_b1_full.yaml declares it (le256 / gt256 drive which prompts the
        # cloud long-10 run selects), but manifest_curated15.yaml and manifest_lambda_sweep.yaml do not.
        # Derive it from the length rather than requiring it, so ONE prep path serves every manifest
        # (dev docs/plan/81 §5b: row 2.1's 17-fold set is manifest_lambda_sweep.yaml).
        band = p.get("band") or ("le256" if c["len"] <= 256 else "gt256")
        resolved.append({
            "id": uid, "len": c["len"], "fold": p.get("fold"), "band": band,
            "pt": f"{uid}.pt", "pdb": f"{uid}.pdb", "contig": c["contig"], "n_motif": c["n_motif"],
        })
        print(f"[prep] {uid}: emb [{emb.shape[0]},{emb.shape[1]}]  motif {c['n_motif']}res  contig {c['contig']}")

    # The four run-config keys are optional for the same reason `band` is: only manifest_b1_full.yaml
    # pins them. A manifest without them still preps fine and the driver supplies its own values; the
    # CLI flags below let a caller pin them explicitly. b1_full's own output is unchanged, because it
    # declares all four.
    payload = {
        "spa_ckpt": man.get("spa_ckpt", args.spa_ckpt),
        "lambda_scale": man.get("lambda_scale", args.lambda_scale),
        "num_designs": man.get("num_designs", args.num_designs),
        "num_seqs": man.get("num_seqs", args.num_seqs),
        "source_manifest": str(Path(args.manifest).resolve()),
        "prompts": resolved,
    }
    (out / args.out_name).write_text(json.dumps(payload, indent=2))
    print(f"[prep] wrote {len(resolved)} prompts -> {out}/{args.out_name}")


if __name__ == "__main__":
    main()
