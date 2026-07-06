"""Stage the three-way sweep's motif-source + target-fold PDBs to GCS for the cloud H100 eval sweep.

The cloud three-way sweep (dev docs/plan/23) needs, in `gs://…/eval/threeway/prep/`, the handful of CDDB
PDBs it references: each **motif source** (for the pinned motif coords) and each **target fold G** (which
the cloud image ESM3-embeds live, and which scoring compares U against). This copies them (keeping the
`AF-<id>-F1-…` naming so `probe_hard_soft_free.py --pdb-dir <prep> --motif-source <id> --target <id>`
resolves them) + writes a manifest, then pushes to GCS. No raw-CDDB NGC pull is needed on the cloud.

    conda run -n spa-dev python scripts/eval/prep_threeway.py \
        --motifs A0A2X2KHU0:A2-20,A0A7C9GW19:A30-50 \
        --folds  A0A090ME36,A0A3P5VTL4,A0A6A0D1E8 \
        --gcs gs://genomancer-spa-cache/eval/threeway/prep
    # add --precompute-prompts to also stage <fold>.pt ESM3 prompts (optional; else the cloud embeds live)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

DEFAULT_PDB_DIR = ("/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/"
                   "atomistica_data_release/pdb")
PATTERN = "AF-{id}-F1-model_v4_esmfold_v1.pdb"
GCLOUD = "/home/user1/google-cloud-sdk/bin/gcloud"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motifs", required=True, help="comma list of <uniprot>:<segment>, e.g. A0A2X2KHU0:A2-20,...")
    ap.add_argument("--folds", required=True, help="comma list of target-fold uniprot ids (the SPA prompts G)")
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--gcs", default="gs://genomancer-spa-cache/eval/threeway/prep")
    ap.add_argument("--stage-dir", default=None, help="local staging dir (default: a temp under the pdb-dir parent)")
    ap.add_argument("--precompute-prompts", action="store_true",
                    help="also ESM3-embed each fold → <id>.pt (else the cloud image embeds live)")
    ap.add_argument("--no-push", action="store_true", help="stage locally only; skip the GCS upload")
    args = ap.parse_args()

    motif_ids = [m.split(":")[0].strip() for m in args.motifs.split(",") if m.strip()]
    motif_segs = [m.split(":", 1) for m in args.motifs.split(",") if m.strip()]
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    ids = sorted(set(motif_ids) | set(folds))

    stage = Path(args.stage_dir) if args.stage_dir else Path(args.pdb_dir).parent / "threeway_prep"
    stage.mkdir(parents=True, exist_ok=True)
    print(f"[prep] staging {len(ids)} PDB(s) ({len(motif_ids)} motif sources, {len(folds)} folds) -> {stage}")

    missing = []
    for pid in ids:
        src = Path(args.pdb_dir) / PATTERN.format(id=pid)
        if not src.exists():
            missing.append(pid); continue
        shutil.copy2(src, stage / src.name)
    if missing:
        raise SystemExit(f"[prep] FATAL: PDBs not found for {missing} under {args.pdb_dir}")

    if args.precompute_prompts:
        import torch
        from spa.prompt.esm3_prompt import esm3_prompt, load_esm3
        model = load_esm3("cuda:0")
        try:
            for pid in folds:
                p = esm3_prompt(str(stage / PATTERN.format(id=pid)), model, strip_bos_eos=True, use_sequence=False)
                torch.save(p.detach().float().cpu(), stage / f"{pid}.pt")
                print(f"[prep]   ESM3 prompt {pid}: [{p.shape[0]},{p.shape[1]}]")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest = {"motifs": [{"id": i, "seg": s} for i, s in motif_segs], "folds": folds,
                "pattern": PATTERN, "prompts_precomputed": bool(args.precompute_prompts)}
    (stage / "threeway_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[prep] wrote manifest ({len(motif_segs)} motifs × {len(folds)} folds)")

    if not args.no_push:
        print(f"[prep] pushing {stage}/* -> {args.gcs}/ (flat)")
        # cp the FILES directly (gcloud storage expands the wildcard) — NOT `cp -r "$dir/."`, which
        # nests the whole dir under {gcs}/<dirname>/ and breaks the run script's `$PREP_URI/*` pull.
        subprocess.run([GCLOUD, "storage", "cp", f"{stage}/*", f"{args.gcs}/"], check=True)
        print(f"[prep] done -> {args.gcs}")
    else:
        print(f"[prep] --no-push: staged locally at {stage} (upload skipped)")


if __name__ == "__main__":
    main()
