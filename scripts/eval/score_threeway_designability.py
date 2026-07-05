"""Designability (self-consistency scRMSD) for the three-way A⊕B⊕C designs — is the weird backbone real?

Runs flywheel Stages 2–4 on EXISTING `probe_hard_soft_free.py` backbones (no regeneration; the three-way
design isn't reproducible by the stock generator): ProteinMPNN (N seqs, spa-dev) → OpenFold3 refold
(spa-verify-of3, MSA-free/no-kernel) → best-of-K Cα **scRMSD** (< 2 Å ⇒ designable) + **refold-side
motif-RMSD** (does the pinned motif survive a redesigned sequence?). Methodology: dev `docs/results/01`;
spec: dev `docs/plan/21`. Adherence (U→G TM etc.) is already covered by the probe — this adds the
foldability leg the probe deliberately skips.

    conda run -n spa-dev python scripts/eval/score_threeway_designability.py \
        --pdbs <design1.pdb> <design2.pdb> ... --contig '90,A2-20,120' \
        --motif-source <A0A2X2KHU0 pdb> --num-seqs 8 --out-dir outputs/eval/threeway_designability
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/user1/projects/spa")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdbs", nargs="+", required=True, help="design PDB backbones to score")
    ap.add_argument("--contig", default="90,A2-20,120", help="the design's RFD3 contig (BAC = '90,A2-20,120')")
    ap.add_argument("--motif-source", default=str(
        ROOT / "training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb/"
               "AF-A0A2X2KHU0-F1-model_v4_esmfold_v1.pdb"))
    ap.add_argument("--num-seqs", type=int, default=8, help="ProteinMPNN sequences per backbone (best-of-K)")
    # Cloud-portability overrides (default: $ENV → local); on the H100: /opt/ProteinMPNN,
    # /workspace/weights/of3-p2-155k.pt, configs/of3/of3_triton.yml.
    ap.add_argument("--proteinmpnn-repo", default=None, help="ProteinMPNN repo (default: $PROTEINMPNN_REPO or local)")
    ap.add_argument("--of3-ckpt", default=None, help="OpenFold3 ckpt (default: $OF3_CKPT or local)")
    ap.add_argument("--of3-runner-yaml", default=None, help="OF3 runner yaml (default: $OF3_RUNNER_YAML or local of3_nokernel.yml)")
    ap.add_argument("--of3-conda-env", default="spa-verify-of3", help="conda env hosting OpenFold3 (same local + cloud)")
    ap.add_argument("--out-dir", default=str(ROOT / "structure-prompt-adapter/outputs/eval/threeway_designability"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from spa.eval.generate import Design, _parse_contig_motif
    from spa.eval.openfold3 import OF3Refolder
    from spa.eval.proteinmpnn import inverse_fold
    from spa.eval.score import _as_struct, _ca_array, score_design, source_positions

    import os
    _p = lambda ov, env, dflt: str(ov or os.environ.get(env) or dflt)

    out_dir = Path(args.out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({
        "paths": {
            "proteinmpnn_repo": _p(args.proteinmpnn_repo, "PROTEINMPNN_REPO", ROOT / "needed_repos/ProteinMPNN"),
            "openfold3_ckpt": _p(args.of3_ckpt, "OF3_CKPT", ROOT / "models/openfold3/of3-p2-155k.pt"),
            "openfold3_runner_yaml": _p(args.of3_runner_yaml, "OF3_RUNNER_YAML",
                                        ROOT / "structure-prompt-adapter/configs/of3/of3_nokernel.yml"),
        },
        "eval": {
            "out_dir": str(out_dir),
            "proteinmpnn": {"num_seqs": int(args.num_seqs), "sampling_temp": 0.1, "batch_size": 1,
                            "model_name": "v_48_020", "weights_dir": None,
                            "designs": None, "design_dir": None, "out_dir": str(out_dir / "seqs")},
            "score": {"scrmsd_cutoff": 2.0, "plddt_cutoff": 80.0, "diversity": False},
        },
    })

    # Motif spec in the design frame (contig → design indices + source positional Cα indices; review #1).
    parsed = _parse_contig_motif(args.contig)                       # [(design_idx, chain, author_resid), ...]
    source_struct = _as_struct(args.motif_source)
    design_idx = [d for d, _c, _r in parsed]
    src_pos = source_positions(source_struct, [(c, r) for _d, c, r in parsed])
    motif_score = (source_struct, design_idx, src_pos)
    print(f"[desig] motif: {len(design_idx)} residues at design idx [{min(design_idx)}..{max(design_idx)}] "
          f"(contig {args.contig!r})")

    designs = []
    for p in args.pdbs:
        p = Path(p); aa = _as_struct(str(p))
        designs.append(Design(prompt_id=p.stem, condition="threeway", lambda_scale=3.0, idx=0,
                              path=p, n_residues=len(_ca_array(aa)), atom_array=aa))
    print(f"[desig] scoring {len(designs)} design(s), N={args.num_seqs} seqs each "
          f"({len(designs) * args.num_seqs} OF3 folds)")

    # Stage 2 — ProteinMPNN
    seqsets = inverse_fold(cfg, designs=designs)
    # Stage 3 — OpenFold3 refold (separate env, one model-load for the whole matrix)
    refolder = OF3Refolder(ckpt_path=cfg.paths.openfold3_ckpt, runner_yaml=cfg.paths.openfold3_runner_yaml,
                           out_dir=str(out_dir / "of3"), conda_env=args.of3_conda_env)
    refolds_by_name = refolder.refold_all([ss for ss in seqsets if ss is not None])

    # Stage 4 — score (designability scRMSD + refold-side motif survival)
    print(f"\n{'design':<30}{'scRMSD(Å)':>10}{'designable':>12}{'pLDDT':>8}{'motifRMSD_design':>18}{'motifRMSD_refold':>18}")
    rows = []
    for d in designs:
        refolds = refolds_by_name.get(d.path.stem)
        s = score_design(d, prompt=None, refolds=refolds, motif=motif_score, cfg=cfg)
        rec = {"design": d.path.stem, "scrmsd": s.scrmsd, "designable": s.designable, "plddt": s.plddt,
               "motif_rmsd_design": s.motif_rmsd, "motif_rmsd_refold": s.motif_rmsd_refold,
               "best_refold_idx": s.best_refold_idx}
        rows.append(rec)
        f = lambda x: "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        print(f"{rec['design']:<30}{f(s.scrmsd):>10}{str(s.designable):>12}{f(s.plddt):>8}"
              f"{f(s.motif_rmsd):>18}{f(s.motif_rmsd_refold):>18}")

    (out_dir / "designability.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n[desig] wrote {out_dir / 'designability.json'}")
    print("[read] designable iff best-of-K scRMSD < 2.0 Å. A weird splayed backbone that no sequence "
          "folds back to will show HIGH scRMSD — that is the honest 'is this a real protein?' test.")


if __name__ == "__main__":
    main()
