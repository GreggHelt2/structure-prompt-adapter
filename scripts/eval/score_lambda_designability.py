"""Phase B of the λ-schedule sweep: designability of EXISTING arm backbones (dev ``docs/plan/57`` §3).

Runs flywheel Stages 2 to 4 on the PDBs ``probe_lambda_schedule.py`` already wrote, with no
regeneration (RFD3 is not bitwise reproducible, dev ``plan/RFD3_irreproducibility.md``, so the
backbones Phase A scored for adherence are the only ones whose designability is comparable to that
adherence): ProteinMPNN (N seqs, ``spa-dev``) then OpenFold3 refold (``spa-verify-of3``,
MSA-free/no-kernel) then best-of-N Cα **scRMSD** (< 2 Å ⇒ designable). Adherence and within-arm
diversity are recomputed from the same PDBs so one table can report all three together, which
``plan/58`` requires.

⭐ ONE OF3 PROCESS PER FOLD, NOT PER RUN
OF3 batching (``of3_batch_patch.py``, 3.32× at bs=8 on the A5000, dev ``plan/23`` §252) needs
**same-length** queries. Every design of one fold has the same length here (Phase A sets
``eval.length`` to the prompt's own residue count), so grouping by fold makes every batch uniform
while still amortizing the ~45 s model load over all of that fold's arms at once. Batching across
folds would mix lengths, which is exactly the case the shim was never validated for.

    conda run -n spa-dev python scripts/eval/score_lambda_designability.py \\
        --run-dir <phaseA run dir> --arms baseline,constant,bump --num-seqs 8 \\
        --of3-batch-size 8 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path("/home/user1/projects/spa")
_OUTPUTS_ROOT = Path(os.environ.get(
    "SPA_OUTPUTS_ROOT",
    Path(os.environ.get("SPA_PROJECT_ROOT", Path.home() / "projects" / "spa")) / "outputs"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="Phase-A run dir: <run>/<fold_id>/<arm>/*.pdb plus <run>/<fold_id>.json")
    ap.add_argument("--arms", required=True,
                    help="comma-separated arm names to score (winners + baseline + constant)")
    ap.add_argument("--folds", default=None, help="comma-separated fold ids; default = all in the run dir")
    ap.add_argument("--prompt-dir", default=str(
        ROOT / "training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb"))
    ap.add_argument("--prompt-pattern", default="AF-{id}-F1-model_v4_esmfold_v1.pdb")
    ap.add_argument("--num-seqs", type=int, default=8,
                    help="ProteinMPNN sequences per backbone (best-of-N designability)")
    ap.add_argument("--proteinmpnn-seed", type=int, default=42,
                    help="FIXED nonzero: ProteinMPNN's --seed 0 means a RANDOM seed each run (dev 21 §4.1)")
    ap.add_argument("--of3-batch-size", type=int, default=8)
    ap.add_argument("--of3-conda-env", default="spa-verify-of3")
    ap.add_argument("--scrmsd-cutoff", type=float, default=2.0)
    ap.add_argument("--out-dir", default=str(_OUTPUTS_ROOT / "_incoming" / "lambda_designability"))
    a = ap.parse_args()

    from omegaconf import OmegaConf
    from spa.eval.generate import Design
    from spa.eval.openfold3 import OF3Refolder
    from spa.eval.proteinmpnn import inverse_fold
    from spa.eval.score import (_as_struct, _ca_array, adherence, pairwise_tm_diversity,
                                score_design)

    run_dir = Path(a.run_dir).expanduser().resolve()
    out_root = Path(a.out_dir).expanduser().resolve(); out_root.mkdir(parents=True, exist_ok=True)
    arms = [s for s in a.arms.split(",") if s]
    folds = ([s for s in a.folds.split(",") if s] if a.folds
             else sorted(p.name for p in run_dir.iterdir() if p.is_dir()))
    print(f"[desigB] run={run_dir}\n[desigB] arms={arms}\n[desigB] folds={folds}")

    B = int(a.of3_batch_size)
    nokernel = Path(__file__).resolve().parents[2] / "configs/of3/of3_nokernel.yml"
    batch_shim = None
    of3_runner_yaml = str(nokernel)
    if B > 1:
        # The triton Evoformer kernel asserts attention-bias batch dim = 1 (evoformer.py:915), so bs>1
        # REQUIRES the nokernel base plus the shim. 3.32x at bs=8 on this card (dev plan/23 §252).
        y = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(str(nokernel)), resolve=False))
        y["data_module_args"] = {**dict(y.get("data_module_args") or {}), "batch_size": B}
        yf = out_root / f"of3_runner_bs{B}.yml"; OmegaConf.save(y, yf)
        of3_runner_yaml = str(yf)
        batch_shim = str(Path(__file__).resolve().parent / "of3_batch_patch.py")
        print(f"[desigB] OF3 batching ON: bs={B} via {yf.name} + {Path(batch_shim).name}")

    rows, per_arm_fold = [], {}
    t_start = time.time()
    for fold in folds:
        fdir = run_dir / fold
        prompt_pdb = Path(a.prompt_dir) / a.prompt_pattern.format(id=fold)
        prompt_struct = _as_struct(str(prompt_pdb)) if prompt_pdb.exists() else None
        if prompt_struct is None:
            print(f"[desigB] ⚠️ {fold}: no prompt structure at {prompt_pdb} -> adherence skipped")

        designs, owner = [], {}
        for arm in arms:
            for p in sorted((fdir / arm).glob("*.pdb")):
                aa = _as_struct(str(p))
                designs.append(Design(prompt_id=fold, condition=arm, lambda_scale=0.0, idx=0,
                                      path=p, n_residues=len(_ca_array(aa)), atom_array=aa))
                owner[p.stem] = arm
        if not designs:
            print(f"[desigB] ⚠️ {fold}: no PDBs for arms {arms} -> skipped")
            continue
        lens = sorted({d.n_residues for d in designs})
        print(f"\n[desigB] === {fold}: {len(designs)} backbone(s), lengths {lens}, "
              f"{len(designs) * a.num_seqs} OF3 folds ===")
        if len(lens) > 1:
            # Would break the same-length premise the batch shim relies on; refuse rather than
            # produce refolds whose batching was never validated.
            raise SystemExit(f"[desigB] ⛔ {fold} mixes design lengths {lens}; OF3 batching needs one "
                             f"length per process. Split this fold or set --of3-batch-size 1.")

        fout = out_root / fold; fout.mkdir(parents=True, exist_ok=True)
        cfg = OmegaConf.create({
            "paths": {
                "proteinmpnn_repo": os.environ.get("PROTEINMPNN_REPO",
                                                   str(ROOT / "needed_repos/ProteinMPNN")),
                "openfold3_ckpt": os.environ.get("OF3_CKPT", str(ROOT / "models/openfold3/of3-p2-155k.pt")),
                "openfold3_runner_yaml": of3_runner_yaml,
            },
            "eval": {
                "out_dir": str(fout),
                "proteinmpnn": {"num_seqs": int(a.num_seqs), "sampling_temp": 0.1, "batch_size": 1,
                                "seed": int(a.proteinmpnn_seed), "model_name": "v_48_020",
                                "weights_dir": None, "designs": None, "design_dir": None,
                                "out_dir": str(fout / "seqs")},
                # ⚠️ plddt_cutoff is INERT here and that is deliberate. `is_designable` applies the
                # pLDDT gate only when a confidence is actually supplied, and `score_design` below is
                # called without `plddt=`, so designability is **scRMSD only**. This matches
                # score_threeway_designability.py and hence every designability number this project has
                # published, which is why it is not "fixed": adding the gate here would silently make
                # these rates incomparable with results/02, /05 and /22. Recorded in results/29 §3.
                "score": {"scrmsd_cutoff": float(a.scrmsd_cutoff), "plddt_cutoff": 80.0,
                          "diversity": False},
            },
        })

        seqsets = inverse_fold(cfg, designs=designs)
        refolder = OF3Refolder(ckpt_path=cfg.paths.openfold3_ckpt, runner_yaml=of3_runner_yaml,
                               out_dir=str(fout / "of3"), conda_env=a.of3_conda_env,
                               batch_patch_shim=batch_shim)
        refolds_by_name = refolder.refold_all([ss for ss in seqsets if ss is not None])

        for d in designs:
            rf = refolds_by_name.get(d.path.stem)
            s = score_design(d, prompt=prompt_struct, refolds=rf, cfg=cfg)
            rows.append({"fold": fold, "arm": owner[d.path.stem], "design": d.path.stem,
                         "n_residues": d.n_residues, "n_refolds": len(rf or []),
                         "scrmsd": s.scrmsd, "designable": s.designable,
                         "tm_prompt": s.tm_norm_prompt, "tm_design": s.tm_norm_design,
                         "pdb": str(d.path)})
        # within-arm diversity over the SAME backbones, so all three metrics share one design set
        for arm in arms:
            structs = [d.atom_array for d in designs if owner[d.path.stem] == arm]
            sub = [r for r in rows if r["fold"] == fold and r["arm"] == arm]
            n_des = sum(1 for r in sub if r["designable"])
            tms = [r["tm_prompt"] for r in sub if r["tm_prompt"] is not None]
            per_arm_fold[(fold, arm)] = {
                "fold": fold, "arm": arm, "n": len(sub), "n_designable": n_des,
                "success_rate": (n_des / len(sub)) if sub else None,
                "tm_prompt_mean": (sum(tms) / len(tms)) if tms else None,
                "diversity_tm": pairwise_tm_diversity(structs) if len(structs) > 1 else None,
                "scrmsd_mean": (sum(r["scrmsd"] for r in sub if r["scrmsd"] == r["scrmsd"])
                                / max(1, sum(1 for r in sub if r["scrmsd"] == r["scrmsd"]))) if sub else None,
            }
        print(f"[desigB] {fold} arm rates: " + "  ".join(
            f"{arm}={per_arm_fold[(fold, arm)]['n_designable']}/{per_arm_fold[(fold, arm)]['n']}"
            for arm in arms))
        (out_root / "designability_partial.json").write_text(json.dumps(
            {"designs": rows, "per_arm_fold": list(per_arm_fold.values())}, indent=2, default=str))

    # ---- pooled per-arm summary (fold is the unit for the paired read; pooled is the headline rate) --
    print(f"\n[desigB] pooled over {len(folds)} fold(s)   (elapsed {time.time() - t_start:.0f}s)")
    print(f"[desigB] {'arm':<14}{'designable':>12}{'rate':>8}{'meanTM':>9}{'div_TM':>8}{'scRMSD':>9}")
    pooled = []
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        cells = [v for (f, ar), v in per_arm_fold.items() if ar == arm]
        n_des = sum(1 for r in sub if r["designable"])
        tms = [r["tm_prompt"] for r in sub if r["tm_prompt"] is not None]
        scr = [r["scrmsd"] for r in sub if r["scrmsd"] == r["scrmsd"]]
        divs = [c["diversity_tm"] for c in cells if c["diversity_tm"] is not None]
        rec = {"arm": arm, "n": len(sub), "n_designable": n_des,
               "success_rate": (n_des / len(sub)) if sub else None,
               "tm_prompt_mean": (sum(tms) / len(tms)) if tms else None,
               "diversity_tm_mean_over_folds": (sum(divs) / len(divs)) if divs else None,
               "scrmsd_mean": (sum(scr) / len(scr)) if scr else None}
        pooled.append(rec)
        h = lambda v: "  n/a" if v is None else f"{v:.3f}"
        print(f"[desigB] {arm:<14}{f'{n_des}/{len(sub)}':>12}{h(rec['success_rate']):>8}"
              f"{h(rec['tm_prompt_mean']):>9}{h(rec['diversity_tm_mean_over_folds']):>8}"
              f"{h(rec['scrmsd_mean']):>9}")

    payload = {"designs": rows, "per_arm_fold": list(per_arm_fold.values()), "pooled": pooled,
               "config": {"run_dir": str(run_dir), "arms": arms, "folds": folds,
                          "num_seqs": a.num_seqs, "proteinmpnn_seed": a.proteinmpnn_seed,
                          "of3_batch_size": B, "of3_runner_yaml": of3_runner_yaml,
                          "scrmsd_cutoff": a.scrmsd_cutoff, "scrmsd_atoms": "CA"}}
    (out_root / "designability.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[desigB] wrote {out_root / 'designability.json'}")
    print("[desigB] read: designable iff best-of-N Cα scRMSD < "
          f"{a.scrmsd_cutoff} Å. Higher meanTM = more adherent; LOWER div_TM = more diverse.")


if __name__ == "__main__":
    main()
