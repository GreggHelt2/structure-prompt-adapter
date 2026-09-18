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
import re
import os
import pathlib
from pathlib import Path

# Run-artifact root — absolute + env-overridable, mirroring configs/paths/default.yaml's
# `outputs_root: ${oc.env:SPA_OUTPUTS_ROOT,${paths.project_root}/outputs}`. A *relative* default
# resolved against the invoking cwd and sent output into whichever repo the script was launched
# from; a *shared* default made runs overwrite each other. See dev docs/plan/30 §6.
_OUTPUTS_ROOT = Path(os.environ.get(
    "SPA_OUTPUTS_ROOT",
    Path(os.environ.get("SPA_PROJECT_ROOT", Path.home() / "projects" / "spa")) / "outputs"))


# Project root for INPUTS (ProteinMPNN repo, OF3 checkpoint, CDDB prompt PDBs). Resolved from
# $SPA_PROJECT_ROOT (default ~/projects/spa), matching configs/paths/default.yaml's `project_root`,
# so this is not bound to one machine. Run artifacts resolve separately, via _OUTPUTS_ROOT.
ROOT = Path(os.environ.get("SPA_PROJECT_ROOT", Path.home() / "projects" / "spa"))



def _prov_prompts(args):
    """Prompt-ish CLI args, for the provenance record's split lookup. Driver-agnostic:
    each probe names its targets differently (--g1/--g2, --target, --foreign)."""
    vals = []
    for k in ("g1", "g2", "target", "foreign", "prompt", "prompt_pdb", "fold", "folds"):
        v = getattr(args, k, None)
        if v:
            vals.extend(str(v).split(",") if isinstance(v, str) else [str(v)])
    return [v.strip() for v in vals if v and v.strip()]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdbs", nargs="+", required=True, help="design PDB backbones to score")
    ap.add_argument("--contig", default="90,A2-20,120", help="the design's RFD3 contig (BAC = '90,A2-20,120')")
    ap.add_argument("--motif-source", default=str(
        ROOT / "training_data/proteina-atomistica_data_vrelease/atomistica_data_release/pdb/"
               "AF-A0A2X2KHU0-F1-model_v4_esmfold_v1.pdb"))
    ap.add_argument("--num-seqs", type=int, default=16,
                    help="ProteinMPNN sequences per backbone (best-of-K). N>=16 halves the best-of-N miss-rate "
                         "for hard backbones (dev 21 §4.1 best-of-N-noise finding)")
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="Score best-of-k over only the FIRST k sequences (q0..q[k-1]) of each "
                         "backbone, instead of all --num-seqs. SOUND BECAUSE ProteinMPNN runs at "
                         "batch_size=1, so sequence i sits at stream position i (dev plan/100 section "
                         "2.2) and the first k ARE an N=k run. NEEDED BECAUSE designability is "
                         "best-of-N, so a rate at N=8 is systematically lower than the same designs at "
                         "N=16; this reports both from one run without regenerating.")
    ap.add_argument("--proteinmpnn-seed", type=int, default=42,
                    help="ProteinMPNN sampling seed — FIXED nonzero for reproducibility (--seed 0 = RANDOM "
                         "each run in ProteinMPNN, which made best-of-N non-reproducible; dev 21 §4.1)")
    # Cloud-portability overrides (default: $ENV → local); on the H100: /opt/ProteinMPNN,
    # /workspace/weights/of3-p2-155k.pt, configs/of3/of3_triton.yml.
    ap.add_argument("--proteinmpnn-repo", default=None, help="ProteinMPNN repo (default: $PROTEINMPNN_REPO or local)")
    ap.add_argument("--of3-ckpt", default=None, help="OpenFold3 ckpt (default: $OF3_CKPT or local)")
    ap.add_argument("--of3-runner-yaml", default=None, help="OF3 runner yaml (default: $OF3_RUNNER_YAML or local of3_nokernel.yml)")
    ap.add_argument("--of3-conda-env", default="spa-verify-of3", help="conda env hosting OpenFold3 (same local + cloud)")
    ap.add_argument("--of3-batch-size", type=int, default=1,
                    help="OF3 refold batch_size. >1 forces the of3_nokernel.yml base + the of3_batch_patch.py "
                         "shim (the triton kernels CANNOT batch — evoformer.py:915); ~2.5x at bs=8, folds "
                         "equivalent (dev 23 §7.8). bs=1 = original per-fold behavior, unchanged.")
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                    help="bitwise determinism, DEFAULT ON since 2026-09-18: OF3 refold via torch's "
                         "deterministic scatter_add (dev plan/91). Requires bs=1 (the default); OF3Refolder "
                         "refuses it with --of3-batch-size>1. Pass --no-deterministic for a deliberate "
                         "non-deterministic refold, and say so in the run's plan/106 row. ⛔ This was "
                         "store_true (default OFF) until row 25 refolded on stock OF3 while being "
                         "labelled contract v2; an omitted opt-in reads exactly like a chosen opt-out.")
    ap.add_argument("--use-msa-server", action="store_true",
                    help="fetch a ColabFold MSA per sequence (api.colabfold.com) instead of MSA-free "
                         "(the project default). GPU-free NETWORK step; the fold itself is ~1.01x. NOT part "
                         "of the reproducibility identity (server DBs change) — for the MSA-vs-noMSA experiment (dev plan/96).")
    ap.add_argument("--out-dir", default=str(_OUTPUTS_ROOT / "_incoming" / "threeway_designability"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from spa.eval.generate import Design, _parse_contig_motif
    from spa.eval.openfold3 import OF3Refolder
    from spa.eval.proteinmpnn import inverse_fold
    from spa.eval.score import _as_struct, _ca_array, score_design, source_positions

    import os
    _p = lambda ov, env, dflt: str(ov or os.environ.get(env) or dflt)

    out_dir = Path(args.out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # Record the as-run config before spending GPU time. This driver bypasses

    # spa.eval.generate.generate(), which writes provenance for the drivers that use it,

    # so it writes its own (dev audit 2026-07-31).

    from spa.eval import provenance as _prov

    _prov.write(out_dir, None, prompts=_prov_prompts(args),

                purpose='score already-generated three-way designs through ProteinMPNN and OpenFold3',

                scope='DESIGNABILITY ONLY; consumes existing backbones, generates none',

                extra=vars(args), started=_prov._now_pacific())

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
                            "seed": int(args.proteinmpnn_seed),   # FIXED nonzero (0 = random in ProteinMPNN)
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
    # Stage 3 — OpenFold3 refold (separate env, one model-load for the whole matrix).
    # OF3 batching (mirrors scripts/eval/bench_of3_batch.py): at bs>1 fold B seqs per forward for a ~2.5x
    # speedup (dev 23 §7.8). The triton kernels CANNOT batch (evoformer.py:915 asserts bias batch dim = 1),
    # so bs>1 FORCES the nokernel base yaml (regardless of --of3-runner-yaml / $OF3_RUNNER_YAML, which may be
    # triton on the cloud) + injects data_module_args.batch_size=B + applies the of3_batch_patch.py shim.
    B = int(args.of3_batch_size)
    of3_runner_yaml = cfg.paths.openfold3_runner_yaml
    batch_shim = None
    if B > 1:
        nokernel = Path(__file__).resolve().parents[2] / "configs/of3/of3_nokernel.yml"
        passed = Path(of3_runner_yaml).name
        if passed != "of3_nokernel.yml":
            print(f"[desig] batch_size={B}: forcing nokernel base ({nokernel.name}) — triton can't batch "
                  f"(was {passed}).")
        base = OmegaConf.load(str(nokernel))
        y = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        y["data_module_args"] = {**dict(y.get("data_module_args") or {}), "batch_size": B}
        yf = out_dir / f"of3_runner_bs{B}.yml"; OmegaConf.save(y, yf)   # persisted under out-dir -> captured by cp -r
        of3_runner_yaml = str(yf)
        batch_shim = str(Path(__file__).resolve().parent / "of3_batch_patch.py")  # sibling; portable local + /opt/spa
        print(f"[desig] OF3 batching ON: bs={B} via {yf.name} + shim {Path(batch_shim).name} "
              f"(~2.5x at bs=8, folds equivalent — dev 23 §7.8)")
    refolder = OF3Refolder(ckpt_path=cfg.paths.openfold3_ckpt, runner_yaml=of3_runner_yaml,
                           out_dir=str(out_dir / "of3"), conda_env=args.of3_conda_env,
                           batch_patch_shim=batch_shim, deterministic=bool(args.deterministic),
                           use_msa_server=bool(args.use_msa_server))
    # ⛔ INDEX, NOT NAME. refold_all returns {design_name: [cifs]}, and a design's NAME IS NOT UNIQUE:
    # this driver is routinely handed 16 backbones all called `free_free_0.pdb` (one per seed), so
    # `out = {nm: [] for nm in names}` collapses them to ONE key and pours every design's refolds into
    # it. Joining on the stem then scores each design against ALL of them, i.e. against other seeds'
    # designs. ⚠️ LATENT, NOT YET OBSERVED TO CHANGE AN ANSWER: on the 2026-09-16 A2 run (32 designs, 2
    # distinct stems) the name-merged scoring and the correct by-index scoring agree cell for cell,
    # because a design's own refolds already supplied its minimum. An earlier version of this comment
    # claimed a measured inversion; that was an artifact of an offline re-score that mis-reproduced the
    # driver's GNU-sort ordering, and it is withdrawn (dev results/59 §6). The defect is still real: the
    # merge is silent, and a min over a superset can only move one way, so it can inflate a rate
    # whenever some other design's refold happens to fit better than the design's own.
    # ⭐ The refold for (design i, sequence j) is at of3_batch/d{i}_q{j} by construction, so the index
    # is authoritative and needs no names at all.
    pairs = [(d, ss) for d, ss in zip(designs, seqsets) if ss is not None]
    refolds_by_name = refolder.refold_all([ss for _, ss in pairs])
    _batch_dir = pathlib.Path(refolder.out_dir) / "of3_batch"
    _by_index = {}
    for _i, (_d, _ss) in enumerate(pairs):
        _cifs = []
        for _j in range(int(args.num_seqs)):
            _c = refolder._cif_path(_batch_dir, f"d{_i}_q{_j}")
            if pathlib.Path(_c).exists():
                _cifs.append(str(_c))
        _by_index[id(_d)] = _cifs
    _names = [getattr(ss, "name", None) for _, ss in pairs]
    if len(set(_names)) != len(_names):
        print(f"[desig] ⚠️ {len(_names) - len(set(_names))} design name collision(s) "
              f"({len(set(_names))} distinct names for {len(_names)} designs). Refolds are joined BY "
              f"INDEX, so scoring is correct; the name-keyed dict from refold_all is unusable here.")

    # Stage 4 — score (designability scRMSD + refold-side motif survival)
    print(f"\n{'design':<30}{'scRMSD(Å)':>10}{'designable':>12}{'pLDDT':>8}{'motifRMSD_design':>18}{'motifRMSD_refold':>18}")
    rows = []
    for d in designs:
        refolds = _by_index.get(id(d))
        # ⛔ Truncate on the PARSED q index, never by slicing the list. refold_all appends
        # d{i}_q{j} in j order but SKIPS a cif that failed to appear, so a hole makes position
        # and sequence index diverge and refolds[:k] would quietly admit q8 while q3 is missing.
        if args.max_seqs is not None and refolds:
            _keep = []
            for _p in refolds:
                _m = re.search(r"_q(\d+)[/_]", str(_p))
                if _m and int(_m.group(1)) < args.max_seqs:
                    _keep.append(_p)
            refolds = _keep
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
