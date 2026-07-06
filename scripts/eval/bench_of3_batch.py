"""Measure OF3 BATCHED-inference speedup + validate equivalence (dev 23; subagent finding: OF3's inference
is already batch-aware — `data_module_args.batch_size>1` in the runner-yaml folds same-length queries in
ONE forward, no fork). Fold N same-length ProteinMPNN sequences (one backbone) at batch_size ∈ {1,8,16};
report wall-clock speedup vs bs=1 AND the scRMSD-to-design distribution per bs (should be equivalent IN
AGGREGATE — best-of-K + designable-rate — though NOT bit-identical per sample; see of3_batch_patch.py
"Equivalence"). NB: same-LENGTH seqs are still RAGGED at the ATOM level (sidechains differ), which is
exactly the case of3_batch_patch.py exists to handle — bs>1 needs that shim (auto-applied for B>1).

    conda run -n spa-dev python scripts/eval/bench_of3_batch.py --design <pdb> --n-seqs 16 --batch-sizes 1,8,16
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

ROOT = Path("/home/user1/projects/spa")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True, help="a backbone PDB to inverse-fold then refold at each batch size")
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--batch-sizes", default="1,8,16")
    ap.add_argument("--of3-ckpt", default=None)
    ap.add_argument("--of3-runner-yaml", default=None, help="BASE runner-yaml (batch_size gets injected per run)")
    ap.add_argument("--proteinmpnn-repo", default=None)
    ap.add_argument("--of3-conda-env", default="spa-verify-of3")
    ap.add_argument("--batch-patch-shim", default=None,
                    help="of3_batch_patch.py (applied only for bs>1; default $OF3_BATCH_SHIM or repo path)")
    ap.add_argument("--out-dir", default=str(ROOT / "structure-prompt-adapter/outputs/eval/of3_batch_bench"))
    args = ap.parse_args()

    import biotite.structure as struc
    from omegaconf import OmegaConf
    from spa.eval.generate import Design
    from spa.eval.openfold3 import OF3Refolder
    from spa.eval.proteinmpnn import inverse_fold
    from spa.eval.score import _as_struct, _ca_array

    _p = lambda ov, env, d: str(ov or os.environ.get(env) or d)
    out = Path(args.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    base_yaml = _p(args.of3_runner_yaml, "OF3_RUNNER_YAML", ROOT / "structure-prompt-adapter/configs/of3/of3_nokernel.yml")
    of3_ckpt = _p(args.of3_ckpt, "OF3_CKPT", ROOT / "models/openfold3/of3-p2-155k.pt")

    cfg = OmegaConf.create({"paths": {"proteinmpnn_repo": _p(args.proteinmpnn_repo, "PROTEINMPNN_REPO", ROOT / "needed_repos/ProteinMPNN")},
        "eval": {"proteinmpnn": {"num_seqs": int(args.n_seqs), "sampling_temp": 0.1, "batch_size": 1, "seed": 42,
                                 "model_name": "v_48_020", "weights_dir": None, "designs": None, "design_dir": None,
                                 "out_dir": str(out / "seqs")}}})
    d = Path(args.design)
    dstruct = _as_struct(str(d)); dca = _ca_array(dstruct)
    design = Design(prompt_id=d.stem, condition="bench", lambda_scale=0.0, idx=0, path=d,
                    n_residues=len(dca), atom_array=dstruct)
    ss = inverse_fold(cfg, designs=[design])[0]
    n = len(ss.sequences)
    print(f"[batch-bench] {n} seqs of len {len(ss.sequences[0])} for {d.stem} (len {len(dca)})", flush=True)

    def scrmsd(cif):
        r = _ca_array(_as_struct(cif)); m = min(len(dca), len(r))
        fit, _ = struc.superimpose(dca[:m], r[:m]); return float(struc.rmsd(dca[:m], fit))

    base = OmegaConf.load(base_yaml)
    rows = []
    for B in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        y = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        y["data_module_args"] = {**dict(y.get("data_module_args") or {}), "batch_size": int(B)}
        yf = out / f"runner_bs{B}.yml"; OmegaConf.save(y, yf)
        bp = (_p(args.batch_patch_shim, "OF3_BATCH_SHIM",
                 ROOT / "structure-prompt-adapter/scripts/eval/of3_batch_patch.py")) if B > 1 else None
        rf = OF3Refolder(ckpt_path=of3_ckpt, runner_yaml=str(yf), out_dir=str(out / f"bs{B}"),
                         conda_env=args.of3_conda_env, batch_patch_shim=bp)
        t0 = time.time(); folds = rf.refold_all([ss]); dt = time.time() - t0
        cifs = folds.get(ss.name, [])
        sc = sorted(scrmsd(c) for c in cifs)
        rows.append({"B": B, "t": dt, "nfold": len(cifs),
                     "scrmsd_min": sc[0] if sc else None, "scrmsd_med": sc[len(sc) // 2] if sc else None})
        print(f"[batch-bench] bs={B:>3}: {len(cifs)}/{n} folds in {dt:6.1f}s  ({len(cifs)/dt:.2f} folds/s)  "
              f"scRMSD min {sc[0]:.2f} med {sc[len(sc)//2]:.2f}" if sc else f"bs={B}: no folds", flush=True)

    t1 = rows[0]["t"]
    print(f"\n{'batch_size':>10}{'wall(s)':>9}{'folds/s':>9}{'speedup':>9}{'scRMSD_min':>11}{'scRMSD_med':>11}")
    for r in rows:
        f = lambda x: "n/a" if x is None else f"{x:.2f}"
        print(f"{r['B']:>10}{r['t']:>9.1f}{r['nfold']/r['t']:>9.2f}{t1/r['t']:>8.2f}x{f(r['scrmsd_min']):>11}{f(r['scrmsd_med']):>11}")
    print("[read] speedup vs bs=1 = throughput gain; scRMSD dists should MATCH across bs (⇒ batching is equivalent).")


if __name__ == "__main__":
    main()
