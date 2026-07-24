"""Benchmark OF3 refold throughput vs concurrency — does running P concurrent `run_openfold` processes
(GPU-sharing) beat the current sequential-per-fold behavior (openfold3.py refold_all folds one at a time)?
OF3 is the sweep bottleneck (~70-80%); the H100-vs-A5000 speedup was only ~2× (vs ~10× raw compute),
implying OF3 underutilizes the GPU → concurrency should fill the idle headroom (dev 21 §4.1 / cloud sweep).

Folds N ProteinMPNN sequences (for one backbone) at each parallelism P: shard the N queries into P groups,
spawn P `run_openfold` subprocesses at once, wait all, time. Reports folds/s + speedup vs P=1.

    conda run -n spa-dev python scripts/eval/bench_of3_parallel.py --design <pdb> --n-seqs 32 --parallels 1,2,4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# Run-artifact root — absolute + env-overridable, mirroring configs/paths/default.yaml's
# `outputs_root: ${oc.env:SPA_OUTPUTS_ROOT,${paths.project_root}/outputs}`. A *relative* default
# resolved against the invoking cwd and sent output into whichever repo the script was launched
# from; a *shared* default made runs overwrite each other. See dev docs/plan/30 §6.
_OUTPUTS_ROOT = Path(os.environ.get(
    "SPA_OUTPUTS_ROOT",
    Path(os.environ.get("SPA_PROJECT_ROOT", Path.home() / "projects" / "spa")) / "outputs"))


ROOT = Path("/home/user1/projects/spa")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True, help="a backbone PDB to inverse-fold (ProteinMPNN) then refold")
    ap.add_argument("--n-seqs", type=int, default=32, help="N sequences to fold (want folds >> the ~45s model-load)")
    ap.add_argument("--parallels", default="1,2,4", help="comma list of concurrency levels to time")
    ap.add_argument("--proteinmpnn-repo", default=None)
    ap.add_argument("--of3-ckpt", default=None)
    ap.add_argument("--of3-runner-yaml", default=None)
    ap.add_argument("--of3-conda-env", default="spa-verify-of3")
    ap.add_argument("--out-dir", default=str(_OUTPUTS_ROOT / "_incoming" / "of3_bench"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from spa.eval.generate import Design
    from spa.eval.openfold3 import OF3Refolder
    from spa.eval.proteinmpnn import inverse_fold
    from spa.eval.score import _as_struct, _ca_array

    _p = lambda ov, env, d: str(ov or os.environ.get(env) or d)
    out_dir = Path(args.out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({"paths": {"proteinmpnn_repo": _p(args.proteinmpnn_repo, "PROTEINMPNN_REPO",
                                                             ROOT / "needed_repos/ProteinMPNN")},
        "eval": {"proteinmpnn": {"num_seqs": int(args.n_seqs), "sampling_temp": 0.1, "batch_size": 1,
                                 "seed": 42, "model_name": "v_48_020", "weights_dir": None,
                                 "designs": None, "design_dir": None, "out_dir": str(out_dir / "seqs")}}})

    d = Path(args.design)
    design = Design(prompt_id=d.stem, condition="bench", lambda_scale=0.0, idx=0, path=d,
                    n_residues=len(_ca_array(_as_struct(str(d)))), atom_array=_as_struct(str(d)))
    ss = inverse_fold(cfg, designs=[design])[0]
    seqs = list(ss.sequences)[: int(args.n_seqs)]
    print(f"[bench] {len(seqs)} seqs of len {len(seqs[0])} for {d.stem}; parallels={args.parallels}")

    rf = OF3Refolder(ckpt_path=_p(args.of3_ckpt, "OF3_CKPT", ROOT / "models/openfold3/of3-p2-155k.pt"),
                     runner_yaml=_p(args.of3_runner_yaml, "OF3_RUNNER_YAML",
                                    ROOT / "structure-prompt-adapter/configs/of3/of3_nokernel.yml"),
                     out_dir=str(out_dir), conda_env=args.of3_conda_env)

    def spawn(shard_seqs, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        qj = run_dir / "queries.json"
        json.dump({"queries": {f"q{j}": rf._chain(s) for j, s in enumerate(shard_seqs)}}, open(qj, "w"))
        return subprocess.Popen(rf._build_command(qj, run_dir),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ.copy())

    base = None
    print(f"\n{'n_parallel':>10}{'wall(s)':>10}{'folds/s':>10}{'speedup':>9}")
    for P in [int(x) for x in args.parallels.split(",") if x.strip()]:
        shards = [seqs[i::P] for i in range(P)]                        # round-robin split
        t0 = time.time()
        procs = [spawn(sh, out_dir / f"p{P}_shard{i}") for i, sh in enumerate(shards) if sh]
        for pr in procs:
            pr.wait()
        dt = time.time() - t0
        fps = len(seqs) / dt
        base = base or fps
        print(f"{P:>10}{dt:>10.1f}{fps:>10.2f}{fps / base:>8.2f}x", flush=True)


if __name__ == "__main__":
    main()
