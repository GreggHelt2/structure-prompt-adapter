"""Feather × designability experiment driver (dev docs/plan/25 §5).

For ONE three-way cell (motif:seg:fold:layout:λ), sweep the feather WIDTH and test whether tapering U's
strained internal seams recovers foldability at fixed central steering. Per width:
  1. regenerate the cell's K designs with that feathered λ-profile  (probe_hard_soft_free.run_grid),
  2. ProteinMPNN (N seqs, fixed seed) → OpenFold3 refold (nokernel bs>1) → designability,
     motif-survival  (reuse scripts/eval/score_threeway_designability.py — the batching-wired scorer),
  3. adherence: U→G TM on the CORE sub-window (U interior, EXCLUDING the feathered edge residues) AND
     full-U, plus C-drag / net-steer  (core TM recomputed here; full-U/drag/net from run_grid).
Persists the full artifact tree per (cell,width) under <out>/<cellkey>_w<width>/  (design PDBs + result.json
from run_grid; FASTAs + OF3 CIFs + designability.json from the scorer; + summary.json with every metric +
the λ-profile used). width 0 == the boxcar baseline.

    conda run -n spa-dev python scripts/eval/feather_experiment.py \
        --cell A0A7C9GW19:A30-50:A0A7S3EB45:CAB:3 --feather-widths 0,9,19,38 \
        --num-seqs 16 --proteinmpnn-seed 42 --of3-batch-size 8 --out-dir outputs/eval/feather
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

# Run-artifact root — absolute + env-overridable, mirroring configs/paths/default.yaml's
# `outputs_root: ${oc.env:SPA_OUTPUTS_ROOT,${paths.project_root}/outputs}`. A *relative* default
# resolved against the invoking cwd and sent output into whichever repo the script was launched
# from; a *shared* default made runs overwrite each other. See dev docs/plan/30 §6.
_OUTPUTS_ROOT = Path(os.environ.get(
    "SPA_OUTPUTS_ROOT",
    Path(os.environ.get("SPA_PROJECT_ROOT", Path.home() / "projects" / "spa")) / "outputs"))

from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_hard_soft_free import (DEFAULT_CKPT, DEFAULT_PDB_DIR, _ca, _internal_u_edges, _profile,
                                  _resolve_pdb, _slice_tm, build_contig, run_grid)

SCORER = Path(__file__).resolve().parent / "score_threeway_designability.py"


def _run_designability(pdbs, contig, motif_pdb, out_dir, args):
    """Shell out to the (batching-wired) designability scorer on the regenerated PDBs; return its rows."""
    cmd = [sys.executable, str(SCORER), "--pdbs", *[str(p) for p in pdbs], "--contig", contig,
           "--motif-source", str(motif_pdb), "--num-seqs", str(args.num_seqs),
           "--proteinmpnn-seed", str(args.proteinmpnn_seed), "--of3-batch-size", str(args.of3_batch_size),
           "--out-dir", str(out_dir)]
    for flag, val in (("--proteinmpnn-repo", args.proteinmpnn_repo), ("--of3-ckpt", args.of3_ckpt),
                      ("--of3-runner-yaml", args.of3_runner_yaml), ("--of3-conda-env", args.of3_conda_env)):
        if val:
            cmd += [flag, str(val)]
    print(f"[feather]   designability -> {out_dir}", flush=True)
    proc = subprocess.run(cmd)
    dj = Path(out_dir) / "designability.json"
    if proc.returncode != 0 or not dj.exists():
        print(f"[feather]   ⚠️ designability FAILED (exit {proc.returncode}); no {dj.name}")
        return None
    return json.loads(dj.read_text())


def _core_window(u_lo, u_hi, L, width):
    """The U interior EXCLUDING the feathered edge residues (only internal seams are feathered)."""
    left_int, right_int = _internal_u_edges(u_lo, u_hi, L)
    effw = min(int(width), u_hi - u_lo + 1)
    lo = u_lo + (effw if (left_int and width > 0) else 0)
    hi = u_hi - (effw if (right_int and width > 0) else 0)          # inclusive
    return lo, hi, (left_int, right_int, effw)


def run_width(args, width):
    mid, seg, fold, layout, lam = args._cell
    cellkey = f"{mid}_{seg}_{fold}_{layout}_l{lam:g}"
    wdir = Path(args.out_dir).expanduser().resolve() / f"{cellkey}_w{width}"
    wdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[feather] ===== cell {cellkey}  width={width}  shape={args.feather_shape} =====", flush=True)

    # (1) regenerate the K designs with this feathered profile (single layout, single λ)
    gargs = SimpleNamespace(
        ckpt=args.ckpt, rfd3_ckpt=args.rfd3_ckpt, motif_source=mid, motif_seg=seg, target=fold,
        u_len=args.u_len, c_len=args.c_len, layout=layout, layouts=layout, lambda_scale=lam,
        lambdas=f"{lam:g}", num_designs=args.num_designs, seed=args.seed, num_timesteps=args.num_timesteps,
        pdb_dir=args.pdb_dir, device=args.device, out_dir=str(wdir),
        feather_width=int(width), feather_shape=args.feather_shape)
    grid, _ = run_grid(gargs)
    lo = grid[0]                                                    # single layout
    u_lo, u_hi_excl = lo["U"]; u_hi = u_hi_excl - 1; L = lo["L"]   # lo["U"] hi is EXCLUSIVE (_contiguous) -> inclusive last U residue
    adh = lo["lambdas"][f"{lam:g}"]                                 # tm_U_loc / U_steer / tm_C_loc / C_drag / net_steer / delta_motif_rmsd

    pdbs = sorted(wdir.glob(f"*/localized_l{lam:g}_*.pdb"))
    if not pdbs:
        print(f"[feather]   ⚠️ no localized PDBs under {wdir} — skipping width {width}"); return None

    # (2) designability + motif survival (reuse the scorer; nokernel bs=of3_batch_size)
    contig = build_contig(seg, args.u_len, args.c_len, layout)[0]
    motif_pdb = _resolve_pdb(mid, args.pdb_dir)
    rows = _run_designability(pdbs, contig, motif_pdb, wdir / "desig", args)

    # (3) CORE-window U→G TM (exclude feathered edges) — recomputed from the design PDBs vs G
    clo, chi, (left_int, right_int, effw) = _core_window(u_lo, u_hi, L, width)
    target_ca = _ca(_resolve_pdb(fold, args.pdb_dir))
    core_tms = [_slice_tm(_ca(str(p)), clo, chi + 1, target_ca) for p in pdbs]
    core_tm = sum(core_tms) / len(core_tms) if core_tms else None

    profile = _profile(L, list(range(u_lo, u_hi + 1)), "cpu", feather_width=int(width), shape=args.feather_shape)
    pv = [round(float(x), 3) for x in profile.tolist()]
    n_des = sum(1 for r in (rows or []) if r.get("designable"))
    best = min((r["scrmsd"] for r in (rows or []) if r.get("scrmsd") is not None), default=None)
    motif_ref = [r["motif_rmsd_refold"] for r in (rows or []) if r.get("motif_rmsd_refold") is not None]

    summary = {
        "cell": cellkey, "motif": mid, "seg": seg, "fold": fold, "layout": layout, "lambda": lam,
        "feather_width": int(width), "feather_shape": args.feather_shape,
        "U": [u_lo, u_hi], "L": L, "core_window": [clo, chi],
        "internal_edges": {"left": left_int, "right": right_int, "eff_width": effw},
        "profile_u_window": pv[u_lo:u_hi + 1],
        "adherence": {"full_U_tm": adh.get("tm_U_loc"), "core_U_tm": core_tm, "U_steer": adh.get("U_steer"),
                      "C_drag": adh.get("C_drag"), "net_steer": adh.get("net_steer"),
                      "delta_motif_rmsd": adh.get("delta_motif_rmsd")},
        "designability": {"per_design": rows, "n_designs": len(rows or []),
                          "designable_rate": f"{n_des}/{len(rows or [])}" if rows else None,
                          "best_scrmsd": best},
        "motif_survival": {"motif_rmsd_refold_mean": (sum(motif_ref) / len(motif_ref)) if motif_ref else None},
    }
    (wdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[feather]   width {width}: full-U TM {adh.get('tm_U_loc')}  core-U TM "
          f"{core_tm:.3f}  net {adh.get('net_steer')}  designable {summary['designability']['designable_rate']}  "
          f"best scRMSD {best}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="motif:seg:fold:layout:lambda, e.g. A0A7C9GW19:A30-50:A0A7S3EB45:CAB:3")
    ap.add_argument("--feather-widths", default="0,9,19,38", help="comma list of feather widths (residues); 0 = boxcar")
    ap.add_argument("--feather-shape", default="cosine", choices=["cosine", "triangular", "gaussian"])
    ap.add_argument("--num-seqs", type=int, default=16, help="ProteinMPNN seqs per design (best-of-N)")
    ap.add_argument("--proteinmpnn-seed", type=int, default=42)
    ap.add_argument("--of3-batch-size", type=int, default=8, help="OF3 refold batch_size (nokernel; ~2.5x@8; dev 23 §7.8)")
    ap.add_argument("--num-designs", type=int, default=8, help="K designs per width (paired noise)")
    ap.add_argument("--u-len", type=int, default=90)
    ap.add_argument("--c-len", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--rfd3-ckpt", default=None)
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(_OUTPUTS_ROOT / "_incoming" / "feather"))
    # scorer portability pass-through (else its $ENV/local defaults)
    ap.add_argument("--proteinmpnn-repo", default=None)
    ap.add_argument("--of3-ckpt", default=None)
    ap.add_argument("--of3-runner-yaml", default=None)
    ap.add_argument("--of3-conda-env", default="spa-verify-of3")
    args = ap.parse_args()

    parts = args.cell.split(":")
    if len(parts) != 5:
        ap.error("--cell must be motif:seg:fold:layout:lambda (5 colon-separated fields)")
    args._cell = (parts[0], parts[1], parts[2], parts[3].upper(), float(parts[4]))
    widths = [int(x) for x in args.feather_widths.split(",") if x.strip() != ""]
    print(f"[feather] cell={args.cell}  widths={widths}  K={args.num_designs}  N={args.num_seqs}  "
          f"of3_bs={args.of3_batch_size}  shape={args.feather_shape}")

    summaries = [s for s in (run_width(args, w) for w in widths) if s]
    out = Path(args.out_dir).expanduser().resolve()
    (out / f"{args._cell[0]}_{args._cell[1]}_{args._cell[2]}_{args._cell[3]}_l{args._cell[4]:g}_feather_sweep.json"
     ).write_text(json.dumps(summaries, indent=2, default=str))
    print("\n[feather] ===== SWEEP SUMMARY =====")
    print(f"{'width':>6}{'full-U TM':>11}{'core-U TM':>11}{'net-steer':>11}{'designable':>12}{'best scRMSD':>13}")
    for s in summaries:
        a, d = s["adherence"], s["designability"]
        f = lambda x: "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        print(f"{s['feather_width']:>6}{f(a['full_U_tm']):>11}{f(a['core_U_tm']):>11}{f(a['net_steer']):>11}"
              f"{str(d['designable_rate']):>12}{f(d['best_scrmsd']):>13}")
    print("[read] feather WINS if core-U TM stays high while designable-rate/best-scRMSD improve vs width 0 (dev 25 §5.4)")


if __name__ == "__main__":
    main()
