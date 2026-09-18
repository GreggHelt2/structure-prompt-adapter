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
        --num-seqs 16 --proteinmpnn-seed 42 --out-dir outputs/eval/feather

⛔ **THREE THINGS CHANGED 2026-09-18 AND THE SCRIPT WOULD NOT RUN WITHOUT THEM.** Recorded here because
each was invisible at the call site:

1. **`--of3-batch-size` now defaults to 1, not 8.** OF3 determinism is incompatible with bs>1 (batched
   refolds are not bit-reproducible per sample), and since the determinism defaults were flipped on
   2026-09-18 the scorer raises when handed both. The old default made this script abort on launch.
2. **The generation args now carry `deterministic`.** `run_grid` builds its cfg with
   `bool(getattr(args, "deterministic", False))` (`probe_hard_soft_free.py:390`), so an ABSENT
   attribute writes an explicit `False` that OVERRIDES the config default. This script passed an
   in-process `SimpleNamespace` with no such field, so it would have generated NON-deterministically
   while every config read said otherwise. That is exactly how queue row 25 came to be labelled
   "contract v2" while refolding on stock OpenFold3.
3. **K is pinned at 1 and the draws come from `--seeds`.** The 2026-07-06 run used K=8 at one seed,
   which is 8 rows of ONE diffusion batch; batch SHAPE selects cuBLAS kernels, so K=8 designs are not
   comparable with K=1 ones (dev plan/100).
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
           # ⛔ PASSED EXPLICITLY, both ways. The scorer's flag is BooleanOptionalAction, so silence
           # would take ITS default rather than this script's, and the two could drift apart unnoticed.
           ("--deterministic" if args.deterministic else "--no-deterministic"),
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
    """The U interior EXCLUDING the feathered edge residues (only internal seams are feathered).

    ⛔ **The clamp is per-SIDE, and that matters.** ``u_hi`` is inclusive, so |U| = ``u_hi - u_lo + 1``.
    When BOTH seams are internal the width is subtracted from both ends, so clamping ``effw`` to |U|
    (as this did until 2026-09-16) still lets ``2*effw`` exceed |U| and returns an INVERTED window:
    at |U| = 90 a width of 45 gave ``[64, 63]`` (empty) and 60 gave ``[79, 48]`` (negative). The
    caller then slices ``design_ca[clo:chi + 1]`` and ``tmtools`` raises *"Sequence is too short <3!"*,
    **after** the designs and the whole refold pass for that width have already been paid for.
    Clamping to ``(|U| - 1) // 2`` per side keeps at least one core residue. Audit: dev ``111``
    §15.1 A1.
    """
    left_int, right_int = _internal_u_edges(u_lo, u_hi, L)
    u_len = u_hi - u_lo + 1
    n_sides = int(bool(left_int)) + int(bool(right_int))
    cap = u_len if n_sides < 2 else max(0, (u_len - 1) // 2)
    effw = min(int(width), cap)
    lo = u_lo + (effw if (left_int and width > 0) else 0)
    hi = u_hi - (effw if (right_int and width > 0) else 0)          # inclusive
    if hi < lo:                                                     # unreachable given the clamp; loud if ever
        raise ValueError(f"_core_window: inverted core [{lo},{hi}] for U=[{u_lo},{u_hi}] width={width}")
    return lo, hi, (left_int, right_int, effw)


def run_width(args, width):
    mid, seg, fold, layout, lam = args._cell
    cellkey = f"{mid}_{seg}_{fold}_{layout}_l{lam:g}"
    wdir = Path(args.out_dir).expanduser().resolve() / f"{cellkey}_w{width}"
    wdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[feather] ===== cell {cellkey}  width={width}  shape={args.feather_shape} =====", flush=True)

    # (1) regenerate this width's designs, ONE PER SEED at K=1 (see the module docstring, item 3).
    per_seed, geom = [], None
    for seed in args.seeds:
        gargs = SimpleNamespace(
            ckpt=args.ckpt, rfd3_ckpt=args.rfd3_ckpt, motif_source=mid, motif_seg=seg, target=fold,
            u_len=args.u_len, c_len=args.c_len, layout=layout, layouts=layout, lambda_scale=lam,
            lambdas=f"{lam:g}", num_designs=int(args.num_designs), seed=int(seed),
            num_timesteps=args.num_timesteps, pdb_dir=args.pdb_dir, device=args.device,
            out_dir=str(wdir / f"s{seed}"),
            # ⛔ THE FIELD WHOSE ABSENCE MADE THIS SILENTLY NON-DETERMINISTIC. See docstring item 2.
            deterministic=bool(args.deterministic),
            feather_width=int(width), feather_shape=args.feather_shape)
        grid, _ = run_grid(gargs)
        lo = grid[0]                                                # single layout
        # lo["U"] hi is EXCLUSIVE (_contiguous) -> inclusive last U residue
        g = (lo["U"][0], lo["U"][1] - 1, lo["L"])
        if geom is None:
            geom = g
        elif g != geom:
            raise ValueError(f"geometry moved between seeds: {g} against {geom}. Every seed of one "
                             "width must share U and L or the pooled core window is meaningless.")
        per_seed.append({"seed": int(seed), **lo["lambdas"][f"{lam:g}"]})
    u_lo, u_hi, L = geom

    def _mean(key):
        vals = [d[key] for d in per_seed if d.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None
    # tm_U_loc / U_steer / tm_C_loc / C_drag / net_steer / delta_motif_rmsd, averaged over the seeds.
    adh = {k: _mean(k) for k in ("tm_U_loc", "U_steer", "tm_C_loc", "C_drag", "net_steer",
                                 "delta_motif_rmsd")}

    # ⛔ RECURSIVE, deliberately. Each seed now gets its own run_grid out_dir, so the designs sit one
    # level DEEPER than before (wdir/s<seed>/<cell>/). The previous non-recursive glob would match
    # nothing here and the width would be skipped with a warning rather than failing.
    pdbs = sorted(wdir.rglob(f"localized_l{lam:g}_*.pdb"))
    expected = len(args.seeds) * int(args.num_designs)
    if not pdbs:
        print(f"[feather]   ⚠️ no localized PDBs under {wdir}; skipping width {width}"); return None
    if len(pdbs) != expected:
        raise ValueError(f"width {width}: {len(pdbs)} designs on disk, expected {expected}. "
                         "An incomplete width must not be scored as if it were whole.")

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
    # ⭐ A FEATHER THAT DID NOTHING IS THE ONE FAILURE THIS EXPERIMENT CANNOT ABSORB: it would report
    # "feathering changes nothing" for a width that was never applied, which reads as a null and is a
    # NO-OP. Provenance proves what was requested, never what took effect (root CLAUDE.md, amended
    # 2026-09-15), so compare the actual profile against the boxcar.
    if int(width) > 0:
        boxcar = _profile(L, list(range(u_lo, u_hi + 1)), "cpu", feather_width=0, shape=args.feather_shape)
        if [round(float(x), 3) for x in boxcar.tolist()] == pv:
            raise ValueError(f"width {width} produced a profile IDENTICAL to the boxcar: the feather is "
                             "a no-op, and a no-op is not a null.")
    n_des = sum(1 for r in (rows or []) if r.get("designable"))
    best = min((r["scrmsd"] for r in (rows or []) if r.get("scrmsd") is not None), default=None)
    motif_ref = [r["motif_rmsd_refold"] for r in (rows or []) if r.get("motif_rmsd_refold") is not None]

    summary = {
        "cell": cellkey, "motif": mid, "seg": seg, "fold": fold, "layout": layout, "lambda": lam,
        "feather_width": int(width), "feather_shape": args.feather_shape,
        # ⛔ THE SAMPLE'S SHAPE IS PART OF THE RESULT. K and the seed list together define draws/width,
        # and K is part of the reproducibility identity, so a summary that records only a rate cannot
        # be compared against another run later (dev plan/100 §9).
        "seeds": list(args.seeds), "K": int(args.num_designs),
        "draws": len(args.seeds) * int(args.num_designs),
        "deterministic": bool(args.deterministic),
        # ⭐ Per-seed adherence is KEPT, not just its mean: the mean over 8 seeds hides the spread, and
        # a spread is exactly what decides whether a width difference is real (dev results/65 §3).
        "per_seed_adherence": per_seed,
        # ⛔ INCLUSIVE bounds, unlike `result.json["U"]` which is HALF-OPEN (`_contiguous` returns
        # `idxs[-1] + 1`). The two files carried the same key name under different conventions until
        # 2026-09-16; the keys are now suffixed so they cannot be read as interchangeable. Every other
        # span in this project is half-open. Audit: dev `111` §15.1 A2.
        "U_inclusive": [u_lo, u_hi], "L": L, "core_window_inclusive": [clo, chi],
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
    ap.add_argument("--of3-batch-size", type=int, default=1,
                    help="OF3 refold batch_size. ⛔ DEFAULT CHANGED 8 -> 1 on 2026-09-18: bs>1 is "
                         "incompatible with OF3 determinism and the scorer now RAISES when given both. "
                         "Set >1 only together with --no-deterministic, for a throughput run.")
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                    help="bitwise determinism at BOTH stages, default ON since 2026-09-18. Passed into "
                         "run_grid (whose absence-means-False getattr would otherwise override the "
                         "config default) and forwarded to the scorer.")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7",
                    help="comma list of generation seeds. K=1 per seed, so this IS the draw count.")
    ap.add_argument("--num-designs", type=int, default=1,
                    help="K, the diffusion batch. ⛔ PINNED AT 1 (dev plan/100): K>1 is one batch whose "
                         "SHAPE selects cuBLAS kernels, so its designs are not comparable with K=1 ones.")
    ap.add_argument("--u-len", type=int, default=90)
    ap.add_argument("--c-len", type=int, default=120)
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
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and VALIDATE everything, print the plan, then exit before any GPU "
                         "work. ⛔ Creates no directories: it returns from main() before run_width, "
                         "which is where the only mkdir lives.")
    args = ap.parse_args()

    parts = args.cell.split(":")
    if len(parts) != 5:
        ap.error("--cell must be motif:seg:fold:layout:lambda (5 colon-separated fields)")
    args._cell = (parts[0], parts[1], parts[2], parts[3].upper(), float(parts[4]))
    widths = [int(x) for x in args.feather_widths.split(",") if x.strip() != ""]
    # ⛔ VALIDATE WIDTHS BEFORE SPENDING ANY GPU. Both U seams are feathered when both are internal,
    # so a width at or above half of |U| leaves a core too short to TM-score, and the failure would
    # otherwise land in step (3) AFTER this width's designs and its whole refold pass are paid for,
    # then abort the remaining widths. `_core_window` clamps so the window can never invert; this is
    # the loud, early version of the same guard. Audit: dev `111` §15.1 A1.
    _min_core = 3                                              # tmtools raises below 3 residues
    _too_wide = [w for w in widths if w > 0 and (args.u_len - 2 * min(w, max(0, (args.u_len - 1) // 2))) < _min_core]
    if _too_wide:
        ap.error(f"--feather-widths {_too_wide} leave fewer than {_min_core} core residues in a "
                 f"U of {args.u_len} (both seams are feathered, so each width is subtracted twice). "
                 f"Use widths below {(args.u_len - _min_core) // 2 + 1}.")
    # ⛔ argparse hands --seeds over as a STRING. `for seed in args.seeds` would then walk CHARACTERS,
    # so "0,1,2" becomes seeds '0', ',', '1'. Parse it before anything touches the GPU.
    args.seeds = [int(x) for x in str(args.seeds).split(",") if x.strip() != ""]
    if not args.seeds:
        ap.error("--seeds parsed to an empty list; at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error(f"--seeds contains duplicates: {args.seeds}. At K=1 the draw count is the number of "
                 "DISTINCT seeds, so a repeat silently shrinks the real sample while the count does not.")
    if int(args.num_designs) != 1:
        print(f"[feather] ⚠️ K={args.num_designs}, not 1. K>1 is ONE diffusion batch whose SHAPE selects "
              "cuBLAS kernels, so these designs are not comparable with K=1 runs (dev plan/100).", flush=True)
    if int(args.of3_batch_size) > 1 and args.deterministic:
        ap.error(f"--of3-batch-size {args.of3_batch_size} with determinism ON: batched refolds are not "
                 "bit-reproducible per sample, and the scorer refuses the combination. Use bs=1, or pass "
                 "--no-deterministic for a deliberate throughput run and say so in the plan/106 row.")
    print(f"[feather] cell={args.cell}  widths={widths}  K={args.num_designs}  seeds={args.seeds}  "
          f"draws/width={len(args.seeds) * int(args.num_designs)}  N={args.num_seqs}  "
          f"of3_bs={args.of3_batch_size}  deterministic={args.deterministic}  shape={args.feather_shape}")

    # ⭐ The dry run exists because every guard above FAILS loudly, so testing them proves only that the
    # error paths work. The happy path, in particular that --seeds parsed to a LIST of ints rather than
    # being iterated as a string, is otherwise unverifiable without spending GPU.
    if args.dry_run:
        draws = len(args.seeds) * int(args.num_designs)
        for w in widths:
            print(f"[feather] DRY width {w}: {len(args.seeds)} seed(s) x K={args.num_designs} = {draws} "
                  f"design(s), then ONE scorer call at N={args.num_seqs} "
                  f"({draws * int(args.num_seqs)} refolds)")
        tot = len(widths) * draws
        print(f"[feather] DRY total: {tot} designs, {tot * int(args.num_seqs)} refolds across "
              f"{len(widths)} width(s). Nothing generated, nothing scored, no directories created.")
        return

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
