"""Two-steer composition probe — hard motif M (pinned) flanked by TWO soft SPA regions, each steered
toward a DIFFERENT foreign fold: region R1 → G1, region R2 → G2, in a SINGLE design.

This extends the three-way probe (``probe_hard_soft_free.py``: A = hard M · B = soft U · C = free) by
making **both** flanks soft-steered — to *possibly different* targets — rather than one soft + one
free. It is enabled by the **multi-prompt SPA path** (``SPAAdapter.set_prompts``: a list of
``(prompt, disjoint-profile)`` pairs; the wrapper loops-and-sums ``base + Σₖ SPA(A, Gₖ)·profileₖ``,
``src/spa/model/wrapper.py``). Because the two region profiles are **disjoint** (R1 vs R2, with the
pinned motif 0 in both), each design residue receives at most one prompt's contribution — mutual
exclusivity by construction, which we then **measure** via the cross terms below.

The chain is a fixed 3-region contig ``R1 | M | R2`` (motif always in the middle). Each cell chooses a
target for each flank, ``R1,R2 ∈ {free, G1, G2}``. The **headliner** cell is ``g1:g2`` =
``SPA-G1 : M : SPA-G2`` — two regions, two folds, one pinned motif, one design (dev: two-steer note;
three-way spec is dev ``21``). N×1536 (variant C) only — the sole variant that honors a per-residue mask.

Metrics (all paired to the ``free:free`` baseline, same seed ⇒ identical initial noise):
  - **motif-RMSD(M)**  ≈ 0 in every cell, Δ ≈ 0 ⇒ the pin holds even with two live profiles (the novel
    check — two ``set_profile`` masks + a hard motif have never run together).
  - **R1 → G1 TM steer** and **R2 → G2 TM steer** (each vs baseline) ⇒ both flanks steer.
  - **cross terms** R1 → G2 and R2 → G1 ⇒ each flank adopts *its own* assigned fold, not the other's
    (no prompt bleed across the disjoint masks — the guarantee, quantified).

Adherence-only (no ProteinMPNN / OF3) ⇒ fast + A5000-friendly. The winner cell's PDBs are written for
downstream ProteinMPNN → OF3 designability on the H100.

Run (A5000) — the two-steer matrix, seed 17:
    conda run -n spa-dev python scripts/eval/probe_two_steer.py \
        --ckpt checkpoints/spa-Nx1536-multigran/spa_C_final.pt \
        --motif-source A0A2X2KHU0 --motif-seg A2-20 \
        --g1 A0A1X7NTP0 --r1-len 90 --g2 A0A7C6QMG4 --r2-len 47 \
        --lambda 2 --num-designs 8 --seed 17 \
        --out-dir outputs/eval/twosteer_seed17/matrix
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Run-artifact root — absolute + env-overridable, mirroring configs/paths/default.yaml's
# `outputs_root: ${oc.env:SPA_OUTPUTS_ROOT,${paths.project_root}/outputs}`. A *relative* default
# resolved against the invoking cwd and sent output into whichever repo the script was launched
# from; a *shared* default made runs overwrite each other. See dev docs/plan/30 §6.
_OUTPUTS_ROOT = Path(os.environ.get(
    "SPA_OUTPUTS_ROOT",
    Path(os.environ.get("SPA_PROJECT_ROOT", Path.home() / "projects" / "spa")) / "outputs"))


# Sibling helpers — same dir is sys.path[0] when either script is run directly.
from probe_hard_soft_free import (
    DEFAULT_CKPT, DEFAULT_PDB_DIR, _ca, _contiguous, _mean, _pair_tm, _precompute_prompt,
    _profile, _resolve_pdb, _rfd3_ckpt, _seg_len, _slice_tm, build_contig, build_partition,
)

# target token → (which precomputed prompt, human label). "free" is handled separately (no prompt).
_TARGETS = ("free", "g1", "g2")


def _parse_cells(spec):
    """``"free:free,g1:g2,..."`` → ``[("free","free"), ("g1","g2"), ...]`` (t1 = R1, t2 = R2)."""
    cells = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        t1, _, t2 = tok.partition(":")
        if t1 not in _TARGETS or t2 not in _TARGETS:
            raise ValueError(f"--cells entry {tok!r}: each side must be one of {_TARGETS}")
        cells.append((t1, t2))
    if ("free", "free") not in cells:
        cells.insert(0, ("free", "free"))         # the paired baseline is mandatory; run it first
    return cells


def _spacered_partition(motif_seg, r1_len, r2_len, sp_inner, sp_term):
    """Contig ``[Ns] R1 [s] M [s] R2 [Ns]`` with RFD3-FREE spacers (s=inner, Ns=terminal). Free residues
    decouple each soft region from the pinned motif M (and, terminally, from the chain ends), giving R1/R2
    geometric slack to adopt their folds without conforming to an immediately-adjacent foreign motif (dev:
    spacer probe; motivated by the Fig-2 self-prompt vs two-steer tension, ``results/09``). Regions stay
    contiguous single blocks; spacer residues get profile 0 (RFD3 designs them freely). Returns
    ``(contig_str, M_idx, R1_idx, R2_idx, L)`` — the drop-in replacement for build_contig+build_partition
    when either spacer > 0 (both 0 ⇒ caller uses the original path, so this is regression-inert)."""
    segs = []
    if sp_term:  segs.append(("_", sp_term))          # N-terminal spacer (before R1)
    segs.append(("R1", r1_len))
    if sp_inner: segs.append(("_", sp_inner))          # R1|M spacer
    segs.append(("M", _seg_len(motif_seg)))
    if sp_inner: segs.append(("_", sp_inner))          # M|R2 spacer
    segs.append(("R2", r2_len))
    if sp_term:  segs.append(("_", sp_term))          # C-terminal spacer (after R2)
    toks, idx, cur = [], {"R1": [], "M": [], "R2": []}, 0
    for role, ln in segs:
        toks.append(str(motif_seg) if role == "M" else str(ln))
        if role in idx:
            idx[role] = list(range(cur, cur + ln))
        cur += ln
    return ",".join(toks), idx["M"], idx["R1"], idx["R2"], cur


def _score_cell(outs, edir, cname, geom, src_struct, src_positions, g1_ca, g2_ca):
    """Score one cell's K designs: motif-RMSD(M) + every region×target region-TM; keep region Cα."""
    from spa.eval.generate import write_pdb
    from spa.eval.score import _ca_array, motif_rmsd

    m_lo, m_hi, r1_lo, r1_hi, r2_lo, r2_hi, M_idx = geom
    rows = {}
    for idx, o in enumerate(outs):
        aa = o.atom_array
        write_pdb(aa, edir / f"{cname}_{idx}.pdb")
        dca = _ca_array(aa)
        rows[idx] = {
            "motif_rmsd": motif_rmsd(aa, src_struct, M_idx, source_residues=src_positions),
            "tm_R1_G1": _slice_tm(dca, r1_lo, r1_hi, g1_ca),
            "tm_R1_G2": _slice_tm(dca, r1_lo, r1_hi, g2_ca),
            "tm_R2_G1": _slice_tm(dca, r2_lo, r2_hi, g1_ca),
            "tm_R2_G2": _slice_tm(dca, r2_lo, r2_hi, g2_ca),
            "ca_M": dca[m_lo:m_hi], "ca_R1": dca[r1_lo:r1_hi], "ca_R2": dca[r2_lo:r2_hi],
        }
    return rows


def _tm_key(region, target):
    """Region-TM dict key for a flank's assigned target (``g1``/``g2``); ``None`` for a free flank."""
    if target == "g1":
        return f"tm_{region}_G1"
    if target == "g2":
        return f"tm_{region}_G2"
    return None


def _summarize_cell(base_rows, cell_rows, t1, t2, lam):
    """One cell: Δmotif, each flank's steer toward ITS target, and the cross terms (bleed check)."""
    def region_disp(key):
        vals = [1.0 - _pair_tm(cell_rows[i][key], base_rows[i][key])
                for i in cell_rows if i in base_rows]
        return sum(vals) / len(vals) if vals else None

    def steer(region, target):
        k = _tm_key(region, target)
        if k is None:
            return None
        return _mean(cell_rows, k) - _mean(base_rows, k)

    m_base, m_cell = _mean(base_rows, "motif_rmsd"), _mean(cell_rows, "motif_rmsd")
    return {
        "lambda": lam, "R1_target": t1, "R2_target": t2,
        "motif_rmsd_baseline": m_base, "motif_rmsd_cell": m_cell, "delta_motif_rmsd": m_cell - m_base,
        "M_disp": region_disp("ca_M"), "R1_disp": region_disp("ca_R1"), "R2_disp": region_disp("ca_R2"),
        # region×target TM in this cell (loc) — all four, so cross-talk is visible.
        "tm_R1_G1": _mean(cell_rows, "tm_R1_G1"), "tm_R1_G2": _mean(cell_rows, "tm_R1_G2"),
        "tm_R2_G1": _mean(cell_rows, "tm_R2_G1"), "tm_R2_G2": _mean(cell_rows, "tm_R2_G2"),
        # steer of each flank toward its OWN assigned target (None if that flank is free).
        "R1_steer": steer("R1", t1), "R2_steer": steer("R2", t2),
        # cross-steer: each flank toward the OTHER flank's fold — want ≈ baseline (no bleed).
        "R1_cross": steer("R1", t2) if t2 != "free" else None,
        "R2_cross": steer("R2", t1) if t1 != "free" else None,
    }


def run_two_steer(args):
    import torch
    from omegaconf import OmegaConf

    from spa.eval.generate import (_parse_contig_motif, _run_once, _seed_all,
                                    build_eval_engine, load_adapter)
    from spa.eval.score import _as_struct, source_positions
    from spa.train.harness import frozen_rfd3_net
    from spa.utils.device import resolve_device

    device = args.device
    dev = resolve_device(device)
    out_dir = Path(args.out_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    cells = _parse_cells(args.cells)
    lam = float(args.lambda_scale)
    # per-region effective λ (effective λ at residue i = λ_scale · profile[i]); fold the per-region λ
    # INTO the mask and keep λ_scale=1, so R1 and R2 can be driven at different strengths (dev: β needs
    # more push than α). Default = the global λ on both.
    lr1 = float(args.r1_lambda) if args.r1_lambda is not None else lam
    lr2 = float(args.r2_lambda) if args.r2_lambda is not None else lam
    K = int(args.num_designs)
    r1_len, r2_len = int(args.r1_len), int(args.r2_len)
    motif_pdb = _resolve_pdb(args.motif_source, args.pdb_dir)
    g1_pdb = _resolve_pdb(args.g1, args.pdb_dir)
    g2_pdb = _resolve_pdb(args.g2, args.pdb_dir)
    print(f"[2steer] R1|M|R2 = {r1_len}|{args.motif_seg}|{r2_len}   λ={lam}  K={K}  seed={args.seed}\n"
          f"[2steer]   G1={args.g1} ({r1_len}-res flank)   G2={args.g2} ({r2_len}-res flank)   "
          f"M={args.motif_seg}@{args.motif_source}\n[2steer]   cells: {['%s:%s' % c for c in cells]}")

    # scoring refs (constant across cells)
    g1_ca, g2_ca = _ca(g1_pdb), _ca(g2_pdb)
    src_struct = _as_struct(motif_pdb)
    src_positions = source_positions(
        src_struct, [(c, r) for (_d, c, r) in _parse_contig_motif(str(args.motif_seg))])

    # Embed BOTH folds ONCE (ESM3 loaded + freed before any RFD3 engine builds; dev 02 §5).
    p1 = torch.load(_precompute_prompt(g1_pdb, True, device, out_dir),
                    weights_only=True).float().to(dev)                       # [|G1|, 1536]
    p2 = torch.load(_precompute_prompt(g2_pdb, True, device, out_dir),
                    weights_only=True).float().to(dev)                       # [|G2|, 1536]

    base_model = {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True}
    base_variant = {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False}

    # Fixed contig (only prompts/profiles vary per cell) ⇒ build the engine ONCE. Optional RFD3-free
    # spacers decouple each soft region from the pinned motif / chain termini (dev: spacer probe).
    sp_in, sp_tm = int(args.inner_spacer or 0), int(args.term_spacer or 0)
    if sp_in or sp_tm:
        contig, M_idx, R1_idx, R2_idx, L = _spacered_partition(args.motif_seg, r1_len, r2_len, sp_in, sp_tm)
    else:
        contig, order = build_contig(args.motif_seg, r1_len, r2_len, "BAC")  # B=R1, A=M, C=R2
    cfg = OmegaConf.create({
        "paths": {"rfd3_ckpt": _rfd3_ckpt(getattr(args, "rfd3_ckpt", None))},
        "hardware": {"device": device},
        "model": base_model, "variant": base_variant,
        "eval": {"num_designs": K, "length": None, "specification": None,
                 "num_timesteps": args.num_timesteps, "seed": int(args.seed), "ckpt": args.ckpt,
                 "out_dir": str(out_dir), "motif": {"source_pdb": motif_pdb, "contig": contig}},
    })
    if sp_in or sp_tm:
        from spa.eval.generate import build_motif, _parse_contig_motif_indices
        motif_spec, _ = build_motif(cfg)
        assert sorted(_parse_contig_motif_indices(contig)) == M_idx, "spacer contig/M-idx mismatch"
    else:
        motif_spec, M_idx, R1_idx, R2_idx, L, _cr = build_partition(
            cfg, args.motif_seg, r1_len, r2_len, order)                      # U→R1, C→R2
    m_lo, m_hi = _contiguous(M_idx); r1_lo, r1_hi = _contiguous(R1_idx); r2_lo, r2_hi = _contiguous(R2_idx)
    geom = (m_lo, m_hi, r1_lo, r1_hi, r2_lo, r2_hi, M_idx)
    print(f"[2steer] contig {contig!r}  L={L}   R1[{r1_lo}:{r1_hi}]  M[{m_lo}:{m_hi}]  R2[{r2_lo}:{r2_hi}]")

    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, dev)
    adapter.eval()
    adtype = next(adapter.parameters()).dtype
    prompt1 = p1[None].expand(K, -1, -1).to(device=dev, dtype=adtype).contiguous()
    prompt2 = p2[None].expand(K, -1, -1).to(device=dev, dtype=adtype).contiguous()
    prof_R1 = lr1 * _profile(L, R1_idx, dev)                                 # effective λ = lr1 on R1 (0 elsewhere)
    prof_R2 = lr2 * _profile(L, R2_idx, dev)                                 # effective λ = lr2 on R2 (disjoint)
    prompt_of = {"g1": prompt1, "g2": prompt2}

    edir = out_dir / f"{args.motif_source}_{args.g1}x{args.g2}_R1{r1_len}M{len(M_idx)}R2{r2_len}"
    edir.mkdir(parents=True, exist_ok=True)

    base_rows, results = None, []
    for (t1, t2) in cells:
        slots = []
        if t1 != "free":
            slots.append((prompt_of[t1], prof_R1))
        if t2 != "free":
            slots.append((prompt_of[t2], prof_R2))
        if slots:
            adapter.set_prompts(slots); adapter.set_scale(1.0)   # per-region λ is folded into the profiles
        else:
            adapter.clear_prompt(); adapter.set_profile(None)               # free:free baseline
        cname = f"{t1}_{t2}"
        _seed_all(int(args.seed))
        with torch.no_grad():
            rows = _score_cell(_run_once(engine, motif_spec), edir, cname, geom,
                               src_struct, src_positions, g1_ca, g2_ca)
        if (t1, t2) == ("free", "free"):
            base_rows = rows
            print(f"[2steer]   baseline free:free   motif-RMSD {_mean(rows,'motif_rmsd'):.3f} Å   "
                  f"R1→G1 {_mean(rows,'tm_R1_G1'):.3f}  R2→G2 {_mean(rows,'tm_R2_G2'):.3f}", flush=True)
            continue
        s = _summarize_cell(base_rows, rows, t1, t2, lam)
        results.append(s)
        r1s = "  n/a " if s["R1_steer"] is None else f"{s['R1_steer']:+.3f}"
        r2s = "  n/a " if s["R2_steer"] is None else f"{s['R2_steer']:+.3f}"
        print(f"[2steer]   {t1}:{t2:<4}  R1→{t1} steer {r1s}   R2→{t2} steer {r2s}   "
              f"Δmotif {s['delta_motif_rmsd']:+.3f} Å", flush=True)

    summary = {"config": _config(args, cells, lam, L), "R1": [r1_lo, r1_hi], "M": [m_lo, m_hi],
               "R2": [r2_lo, r2_hi], "cells": results}
    (edir / "two_steer_result.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "two_steer_result.json").write_text(json.dumps(summary, indent=2, default=str))
    return results, lam, out_dir


def _config(args, cells, lam, L):
    # record the EFFECTIVE per-region λ (not just the global λ_scale) — else the JSON reads λ=global
    # default while the run actually used r1≠r2 (dev: this exact provenance gap was hit for r1_2.5_r2_1.5).
    lr1 = float(args.r1_lambda) if args.r1_lambda is not None else lam
    lr2 = float(args.r2_lambda) if args.r2_lambda is not None else lam
    return {"motif_source": args.motif_source, "motif_seg": args.motif_seg,
            "g1": args.g1, "r1_len": int(args.r1_len), "g2": args.g2, "r2_len": int(args.r2_len),
            "lambda": lam, "r1_lambda": lr1, "r2_lambda": lr2,
            "K": int(args.num_designs), "seed": int(args.seed), "L": int(L),
            "inner_spacer": int(args.inner_spacer or 0), "term_spacer": int(args.term_spacer or 0),
            "ckpt": args.ckpt, "cells": ["%s:%s" % c for c in cells]}


def _print_matrix(results, lam):
    f = lambda x: "  n/a " if x is None else f"{x:+.3f}"
    g = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"\n{'='*104}\nTWO-STEER  R1|M|R2  (λ={lam:g}) — each flank steers toward ITS target; cross = toward the "
          f"OTHER fold (want ~0)\n{'='*104}")
    hdr = (f"  {'cell (R1:R2)':<14}{'R1→G1':>8}{'R1→G2':>8}{'R2→G1':>8}{'R2→G2':>8}"
           f"{'R1steer':>9}{'R2steer':>9}{'R1cross':>9}{'R2cross':>9}{'Δmotif':>9}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for s in results:
        print(f"  {s['R1_target']+':'+s['R2_target']:<14}"
              f"{g(s['tm_R1_G1']):>8}{g(s['tm_R1_G2']):>8}{g(s['tm_R2_G1']):>8}{g(s['tm_R2_G2']):>8}"
              f"{f(s['R1_steer']):>9}{f(s['R2_steer']):>9}{f(s['R1_cross']):>9}{f(s['R2_cross']):>9}"
              f"{f(s['delta_motif_rmsd']):>9}")
    print("  " + "-" * (len(hdr) - 2))
    hero = next((s for s in results if (s["R1_target"], s["R2_target"]) == ("g1", "g2")), None)
    if hero:
        print(f"  HEADLINER g1:g2 — R1→G1 steer {f(hero['R1_steer'])}, R2→G2 steer {f(hero['R2_steer'])}, "
              f"cross R1→G2 {f(hero['R1_cross'])} / R2→G1 {f(hero['R2_cross'])}, Δmotif {f(hero['delta_motif_rmsd'])} Å")
    print(f"{'='*104}")
    print("[read] both R{1,2}steer > 0 with cross ≈ 0 (and Δmotif ≈ 0) ⇒ two disjoint region masks steer two")
    print("       flanks to two folds around one pinned motif, without bleed. Pick g1:g2 for the figure.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="N×1536 multigranularity SPA checkpoint (variant C)")
    ap.add_argument("--rfd3-ckpt", default=None, help="RFD3 frozen ckpt (default: $RFD3_CKPT or local models/)")
    ap.add_argument("--motif-source", default="A0A2X2KHU0", help="CDDB uniprot id or PDB path for the hard motif M")
    ap.add_argument("--motif-seg", default="A2-20", help="contig motif segment (chain-prefixed author range)")
    ap.add_argument("--g1", default="A0A1X7NTP0", help="fold G1 for flank R1 (CDDB uniprot id or PDB path)")
    ap.add_argument("--r1-len", type=int, default=90, help="|R1| — the R1 flank length (≈ |G1|)")
    ap.add_argument("--g2", default="A0A7C6QMG4", help="fold G2 for flank R2 (CDDB uniprot id or PDB path)")
    ap.add_argument("--r2-len", type=int, default=47, help="|R2| — the R2 flank length (≈ |G2|)")
    ap.add_argument("--lambda", dest="lambda_scale", type=float, default=2.0, help="SPA strength λ (global, both regions)")
    ap.add_argument("--r1-lambda", type=float, default=None, help="per-region effective λ on R1 (overrides --lambda for R1; effective λ = λ·mask)")
    ap.add_argument("--r2-lambda", type=float, default=None, help="per-region effective λ on R2 (overrides --lambda for R2)")
    ap.add_argument("--inner-spacer", type=int, default=0, help="free RFD3 residues inserted at R1|M and M|R2 (decouple soft regions from the pinned motif)")
    ap.add_argument("--term-spacer", type=int, default=0, help="free RFD3 residues at N-term (before R1) and C-term (after R2)")
    ap.add_argument("--cells", default="free:free,g1:g2,g2:g1,g1:g1,g2:g2,g1:free,free:g2",
                    help="comma list of R1:R2 target pairs; each side ∈ {free,g1,g2}. free:free is forced first.")
    ap.add_argument("--num-designs", type=int, default=8, help="K designs (paired noise)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--num-timesteps", type=int, default=None, help="sampler steps (None → rfd3 edm.yaml 100)")
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(_OUTPUTS_ROOT / "_incoming" / "twosteer"))
    args = ap.parse_args()

    results, lam, out_dir = run_two_steer(args)
    _print_matrix(results, lam)
    print(f"[2steer] wrote {out_dir / 'two_steer_result.json'}")


if __name__ == "__main__":
    main()
