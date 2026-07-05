"""Three-way composition probe — hard motif M ⊥ localized soft U (→ foreign fold G) ⊕ free C.

The **true A⊕B⊕C** (= dev ``16``'s H/S/F partition): in a **single design**, a **hard native RFD3
motif M** (real pinned coordinates), **disjoint from** a **localized soft-SPA region U** (steered
toward a *whole foreign fold G*), plus a genuinely **free region C**. This is the *union* of the two
halves already run — hard⊕soft (``02`` §4, F=∅) and B–C localization (``02`` §7 / dev ``20``, no
motif) — and the composition the abstract's "composable conditioning" headline points at. Full spec:
**dev ``docs/plan/21``**; conceptual model in ``21`` §2, orthogonality in ``21`` §3.

Letter map (``21`` §1):  **A = hard M · B = soft U · C = free**.

Mechanism / why M ∩ U = ∅ by construction (``21`` §2.2 / §3):
  - **A (hard M):** RFD3 native motif — real coords pinned via ``build_motif`` → ``motif_spec`` passed
    to ``_run_once(engine, motif_spec)`` (the §4 path, RFD3 pins carved CDDB self-motifs bulletproof:
    ``02`` §4 big-n = 0.018 Å mean, 100 %). Design-side ``set_profile = 0`` on M ⇒ SPA silent there.
  - **B (soft U):** design-side ``set_profile = 1`` on U only (``cross_attention.lambda_profile`` [L]).
    SPA still *reads* the whole G (keys), but its *effect* survives only on U (``21`` §2.2).
  - **C (free):** ``set_profile = 0``, no motif — RFD3-dreamed; its only path to move is RFD3's
    frozen-host U↔C coupling (the C-drag we measure).

Two orthogonal masks (``21`` §2.2): the **prompt/key** side (whole fold G, ``set_prompt``) and the
**design/query** side (per-residue ``set_profile`` gate) are independent. This forks the localization
probe convention (no ``N == L`` constraint on the prompt) and adds the motif — so it **bypasses**
``generate.py:516``'s subregion×motif guard (that guard is about *prompt-side* ``subregion_keep``,
not the design-side ``set_profile``; ``21`` §3). **N×1536 (variant C) only** — the sole variant that
can honor a per-residue "off on M / on for U / off for C" mask (``15`` §3/§4, ``17`` §5).

Metrics (all paired to a **motif-only baseline**, same seed ⇒ identical initial noise; ``21`` §4):
  - **motif-RMSD(M)** — Kabsch RMSD over M's Cα vs the source; ≈ 0 in *both* conditions, and
    **Δ(motif-RMSD) ≈ 0** ⇒ **A held** (SPA-on-U must not disturb the pin). *The key novel check —
    motif + ``set_profile`` have never run together.*
  - **U→G TM** (loc − baseline; region-TM, superposition-invariant, lead with TM per ``21`` §2.6)
    ⇒ **B worked** (U adopted G).
  - **C→G drag** (loc − baseline) and **C displacement** (1 − paired-TM) ⇒ **C free** (want drag ≪
    U-steer; some drag is the known strong-steer trade-off, ``21`` §2.7 / ``02`` §7).

Adherence-only (no ProteinMPNN/OF3) ⇒ fast + A5000-friendly. Smoke test = ONE example (``21`` §8).

Run (A5000) — the confirmed smoke-test case (dev ``21``; 2026-07-05):
    conda run -n spa-dev python scripts/eval/probe_hard_soft_free.py \
        --ckpt checkpoints/spa_C_multigran_final.pt \
        --motif-source A0A2X2KHU0 --motif-seg A2-20 \
        --target A0A090ME36 --u-len 90 --c-len 120 \
        --lambda 1 --num-designs 4 --seed 0 \
        --out-dir outputs/eval/threeway
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PDB_DIR = ("/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/"
                   "atomistica_data_release/pdb")
DEFAULT_PATTERN = "AF-{id}-F1-model_v4_esmfold_v1.pdb"
DEFAULT_CKPT = "checkpoints/spa_C_multigran_final.pt"


# --------------------------------------------------------------------------------------------------
# Structure / prompt helpers (region-TM is superposition-invariant; see probe_localization.py)
# --------------------------------------------------------------------------------------------------


def _resolve_pdb(spec, pdb_dir):
    """A motif/target arg may be a bare CDDB uniprot id (→ default pattern) or an explicit path."""
    p = Path(spec)
    if p.exists():
        return str(p)
    return str(Path(pdb_dir) / DEFAULT_PATTERN.format(id=spec))


def _ca(pdb_path):
    from spa.eval.score import _as_struct, _ca_array

    return _ca_array(_as_struct(pdb_path))


def _slice_tm(design_ca, lo, hi, target_ca):
    """max-normalized Cα TM of ``design[lo:hi]`` vs the whole ``target`` (superposition-invariant)."""
    import tmtools

    from spa.eval.score import _coords64, _seq_of

    d = design_ca[lo:hi]
    res = tmtools.tm_align(_coords64(d), _coords64(target_ca), _seq_of(d), _seq_of(target_ca))
    return max(float(res.tm_norm_chain1), float(res.tm_norm_chain2))


def _pair_tm(ca_a, ca_b):
    """TM between two equal-length Cα slices (loc[region] vs baseline[region]); disp = 1 − this."""
    import tmtools

    from spa.eval.score import _coords64, _seq_of

    res = tmtools.tm_align(_coords64(ca_a), _coords64(ca_b), _seq_of(ca_a), _seq_of(ca_b))
    return max(float(res.tm_norm_chain1), float(res.tm_norm_chain2))


def _contiguous(idxs):
    """(lo, hi) if ``idxs`` is a contiguous ascending run [lo, hi); else raise (region TM needs a slice)."""
    lo, hi = idxs[0], idxs[-1] + 1
    if list(idxs) != list(range(lo, hi)):
        raise ValueError(f"region indices are not contiguous: {idxs[:3]}…{idxs[-3:]}")
    return lo, hi


# --------------------------------------------------------------------------------------------------
# Partition: (layout of A/B/C) → contig → (motif M indices, U indices, C indices), all disjoint.
# The three regions are one contiguous block each; only their ORDER along the chain varies, so we can
# sweep sequence placement (A-B-C vs C-A-B vs …; dev 21 §4.1) — a robustness + chain-locality probe:
# does the pin hold anywhere, and is C freer when NOT adjacent to U (21 §5, RFD3 host-coherence)?
# --------------------------------------------------------------------------------------------------


def _seg_len(seg):
    """Residue count of a chain-prefixed motif segment token (``A2-20`` → 19; ``A102`` → 1)."""
    import re

    m = re.fullmatch(r"([A-Za-z]+)(\d+)(?:-(\d+))?", seg.strip())
    if not m:
        raise ValueError(f"--motif-seg {seg!r}: expected chain+range like A2-20 or A102")
    start = int(m.group(2)); end = int(m.group(3)) if m.group(3) else start
    if end < start:
        raise ValueError(f"--motif-seg {seg!r}: end < start")
    return end - start + 1


def build_contig(motif_seg, u_len, c_len, layout):
    """Compose the RFD3 contig in the requested region order. A=hard motif seg, B=soft gap, C=free gap.

    ``layout`` is a permutation of ``A,B,C`` (e.g. ``ABC`` = motif|U|C, ``CAB`` = C|motif|U). Returns
    ``(contig_str, order_list)``.
    """
    order = list(str(layout).upper())
    if sorted(order) != ["A", "B", "C"]:
        raise ValueError(f"--layout {layout!r} must be a permutation of A,B,C (A=hard M / B=soft U / C=free)")
    role_tok = {"A": str(motif_seg), "B": str(int(u_len)), "C": str(int(c_len))}
    return ",".join(role_tok[r] for r in order), order


def _layout_spans(order, n_motif, u_len, c_len):
    """Walk the region order left→right; return ``{role: [design indices]}`` and total length L."""
    size = {"A": int(n_motif), "B": int(u_len), "C": int(c_len)}
    cursor, spans = 0, {}
    for r in order:
        spans[r] = list(range(cursor, cursor + size[r])); cursor += size[r]
    return spans, cursor


def build_partition(cfg, motif_seg, u_len, c_len, order):
    """Build the motif spec + the M/U/C design-index partition for the given region ``order`` (``21`` §4).

    Returns ``(motif_spec, M_idx, U_idx, C_idx, L, chain_resids)`` — M = the motif segment's design
    indices (cross-checked against the contig parser, authoritative), U/B = the soft gap, C = the free
    gap; each a contiguous block, disjoint, covering ``[0, L)``.
    """
    from spa.eval.generate import build_motif, _parse_contig_motif, _parse_contig_motif_indices

    motif_spec, _residues = build_motif(cfg)           # cfg.eval.motif.contig already set (build_contig)
    if motif_spec is None:
        raise ValueError("build_partition: no eval.motif configured (need a hard motif M).")
    contig = str(cfg.eval.motif["contig"])
    spans, L = _layout_spans(order, _seg_len(motif_seg), u_len, c_len)
    M_idx, U_idx, C_idx = spans["A"], spans["B"], spans["C"]
    parsed = sorted(_parse_contig_motif_indices(contig))   # the contig walk is authoritative on M
    if parsed != M_idx:
        raise ValueError(f"layout/contig motif-index mismatch: layout M={M_idx[:2]}…{M_idx[-2:]} "
                         f"vs contig-parsed {parsed[:2]}…{parsed[-2:]}")
    chain_resids = [(c, r) for (_d, c, r) in _parse_contig_motif(contig)]
    return motif_spec, M_idx, U_idx, C_idx, L, chain_resids


def _profile(L, U_idx, device):
    """Per-residue λ weight [L]: 1 on U, 0 on M ∪ C (design-side ``set_profile``; ``21`` §2.2/§3)."""
    import torch

    w = torch.zeros(L)
    w[U_idx] = 1.0
    return w.to(device)


# --------------------------------------------------------------------------------------------------
# ESM3 prompt for G (one residency; freed before RFD3 builds — they must not co-reside; dev 02 §5)
# --------------------------------------------------------------------------------------------------


def _precompute_prompt(target_pdb, strip, device, out_dir):
    import torch

    from spa.prompt.esm3_prompt import esm3_prompt, load_esm3

    cdir = out_dir / "prompts"; cdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device) if isinstance(device, str) else device
    model = load_esm3(dev)
    try:
        p = esm3_prompt(target_pdb, model, strip_bos_eos=strip, use_sequence=False)
        p = p.detach().float().cpu()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = cdir / f"{Path(target_pdb).stem}.pt"
    torch.save(p, out)
    print(f"[hsf] ESM3 prompt G: [{p.shape[0]},{p.shape[1]}] -> {out}")
    return out


def _rfd3_ckpt():
    return str(Path.home() / "projects/spa/models/rfdiffusion3/rfd3_latest.ckpt")


# --------------------------------------------------------------------------------------------------
# Grid: sweep layouts × λ. G is embedded ONCE (constant); per layout the motif-only baseline is run
# ONCE (λ-independent) and reused for every λ. Engine is rebuilt per layout (the proven single-run
# path; ~30 s, and keeps ESM3/RFD3 from co-residing). Same --seed for every run ⇒ within-layout
# paired A/B and the same initial-noise cloud across the grid (21 §2.4; see the seed note in the doc).
# --------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return sum(r[key] for r in rows.values()) / len(rows)


def _score_condition(outs, edir, cname, geom, src_struct, src_positions, target_ca):
    """Score one condition's K designs: motif-RMSD(M), U→G / C→G region-TM, and keep region Cα."""
    from spa.eval.generate import write_pdb
    from spa.eval.score import _ca_array, motif_rmsd

    m_lo, m_hi, u_lo, u_hi, c_lo, c_hi, M_idx = geom
    rows = {}
    for idx, o in enumerate(outs):
        aa = o.atom_array
        write_pdb(aa, edir / f"{cname}_{idx}.pdb")
        dca = _ca_array(aa)
        rows[idx] = {
            "motif_rmsd": motif_rmsd(aa, src_struct, M_idx, source_residues=src_positions),
            "tm_U": _slice_tm(dca, u_lo, u_hi, target_ca),
            "tm_C": _slice_tm(dca, c_lo, c_hi, target_ca),
            "ca_M": dca[m_lo:m_hi], "ca_U": dca[u_lo:u_hi], "ca_C": dca[c_lo:c_hi],
        }
    return rows


def _summarize_pair(base_rows, loc_rows, lam):
    """One (layout, λ) cell: A held? / U-steer / C-drag / paired region displacements (21 §4)."""
    def disp(key):
        vals = [1.0 - _pair_tm(loc_rows[i][key], base_rows[i][key]) for i in loc_rows if i in base_rows]
        return sum(vals) / len(vals) if vals else None

    m_base, m_loc = _mean(base_rows, "motif_rmsd"), _mean(loc_rows, "motif_rmsd")
    u_steer = _mean(loc_rows, "tm_U") - _mean(base_rows, "tm_U")
    c_drag = _mean(loc_rows, "tm_C") - _mean(base_rows, "tm_C")
    return {
        "lambda": lam,
        "motif_rmsd_baseline": m_base, "motif_rmsd_loc": m_loc, "delta_motif_rmsd": m_loc - m_base,
        "M_disp": disp("ca_M"),
        "tm_U_baseline": _mean(base_rows, "tm_U"), "tm_U_loc": _mean(loc_rows, "tm_U"),
        "U_steer": u_steer, "U_disp": disp("ca_U"),
        "tm_C_baseline": _mean(base_rows, "tm_C"), "tm_C_loc": _mean(loc_rows, "tm_C"),
        "C_drag": c_drag, "C_disp": disp("ca_C"),
        # localization quality: how much more U steers than C drags (want » 1); net = U_steer − C_drag.
        "loc_ratio": (u_steer / c_drag) if c_drag not in (0.0, None) and c_drag > 0 else None,
        "net_steer": u_steer - c_drag,
    }


def run_grid(args):
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

    layouts = ([s.strip().upper() for s in args.layouts.split(",") if s.strip()]
               if args.layouts else [str(args.layout).upper()])
    lambdas = ([float(x) for x in args.lambdas.split(",") if x.strip()]
               if args.lambdas else [float(args.lambda_scale)])
    K = int(args.num_designs)
    motif_pdb = _resolve_pdb(args.motif_source, args.pdb_dir)
    target_pdb = _resolve_pdb(args.target, args.pdb_dir)
    print(f"[hsf] GRID  layouts={layouts}  λ={lambdas}  K={K}  seed={args.seed}  "
          f"M={args.motif_seg}@{args.motif_source}  U→G={args.target}  |U|={args.u_len} |C|={args.c_len}")

    # scoring refs — the motif segment + G are constant across the grid, so compute once.
    target_ca = _ca(target_pdb)
    src_struct = _as_struct(motif_pdb)
    src_positions = source_positions(  # (chain, author_resid) of the motif seg → positional Cα idx
        src_struct, [(c, r) for (_d, c, r) in _parse_contig_motif(str(args.motif_seg))])

    # Embed G ONCE (ESM3 loaded + freed here, before any RFD3 engine builds; dev 02 §5).
    prompt_pt = _precompute_prompt(target_pdb, True, device, out_dir)
    p = torch.load(prompt_pt, weights_only=True).float().to(dev)          # [|G|, 1536]

    base_model = {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True}
    base_variant = {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False}

    grid = []
    for layout in layouts:
        contig, order = build_contig(args.motif_seg, args.u_len, args.c_len, layout)
        cfg = OmegaConf.create({
            "paths": {"rfd3_ckpt": _rfd3_ckpt()},
            "hardware": {"device": device},
            "model": base_model, "variant": base_variant,
            "eval": {"num_designs": K, "length": None, "specification": None,
                     "num_timesteps": args.num_timesteps, "seed": int(args.seed), "ckpt": args.ckpt,
                     "out_dir": str(out_dir), "motif": {"source_pdb": motif_pdb, "contig": contig}},
        })
        motif_spec, M_idx, U_idx, C_idx, L, _cr = build_partition(
            cfg, args.motif_seg, int(args.u_len), int(args.c_len), order)
        m_lo, m_hi = _contiguous(M_idx); u_lo, u_hi = _contiguous(U_idx); c_lo, c_hi = _contiguous(C_idx)
        geom = (m_lo, m_hi, u_lo, u_hi, c_lo, c_hi, M_idx)
        adjacency = "U|C adjacent" if (u_hi == c_lo or c_hi == u_lo) else "U,C separated by M"
        print(f"\n[hsf] === layout {layout}  contig {contig!r}  L={L}  "
              f"M[{m_lo}:{m_hi}] U[{u_lo}:{u_hi}] C[{c_lo}:{c_hi}]  ({adjacency}) ===")

        engine = build_eval_engine(cfg)                       # proven single-run path, per layout
        net = frozen_rfd3_net(engine)
        adapter = load_adapter(net, cfg, dev)
        adapter.eval()
        adtype = next(adapter.parameters()).dtype
        prompt = p[None].expand(K, -1, -1).to(device=dev, dtype=adtype).contiguous()
        profile = _profile(L, U_idx, dev)

        edir = out_dir / f"{layout}_{args.motif_source}_{args.target}_M{len(M_idx)}U{len(U_idx)}C{len(C_idx)}"
        edir.mkdir(parents=True, exist_ok=True)

        # baseline (motif-only, SPA off) — ONCE per layout, reused for every λ (λ-independent).
        adapter.clear_prompt(); adapter.set_profile(None)
        _seed_all(int(args.seed))
        with torch.no_grad():
            base_rows = _score_condition(_run_once(engine, motif_spec), edir,
                                         "baseline_motif_only", geom, src_struct, src_positions, target_ca)
        print(f"[hsf]   baseline           motif-RMSD {_mean(base_rows,'motif_rmsd'):.3f} Å   "
              f"U→G {_mean(base_rows,'tm_U'):.3f}   C→G {_mean(base_rows,'tm_C'):.3f}", flush=True)

        cells = {}
        for lam in lambdas:
            adapter.set_prompt(prompt); adapter.set_scale(lam); adapter.set_profile(profile)
            _seed_all(int(args.seed))
            with torch.no_grad():
                loc_rows = _score_condition(_run_once(engine, motif_spec), edir,
                                            f"localized_l{lam:g}", geom, src_struct, src_positions, target_ca)
            s = _summarize_pair(base_rows, loc_rows, lam)
            cells[f"{lam:g}"] = s
            print(f"[hsf]   λ={lam:<4g}  U→G {s['tm_U_loc']:.3f} (steer {s['U_steer']:+.3f})   "
                  f"C→G {s['tm_C_loc']:.3f} (drag {s['C_drag']:+.3f})   net {s['net_steer']:+.3f}   "
                  f"Δmotif {s['delta_motif_rmsd']:+.3f} Å", flush=True)

        layout_summ = {"layout": layout, "contig": contig, "L": L, "adjacency": adjacency,
                       "M": [m_lo, m_hi], "U": [u_lo, u_hi], "C": [c_lo, c_hi], "lambdas": cells}
        (edir / "result.json").write_text(json.dumps(
            {"config": _grid_config(args, layouts, lambdas), **layout_summ}, indent=2, default=str))
        grid.append(layout_summ)

        del engine, adapter, net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return grid, out_dir


def _grid_config(args, layouts, lambdas):
    return {"motif_source": args.motif_source, "motif_seg": args.motif_seg, "target_G": args.target,
            "u_len": int(args.u_len), "c_len": int(args.c_len), "K": int(args.num_designs),
            "seed": int(args.seed), "ckpt": args.ckpt, "layouts": layouts, "lambdas": lambdas}


# --------------------------------------------------------------------------------------------------
# Grid table + best-cell read
# --------------------------------------------------------------------------------------------------


def _print_grid(grid, lambdas):
    f = lambda x: "  n/a " if x is None else f"{x:+.3f}"
    g = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"\n{'='*96}\nTHREE-WAY A⊕B⊕C GRID — U-steer (want ↑) · C-drag (want ~0) · net=steer−drag (want ↑) · "
          f"Δmotif (want 0)\n{'='*96}")
    hdr = f"  {'layout':<7}{'adjacency':<20}{'λ':>5}{'U→G':>8}{'steer':>8}{'C→G':>8}{'drag':>8}{'net':>8}{'Δmotif':>8}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    best = None
    for ls in grid:
        for lam in lambdas:
            d = ls["lambdas"].get(f"{lam:g}")
            if d is None:
                continue
            print(f"  {ls['layout']:<7}{ls['adjacency']:<20}{lam:>5g}{g(d['tm_U_loc']):>8}"
                  f"{f(d['U_steer']):>8}{g(d['tm_C_loc']):>8}{f(d['C_drag']):>8}{f(d['net_steer']):>8}"
                  f"{f(d['delta_motif_rmsd']):>8}")
            # "best" = strongest net localized steer (U-steer − C-drag) with the pin still held.
            if abs(d["delta_motif_rmsd"]) < 0.5 and (best is None or d["net_steer"] > best[3]["net_steer"]):
                best = (ls["layout"], ls["adjacency"], lam, d)
    print("  " + "-" * (len(hdr) - 2))
    if best:
        lay, adj, lam, d = best
        print(f"  BEST net-steer: layout {lay} ({adj}) @ λ={lam:g}  →  U→G {d['tm_U_loc']:.3f} "
              f"(steer {d['U_steer']:+.3f}), C-drag {d['C_drag']:+.3f}, net {d['net_steer']:+.3f}, "
              f"Δmotif {d['delta_motif_rmsd']:+.3f} Å")
    print(f"{'='*96}")
    print("[read] A⊕B⊕C composes (Δmotif≈0 everywhere expected). Pick the (layout,λ) with strong U→G AND")
    print("       small C-drag for the figure; compare U|C-adjacent vs M-separates-U,C to see if placement frees C.")


# --------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="N×1536 multigranularity SPA checkpoint (variant C)")
    ap.add_argument("--motif-source", default="A0A2X2KHU0", help="CDDB uniprot id or PDB path for the hard motif M")
    ap.add_argument("--motif-seg", default="A2-20", help="contig motif segment (chain-prefixed author range), e.g. A2-20")
    ap.add_argument("--target", default="A0A090ME36", help="foreign fold G (CDDB uniprot id or PDB path) U steers toward")
    ap.add_argument("--u-len", type=int, default=90, help="|U| — the soft region, sized ≈ |G|")
    ap.add_argument("--c-len", type=int, default=120, help="|C| — the free region")
    ap.add_argument("--layout", default="ABC",
                    help="single region order (a permutation of A=hard M / B=soft U / C=free); "
                         "used when --layouts is not given. e.g. ABC (default), CAB, BAC (dev 21 §4.1)")
    ap.add_argument("--layouts", default=None,
                    help="comma list of layouts to sweep (overrides --layout), e.g. ABC,BAC,CAB")
    ap.add_argument("--lambda", dest="lambda_scale", type=float, default=1.0,
                    help="single SPA strength λ on U; used when --lambdas is not given (02 §7)")
    ap.add_argument("--lambdas", default=None, help="comma list of λ to sweep (overrides --lambda), e.g. 1,2,3")
    ap.add_argument("--num-designs", type=int, default=8, help="K designs (paired noise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-timesteps", type=int, default=None, help="sampler steps (None → rfd3 edm.yaml 100)")
    ap.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="outputs/eval/threeway")
    args = ap.parse_args()

    grid, out_dir = run_grid(args)
    lambdas = ([float(x) for x in args.lambdas.split(",") if x.strip()]
               if args.lambdas else [float(args.lambda_scale)])
    _print_grid(grid, lambdas)
    (out_dir / "grid_result.json").write_text(json.dumps(
        {"config": _grid_config(args, [ls["layout"] for ls in grid], lambdas), "grid": grid},
        indent=2, default=str))
    print(f"[hsf] wrote {out_dir / 'grid_result.json'}")


if __name__ == "__main__":
    main()
