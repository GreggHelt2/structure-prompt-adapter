"""Driver for the SPA validation flywheel — wires Stages 1→4 end-to-end (dev ``05`` §1–§4).

Spec: dev ``05_validation_pipeline.md`` §1–§4 (the self-consistency flywheel). The four stages are
already built with clean APIs; this module only **orchestrates** them — it adds no scoring/gen logic
of its own:

  1. **Generate** (:func:`spa.eval.generate.generate`) — RFD3 ± SPA backbones for every
     ``eval.conditions`` × ``eval.lambda_scale``, K = ``eval.num_designs`` each.
  2. **Inverse fold** (:func:`spa.eval.proteinmpnn.inverse_fold`) — ProteinMPNN N sequences per
     backbone, keyed by design name (== Stage-1 PDB stem == :attr:`SequenceSet.name`).
  3. **Refold (OpenFold3) — pluggable + STUBBED.** A :class:`spa.eval.score.Refolder` (passed as the
     ``refolder`` arg, or instantiated from ``eval.flywheel.refolder``) turns each ``SequenceSet`` into
     refold structures for designability. OF3 is **not** implemented here (it runs in a separate env,
     dev ``05`` Stage 3); with no refolder, designability is skipped (``refolds=None`` ⇒ adherence-only)
     and that is logged clearly.
  4. **Score** (:func:`spa.eval.score.score_design` / :func:`aggregate` / :func:`delta_vs_baseline`) —
     per-design adherence (vs the prompt's source structure) + designability (vs the refolds), then
     per-``(condition, λ)`` summaries and the headline Δ(SPA − baseline).

The adherence reference is the eval **structure** that produced the SPA prompt — ``eval.prompt_pdb``
or a dedicated ``eval.flywheel.prompt_struct`` (``eval.prompt_cache`` is an ESM3 *tensor*, not a
structure, so it cannot serve here); with neither, adherence is skipped (logged). Results (per-design
:class:`DesignScore` + summaries + deltas) are written to ``eval.out_dir/flywheel_results.json`` and a
compact per-condition table is printed.

All cost/threshold knobs live in config (``eval`` group; dev root ``CLAUDE.md`` portability rule) —
nothing hardware- or cost-specific is hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .generate import generate
from .gpu_lock import generation_lock
from .proteinmpnn import inverse_fold
from .score import aggregate, delta_vs_baseline, score_design


# --------------------------------------------------------------------------------------------------
# Resolution helpers (Stage-3 refolder + adherence reference structure)
# --------------------------------------------------------------------------------------------------


def _flywheel_cfg(cfg):
    """The ``eval.flywheel`` sub-config (a plain view), or an empty dict if absent."""
    try:
        return cfg.eval.get("flywheel") or {}
    except Exception:
        return {}


def _resolve_refolder(cfg, refolder):
    """Resolve the Stage-3 OF3 refolder: the passed ``refolder`` wins, else instantiate
    ``eval.flywheel.refolder`` (a Hydra ``_target_`` spec) if configured, else ``None`` (skip).

    OF3 is **not** implemented here — this only wires a pluggable :class:`~spa.eval.score.Refolder`.
    """
    if refolder is not None:
        return refolder
    spec = _flywheel_cfg(cfg).get("refolder") if hasattr(_flywheel_cfg(cfg), "get") else None
    if not spec:
        return None
    from hydra.utils import instantiate  # only reached when a refolder is actually configured

    return instantiate(spec)


def _resolve_prompt_struct(cfg):
    """The structure file to score prompt-adherence against (the eval structure behind the SPA prompt).

    Priority ``eval.flywheel.prompt_struct`` → ``eval.prompt_pdb``. ``eval.prompt_cache`` is an ESM3
    ``[N,1536]`` tensor (not a structure), so it is intentionally **not** a fallback here.
    """
    ev = cfg.eval
    for src in (_flywheel_cfg(cfg).get("prompt_struct") if hasattr(_flywheel_cfg(cfg), "get") else None,
                ev.get("prompt_pdb")):
        if src:
            return str(src)
    return None


# --------------------------------------------------------------------------------------------------
# Summary table (per condition/λ: n, success_rate, mean TM, Δ vs baseline)
# --------------------------------------------------------------------------------------------------


def _fmt(value, prec: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{prec}f}"


def _print_summary(summaries, deltas) -> None:
    """Print the compact per-``(condition, λ)`` table the poster reads (dev ``05`` §3 / ``06`` §6).

    Adds the hard⊕soft **motif** columns (motif-RMSD mean, satisfied-rate, Δmotif-RMSD) only when a motif
    was scored (review #10) — keeps non-motif runs uncluttered.
    """
    dmap = {(d.condition, float(d.lambda_scale)): d for d in deltas}
    has_motif = any(getattr(s, "motif_satisfied_rate", None) is not None
                    or (getattr(s, "motif_rmsd", None) is not None and s.motif_rmsd.n > 0)
                    for s in summaries)
    print("\n=== SPA flywheel summary (dev 05 §3 / 06 §6) ===")
    header = f"{'condition':<10}{'lambda':>8}{'n':>5}{'succ_rate':>11}{'mean_TM':>10}{'dTM':>9}{'d_succ':>9}"
    if has_motif:
        header += f"{'motifRMSD':>11}{'motif_sat':>10}{'dMotif':>9}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        d = dmap.get((s.condition, float(s.lambda_scale)))
        d_tm = _fmt(d.d_tm_mean) if d else "—"
        d_succ = _fmt(d.d_success_rate) if d else "—"
        row = (f"{s.condition:<10}{_fmt(s.lambda_scale, 3):>8}{s.n_designs:>5}"
               f"{_fmt(s.success_rate):>11}{_fmt(s.tm.mean):>10}{d_tm:>9}{d_succ:>9}")
        if has_motif:
            d_motif = _fmt(d.d_motif_rmsd_mean) if d else "—"
            row += f"{_fmt(s.motif_rmsd.mean):>11}{_fmt(s.motif_satisfied_rate):>10}{d_motif:>9}"
        print(row)
    print()


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------


def release_gpu_memory(stage: str) -> dict:
    """Drop CUDA allocations held by a finished stage, so the next one can actually use the card.

    **Why this exists.** Stage 3 refolds by launching OpenFold3 as a SUBPROCESS (its deps cannot be
    imported here), and a subprocess gets its own CUDA context: it cannot reuse anything this process
    is still holding. PyTorch's caching allocator does not return freed blocks to the driver on its
    own, so after Stage 1 the RFD3 engine's arena stays reserved for the life of the process even
    once the engine itself is unreachable.

    **Measured, by hitting it.** A 374-residue b1-full prompt generated all 16 backbones fine and then
    died in the refold stage with `torch.OutOfMemoryError`, whose own text reads: *"Process 310530 has
    21.95 GiB memory in use. Including non-PyTorch memory, this process has 1.55 GiB memory in use."*
    The parent held 21.95 GiB of a 23.55 GiB card; the OF3 child got 1.55 GiB and died asking for 20
    MiB. OpenFold3 alone needs about 4.3 GiB at that length (dev `docs/results/data/28_of3_length_ceiling.json`),
    so it fits comfortably; it simply never had the room. On an 80 GB H100 the slack hides this, which
    is why long prompts were believed to be "H100-only" (`configs/eval/manifest_b1_full.yaml`).

    Returns the before/after figures so callers can log them, and is a no-op without CUDA.
    """
    import gc

    import torch

    if not torch.cuda.is_available():
        return {}
    before_alloc = torch.cuda.memory_allocated() / 2**30
    before_resv = torch.cuda.memory_reserved() / 2**30
    gc.collect()
    torch.cuda.empty_cache()
    after_alloc = torch.cuda.memory_allocated() / 2**30
    after_resv = torch.cuda.memory_reserved() / 2**30
    print(f"[flywheel] released GPU memory {stage}: "
          f"allocated {before_alloc:.2f} -> {after_alloc:.2f} GiB, "
          f"reserved {before_resv:.2f} -> {after_resv:.2f} GiB", flush=True)
    return {"before_allocated_gib": before_alloc, "after_allocated_gib": after_alloc,
            "before_reserved_gib": before_resv, "after_reserved_gib": after_resv}


def run_flywheel(cfg, *, refolder=None) -> dict:
    """Run the full SPA validation flywheel (Stages 1→4) from a composed config; return the artifacts.

    Args:
        cfg: composed config (the ``eval`` group + ``model`` / ``variant`` / ``hardware`` / ``paths``).
        refolder: an optional :class:`~spa.eval.score.Refolder` (the OF3 injection point) — wins over
            ``eval.flywheel.refolder``; with neither, designability is skipped (adherence-only).

    Returns:
        ``{designs, seqsets, scores, summaries, deltas, results_path}``.
    """
    from .generate import _resolve_out_dir

    # Stage 1 — generate RFD3 ± SPA backbones.
    # ⭐ Held under a cross-process lock: generation is the ~5x VRAM peak (16.3 GB at K=16/L=227 against
    # 3.2 GB refolding), so two concurrent shards GENERATING cannot fit a 24 GB card while three
    # refolding fit easily. Serializing only this stage removes the collision class at a bounded cost;
    # staggering shard starts does not, because balanced shards stay in phase. See spa.eval.gpu_lock
    # and dev plan/85 §4f. The lock is released before Stage 2, so refolding stays fully concurrent.
    with generation_lock(label=str(_resolve_out_dir(cfg))):
        designs = generate(cfg)
        # The engine built inside generate() is unreachable now (run_flywheel passes none in, so
        # nothing holds it), but its allocator arena is not. Release INSIDE the lock, so the next
        # shard sees a card this process has already given back rather than one it is still holding.
        release_gpu_memory("after Stage 1 (generate)")

    # Stage 2 — inverse-fold each backbone; key by design name (== PDB stem == SequenceSet.name).
    seqsets = inverse_fold(cfg, designs=designs)
    release_gpu_memory("after Stage 2 (inverse fold)")
    seqsets_by_name = {ss.name: ss for ss in seqsets}

    # Stage 3 — refold (OF3): pluggable + stubbed.
    refolder = _resolve_refolder(cfg, refolder)
    if refolder is None:
        print("[flywheel] OF3 refold NOT wired (no `refolder` arg and `eval.flywheel.refolder` unset) "
              "-> skipping designability; scoring adherence only (refolds=None).")

    # Adherence reference: the eval structure that produced the SPA prompt.
    prompt_struct = _resolve_prompt_struct(cfg)
    if prompt_struct is None:
        print("[flywheel] no adherence prompt structure (set `eval.flywheel.prompt_struct` or "
              "`eval.prompt_pdb`) -> skipping adherence.")

    # Stage 3b — BATCHED refold when the refolder supports it (one OF3 model-load for the whole matrix,
    # vs one-per-backbone; ~halves eval wall-time at scale). Falls back to per-design refold otherwise.
    refolds_by_name = None
    if refolder is not None and hasattr(refolder, "refold_all"):
        refolds_by_name = refolder.refold_all([ss for ss in seqsets if ss is not None])

    # Run-B hard⊕soft: if a native motif was scaffolded, score motif-RMSD vs its source over the motif
    # residues (dev 14 §3). `(source_pdb, motif_residues)` — derived from the same eval.motif the generator
    # used, so the indices match the design frame. None ⇒ no motif scored (unconditional evals unchanged).
    motif_score = None
    if cfg.eval.get("motif") and not cfg.eval.motif.get("contig"):
        # UNINDEXED motif (shape A, dev 27 §5): RFD3 chooses each design's motif positions (its
        # diffused_index_map), so there are no a-priori design indices to score the motif-RMSD against here.
        # scRMSD designability + TM→G adherence need no motif indices, so they still score; the design-side
        # pin (guaranteed by RFD3's freeze) is validated post-hoc from diffused_index_map. motif_score stays None.
        print("[flywheel] unindexed motif (no contig) -> design positions are model-chosen; skipping "
              "design-side motif-RMSD (scRMSD + adherence still scored; pin guaranteed, dev 27 §5).")
    elif cfg.eval.get("motif"):
        from .generate import _parse_contig_motif, motif_atom_spec
        from .score import _as_struct, source_positions
        m = cfg.eval.motif
        parsed = _parse_contig_motif(str(m["contig"]))               # [(design_idx, chain, author_resid), ...]
        source_struct = _as_struct(str(m["source_pdb"]))             # load the source ONCE (review #8)
        design_idx = [d for d, _ch, _r in parsed]
        # map the contig's author-numbered motif residues -> source positional Cα indices (review #1):
        src_pos = source_positions(source_struct, [(ch, r) for _d, ch, r in parsed])
        atom_spec = motif_atom_spec(m)               # non-None only for an explicit per-residue tip-atom sel
        if atom_spec:                                # atomic enzyme motif (dev 26 §8.1): score the FIXED atoms
            n_atoms = sum(len(r["atoms"]) for r in atom_spec)
            print(f"[flywheel] atomic (tip-atom) motif: scoring {n_atoms} fixed atoms over "
                  f"{len(atom_spec)} residues via motif_atom_rmsd (dev 26 §8.6).")
            motif_score = {"source": source_struct, "design_residues": design_idx,
                           "source_residues": src_pos, "atom_spec": atom_spec}
        else:
            motif_score = (source_struct, design_idx, src_pos)
    elif cfg.eval.get("subregion"):
        # Sub-region "scaffolding" eval (dev 17 §7 / 16 §9.5): score the design's S region vs the
        # prompt structure's S region (self-aligned — design length == N == prompt length, so S indexes
        # both identically). prompt_struct is the held-out structure behind the SPA prompt (the same
        # adherence reference). Fills design-side + refold-side sub-region motif-RMSD like the Run-B path.
        from .generate import subregion_keep
        keep = subregion_keep(cfg)
        if prompt_struct is None:
            print("[flywheel] eval.subregion set but no prompt structure -> cannot score sub-region "
                  "motif-RMSD; skipping it (set eval.prompt_pdb / eval.flywheel.prompt_struct).")
        else:
            from .score import _as_struct
            motif_score = (_as_struct(prompt_struct), keep)   # self-aligned: design[S] vs prompt[S]

    # Stage 4 — score each design (adherence if a prompt struct exists; designability if refolds exist;
    # motif-RMSD if a motif was scaffolded).
    scores = []
    for d in designs:
        ss = seqsets_by_name.get(d.path.stem)
        if refolds_by_name is not None:
            refolds = refolds_by_name.get(d.path.stem)
        elif refolder is not None and ss is not None:
            refolds = refolder.refold(ss)
        else:
            refolds = None
        scores.append(score_design(d, prompt=prompt_struct, refolds=refolds, motif=motif_score, cfg=cfg))

    structs_by_name = {d.path.stem: d for d in designs}   # for the diversity leg in aggregate
    summaries = aggregate(scores, structs_by_name=structs_by_name, cfg=cfg)
    deltas = delta_vs_baseline(summaries)

    out_dir = _resolve_out_dir(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "flywheel_results.json"
    payload = {
        "scores": [asdict(s) for s in scores],
        "summaries": [asdict(s) for s in summaries],
        "deltas": [asdict(s) for s in deltas],
    }
    with open(results_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[flywheel] wrote results -> {results_path}")

    _print_summary(summaries, deltas)
    return {
        "designs": designs,
        "seqsets": seqsets,
        "scores": scores,
        "summaries": summaries,
        "deltas": deltas,
        "results_path": results_path,
    }
