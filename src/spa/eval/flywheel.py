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
    """Print the compact per-``(condition, λ)`` table the poster reads (dev ``05`` §3 / ``06`` §6)."""
    dmap = {(d.condition, float(d.lambda_scale)): d for d in deltas}
    print("\n=== SPA flywheel summary (dev 05 §3 / 06 §6) ===")
    print(f"{'condition':<10}{'lambda':>8}{'n':>5}{'succ_rate':>11}{'mean_TM':>10}{'dTM':>9}{'d_succ':>9}")
    print("-" * 62)
    for s in summaries:
        d = dmap.get((s.condition, float(s.lambda_scale)))
        d_tm = _fmt(d.d_tm_mean) if d else "—"
        d_succ = _fmt(d.d_success_rate) if d else "—"
        print(f"{s.condition:<10}{_fmt(s.lambda_scale, 3):>8}{s.n_designs:>5}"
              f"{_fmt(s.success_rate):>11}{_fmt(s.tm.mean):>10}{d_tm:>9}{d_succ:>9}")
    print()


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------


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
    designs = generate(cfg)

    # Stage 2 — inverse-fold each backbone; key by design name (== PDB stem == SequenceSet.name).
    seqsets = inverse_fold(cfg, designs=designs)
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

    # Stage 4 — score each design (adherence if a prompt struct exists; designability if refolds exist).
    scores = []
    for d in designs:
        ss = seqsets_by_name.get(d.path.stem)
        refolds = refolder.refold(ss) if (refolder is not None and ss is not None) else None
        scores.append(score_design(d, prompt=prompt_struct, refolds=refolds, cfg=cfg))

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
