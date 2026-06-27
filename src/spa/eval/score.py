"""Stage 4 of the SPA validation flywheel: score designs (designability + prompt adherence).

Spec: dev ``05_validation_pipeline.md`` §3 ("Metrics") and ``06_experiments.md`` §6 ("Success
criteria"). Consumes Stage-1 :class:`spa.eval.generate.Design` backbones, the Stage-2
:class:`spa.eval.proteinmpnn.SequenceSet` sequences, and the Stage-3 OpenFold3 refolds, and produces
the comparison numbers the poster reports: **designability** (best-of-K self-consistency RMSD) and
**prompt adherence** (does an SPA design honor its structural prompt's fold), aggregated per
``(condition, λ)`` with the headline **Δ(SPA − baseline)**.

Two metric families (dev ``05`` §3):

- **Adherence (no OF3 needed)** — the SPA-vs-baseline headline. Align a design to its prompt's source
  structure and report a length-independent **TM-score** (primary; ``tmtools.tm_align`` — robust to
  the design and prompt differing in length) plus a **Cα Kabsch prompt-RMSD** (reported only when the
  residue counts match, since RMSD needs a 1:1 correspondence). Computed straight from the two
  backbones — it does **not** depend on the ProteinMPNN→OF3 legs, so it is available immediately.
- **Designability (self-consistency)** — best-of-K Cα **scRMSD** between a design backbone and the
  OF3 **refold** of each ProteinMPNN sequence for it; ``designable`` iff ``scRMSD < scrmsd_cutoff``
  (and OF3 pLDDT ≥ ``plddt_cutoff`` when a confidence is supplied). The **refold is a pluggable
  input** (an in-memory ``AtomArray``, a path, or a :class:`Refolder` that turns a ``SequenceSet``
  into refold structures) — OF3 itself is **not** implemented here (dev ``05`` Stage 3 runs in a
  separate env); a stand-in refold validates the math in the smoke test.

Plus two best-effort population checks (dev ``05`` §3): **diversity** (pairwise TM among a condition's
designable designs — does SPA collapse to one fold or span the prompt's topology?) and **novelty**
(vs PDB — stubbed; Foldseek may be absent, so it logs + ``NotImplementedError`` rather than blocking).

All thresholds are config (``eval.score`` — ``scrmsd_cutoff``/``plddt_cutoff``/``tm_norm``/…), never
hardcoded (dev root ``CLAUDE.md`` portability rule); every scoring function also takes them as
explicit args so a driver/test can call it without a full Hydra config. TM-score and RMSD are
CPU-only and tiny, so scoring needs no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------------------------------
# Config defaults (mirror the eval config block; see configs/eval/default.yaml -> `score:`)
# --------------------------------------------------------------------------------------------------

DEFAULTS = {
    "scrmsd_cutoff": 2.0,   # Å — best-of-K Cα scRMSD designability threshold (dev 05 §3 / 06 §6)
    "plddt_cutoff": 80.0,   # OF3 pLDDT designability gate (applied only when a pLDDT is supplied)
    "tm_norm": "prompt",    # primary TM normalization for adherence: prompt | design | max
    "diversity": True,      # compute pairwise-TM diversity among the designable set
    "novelty": False,       # novelty-vs-PDB (Foldseek) — stub; raises NotImplemented if requested
}


@dataclass
class ScoreConfig:
    """Resolved ``eval.score`` knobs (defaults from :data:`DEFAULTS`)."""

    scrmsd_cutoff: float = DEFAULTS["scrmsd_cutoff"]
    plddt_cutoff: float = DEFAULTS["plddt_cutoff"]
    tm_norm: str = DEFAULTS["tm_norm"]
    diversity: bool = DEFAULTS["diversity"]
    novelty: bool = DEFAULTS["novelty"]


def score_config(cfg=None) -> ScoreConfig:
    """Read the ``eval.score`` sub-config into a :class:`ScoreConfig`, falling back to defaults.

    Tolerant of a missing ``eval``/``score`` group (returns all defaults) so the scoring functions
    run from a partial config or none at all.
    """
    sc: dict = {}
    try:
        ev = cfg.eval if cfg is not None else None
        sc = dict((ev.get("score") if ev is not None else None) or {})
    except Exception:
        sc = {}
    return ScoreConfig(
        scrmsd_cutoff=float(sc.get("scrmsd_cutoff", DEFAULTS["scrmsd_cutoff"])),
        plddt_cutoff=float(sc.get("plddt_cutoff", DEFAULTS["plddt_cutoff"])),
        tm_norm=str(sc.get("tm_norm", DEFAULTS["tm_norm"])),
        diversity=bool(sc.get("diversity", DEFAULTS["diversity"])),
        novelty=bool(sc.get("novelty", DEFAULTS["novelty"])),
    )


# --------------------------------------------------------------------------------------------------
# Result dataclasses
# --------------------------------------------------------------------------------------------------


@dataclass
class Adherence:
    """Prompt-adherence metrics for one design vs its prompt's source structure (dev ``05`` §3).

    Attributes:
        tm_score: the primary length-independent TM-score (per ``tm_norm``; in [0, 1], higher = more
            adherent). Equal to the prompt- or design-normalized value, or their max.
        tm_norm_design: TM-score normalized by the **design** length (``tm_norm_chain1``).
        tm_norm_prompt: TM-score normalized by the **prompt** length (``tm_norm_chain2``) — "fraction
            of the prompt fold reproduced".
        prompt_rmsd: Cα Kabsch RMSD (Å) over the 1:1 residue correspondence, or ``None`` when the
            design and prompt residue counts differ (no correspondence ⇒ RMSD undefined).
        n_design: design Cα count.
        n_prompt: prompt Cα count.
    """

    tm_score: float
    tm_norm_design: float
    tm_norm_prompt: float
    prompt_rmsd: float | None
    n_design: int
    n_prompt: int


@dataclass
class SelfConsistency:
    """Best-of-K self-consistency result for one design (dev ``05`` §3 "scRMSD").

    Attributes:
        scrmsd: the **best** (minimum) Cα RMSD (Å) over the K refolds (``nan`` if none were valid).
        best_refold_idx: index of the refold achieving ``scrmsd`` (``-1`` if none valid).
        per_refold: per-refold Cα RMSD (``nan`` for a refold whose length didn't match the design).
    """

    scrmsd: float
    best_refold_idx: int
    per_refold: list[float] = field(default_factory=list)


@dataclass
class DesignScore:
    """All scores for a single design (dev ``05`` §3 / ``06`` §6), the unit of aggregation.

    Adherence fields are ``None`` when no prompt was supplied; designability fields are ``None`` when
    no refold was supplied — so a baseline (no-prompt) design carries designability only, and an
    adherence-only pass (no OF3 yet) carries adherence only.
    """

    name: str
    condition: str
    lambda_scale: float
    n_residues: int
    # adherence (vs the structural prompt)
    tm_score: float | None = None
    tm_norm_design: float | None = None
    tm_norm_prompt: float | None = None
    prompt_rmsd: float | None = None
    # designability (vs OF3 refolds of the ProteinMPNN sequences)
    scrmsd: float | None = None
    best_refold_idx: int | None = None
    plddt: float | None = None
    designable: bool | None = None


@dataclass
class Distribution:
    """A small numeric summary of a metric over a group of designs (None/NaN dropped)."""

    n: int
    mean: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None


@dataclass
class ConditionSummary:
    """Aggregate over all designs sharing a ``(condition, λ)`` (dev ``06`` §6: rate + distribution)."""

    condition: str
    lambda_scale: float
    n_designs: int
    n_designable: int
    success_rate: float | None              # fraction designable (None if no designability scored)
    tm: Distribution                        # adherence TM-score distribution
    prompt_rmsd: Distribution               # adherence Cα-RMSD distribution
    scrmsd: Distribution                    # best-of-K scRMSD distribution
    diversity_tm: float | None = None       # mean pairwise TM among the designable set (lower = more diverse)
    novelty: float | None = None            # stubbed (Foldseek) — always None here


@dataclass
class DeltaSummary:
    """Δ(SPA − baseline) for one SPA ``(condition, λ)`` vs the baseline group (dev ``06`` §6 headline)."""

    condition: str
    lambda_scale: float
    d_success_rate: float | None
    d_tm_mean: float | None
    d_prompt_rmsd_mean: float | None
    spa_success_rate: float | None
    baseline_success_rate: float | None
    spa_tm_mean: float | None
    baseline_tm_mean: float | None


# --------------------------------------------------------------------------------------------------
# Structure extraction (AtomArray / path / Design -> Cα coords + 1-letter sequence)
# --------------------------------------------------------------------------------------------------


def _first_model(arr):
    """Collapse an ``AtomArrayStack`` (multi-model file) to its first model; pass an ``AtomArray``."""
    from biotite.structure import AtomArrayStack

    if isinstance(arr, AtomArrayStack):
        return arr[0]
    return arr


def _as_struct(obj):
    """Coerce a Design / AtomArray / path / str into a biotite ``AtomArray``.

    A :class:`spa.eval.generate.Design` is used via its in-memory ``.atom_array`` when present (no
    disk round-trip), else loaded from ``.path``; a bare path/str is loaded with biotite's format
    auto-detection (PDB or CIF); an ``AtomArray`` is passed through.
    """
    from biotite.structure.io import load_structure

    if obj is None:
        raise ValueError("score: structure is None")
    aa = getattr(obj, "atom_array", None)
    if aa is not None:
        return _first_model(aa)
    path = getattr(obj, "path", None)
    if path is not None:
        return _first_model(load_structure(str(path)))
    if isinstance(obj, (str, Path)):
        return _first_model(load_structure(str(obj)))
    if hasattr(obj, "coord") and hasattr(obj, "atom_name"):
        return _first_model(obj)
    raise TypeError(f"score: cannot coerce {type(obj)!r} to an AtomArray")


def _ca_array(struct):
    """The one-Cα-per-residue ``AtomArray`` (``arr[arr.atom_name == 'CA']``; dev task spec)."""
    arr = _as_struct(struct)
    return arr[arr.atom_name == "CA"]


def _seq_of(ca) -> str:
    """1-letter sequence from a Cα ``AtomArray`` (unknown residues -> 'X'); seeds TM-align."""
    from biotite.sequence import ProteinSequence

    out: list[str] = []
    for rn in ca.res_name:
        try:
            out.append(ProteinSequence.convert_letter_3to1(rn))
        except Exception:
            out.append("X")
    return "".join(out)


def _coords64(ca):
    """Cα coordinates as a contiguous ``float64`` ``[N, 3]`` (tmtools/biotite want double)."""
    return ca.coord.astype("float64")


# --------------------------------------------------------------------------------------------------
# Adherence — TM-score (primary) + Cα prompt-RMSD (dev 05 §3)
# --------------------------------------------------------------------------------------------------


def tm_score(struct1, struct2) -> tuple[float, float]:
    """Cα TM-score of ``struct1`` vs ``struct2`` → ``(tm_norm_chain1, tm_norm_chain2)`` via tmtools.

    ``tm_norm_chain1`` is normalized by ``struct1``'s length, ``tm_norm_chain2`` by ``struct2``'s;
    they coincide when the two have equal length. TM-score is length-independent and superposition-
    (rotation/translation-)invariant, so it is the primary adherence/diversity metric (dev ``05`` §3).
    """
    import tmtools

    c1 = _ca_array(struct1)
    c2 = _ca_array(struct2)
    res = tmtools.tm_align(_coords64(c1), _coords64(c2), _seq_of(c1), _seq_of(c2))
    return float(res.tm_norm_chain1), float(res.tm_norm_chain2)


def ca_rmsd(fixed_struct, mobile_struct) -> float:
    """Cα Kabsch-superposed RMSD (Å): fit ``mobile`` onto ``fixed`` then RMSD over the 1:1 Cα pairs.

    Requires equal Cα counts (a residue-by-residue correspondence); raises otherwise. Used for both
    prompt-RMSD (design vs prompt) and scRMSD (design vs refold).
    """
    import biotite.structure as struc

    f = _ca_array(fixed_struct)
    m = _ca_array(mobile_struct)
    if len(f) != len(m):
        raise ValueError(f"ca_rmsd: residue-count mismatch ({len(f)} vs {len(m)}) — no 1:1 correspondence")
    fitted, _ = struc.superimpose(f, m)
    return float(struc.rmsd(f, fitted))


def adherence(design_struct, prompt_struct, *, tm_norm: str = "prompt") -> Adherence:
    """Prompt-adherence of a design vs the prompt's source structure (dev ``05`` §3, SPA headline).

    TM-score is always reported (length-independent); prompt-RMSD is reported only when the residue
    counts match. ``tm_norm`` selects which TM normalization is the primary :attr:`Adherence.tm_score`
    (``"prompt"`` = fraction of the prompt fold reproduced; ``"design"``; or ``"max"``).
    """
    d = _ca_array(design_struct)
    p = _ca_array(prompt_struct)
    tm_d, tm_p = tm_score(d, p)  # chain1 = design, chain2 = prompt
    primary = {"design": tm_d, "prompt": tm_p, "max": max(tm_d, tm_p)}.get(tm_norm)
    if primary is None:
        raise ValueError(f"adherence: unknown tm_norm {tm_norm!r} (expected 'prompt', 'design' or 'max')")
    prompt_rmsd = ca_rmsd(p, d) if len(d) == len(p) else None
    return Adherence(
        tm_score=primary, tm_norm_design=tm_d, tm_norm_prompt=tm_p,
        prompt_rmsd=prompt_rmsd, n_design=len(d), n_prompt=len(p),
    )


# --------------------------------------------------------------------------------------------------
# Designability — best-of-K self-consistency scRMSD (dev 05 §3); refold is a pluggable input
# --------------------------------------------------------------------------------------------------


@runtime_checkable
class Refolder(Protocol):
    """Turns a Stage-2 sequence set into Stage-3 refold structures (the OF3 leg, pluggable).

    OF3 is **not** implemented here (it runs in a separate env, dev ``05`` Stage 3); any object with
    a ``refold(sequence_set) -> list[AtomArray|path]`` satisfies this, letting a driver inject the
    real OF3 refolds (or, in the smoke, a stand-in) without this module importing OpenFold3.
    """

    def refold(self, sequence_set) -> list: ...


def self_consistency(design_struct, refolds) -> SelfConsistency:
    """Best-of-K Cα scRMSD between a design backbone and its K OF3 refolds (dev ``05`` §3).

    ``refolds`` is any iterable of refold structures (``AtomArray`` / path / Design). A refold whose
    Cα count differs from the design's contributes ``nan`` (skipped from the best-of). The minimum
    over the valid refolds is the design's scRMSD.
    """
    d = _ca_array(design_struct)
    per: list[float] = []
    for rf in refolds:
        try:
            r = _ca_array(rf)
            if len(r) != len(d):
                per.append(float("nan"))
                continue
            per.append(ca_rmsd(d, r))
        except Exception:
            per.append(float("nan"))
    valid = [(i, v) for i, v in enumerate(per) if v == v]  # drop NaN
    if not valid:
        return SelfConsistency(scrmsd=float("nan"), best_refold_idx=-1, per_refold=per)
    best_idx, best = min(valid, key=lambda t: t[1])
    return SelfConsistency(scrmsd=best, best_refold_idx=best_idx, per_refold=per)


def is_designable(scrmsd, plddt=None, *, scrmsd_cutoff: float = 2.0, plddt_cutoff: float = 80.0) -> bool:
    """Designability flag: ``scRMSD < scrmsd_cutoff`` (and pLDDT ≥ ``plddt_cutoff`` when supplied)."""
    if scrmsd is None or scrmsd != scrmsd:  # None or NaN
        return False
    ok = float(scrmsd) < float(scrmsd_cutoff)
    if plddt is not None:
        ok = ok and float(plddt) >= float(plddt_cutoff)
    return bool(ok)


# --------------------------------------------------------------------------------------------------
# Per-design scoring (compose adherence + designability into a DesignScore)
# --------------------------------------------------------------------------------------------------


def _design_meta(design) -> tuple[str, str, float, int]:
    """``(name, condition, lambda_scale, n_residues)`` from a Design (sensible fallbacks for a path)."""
    name = getattr(design, "name", None)
    if name is None:
        path = getattr(design, "path", None)
        name = Path(str(path)).stem if path is not None else "design"
    condition = getattr(design, "condition", "unknown")
    lam = float(getattr(design, "lambda_scale", 0.0))
    n_res = getattr(design, "n_residues", None)
    if n_res is None:
        n_res = len(_ca_array(design))
    return str(name), str(condition), lam, int(n_res)


def score_design(design, *, prompt=None, refolds=None, plddt=None, cfg=None, score_cfg=None) -> DesignScore:
    """Score one design: adherence (if ``prompt`` given) + designability (if ``refolds`` given).

    Args:
        design: a :class:`spa.eval.generate.Design` (uses ``.atom_array``), an ``AtomArray``, or a PDB path.
        prompt: the prompt's source structure (AtomArray / path) for adherence, or ``None`` to skip.
        refolds: iterable of OF3 refold structures for designability (best-of-K), or ``None`` to skip.
        plddt: optional OF3 pLDDT for the best refold (gates designability when supplied).
        cfg / score_cfg: thresholds (a Hydra cfg with ``eval.score`` or a :class:`ScoreConfig`).
    """
    sc = score_cfg if isinstance(score_cfg, ScoreConfig) else score_config(cfg)
    name, condition, lam, n_res = _design_meta(design)
    ds = DesignScore(name=name, condition=condition, lambda_scale=lam, n_residues=n_res)

    if prompt is not None:
        adh = adherence(design, prompt, tm_norm=sc.tm_norm)
        ds.tm_score = adh.tm_score
        ds.tm_norm_design = adh.tm_norm_design
        ds.tm_norm_prompt = adh.tm_norm_prompt
        ds.prompt_rmsd = adh.prompt_rmsd

    if refolds is not None:
        res = self_consistency(design, refolds)
        ds.scrmsd = res.scrmsd
        ds.best_refold_idx = res.best_refold_idx
        ds.plddt = plddt
        ds.designable = is_designable(
            res.scrmsd, plddt, scrmsd_cutoff=sc.scrmsd_cutoff, plddt_cutoff=sc.plddt_cutoff
        )
    return ds


# --------------------------------------------------------------------------------------------------
# Diversity (pairwise TM among the designable set) + novelty stub (dev 05 §3)
# --------------------------------------------------------------------------------------------------


def pairwise_tm_diversity(structs) -> float | None:
    """Mean pairwise Cα TM-score over a set of structures (dev ``05`` §3 diversity).

    Returns the mean over all unordered pairs (each pair's max-normalized TM); **lower = more
    diverse**. ``None`` for fewer than two structures (no pair to compare).
    """
    items = list(structs)
    if len(items) < 2:
        return None
    cas = [_ca_array(s) for s in items]
    vals: list[float] = []
    for i in range(len(cas)):
        for j in range(i + 1, len(cas)):
            a, b = tm_score(cas[i], cas[j])
            vals.append(max(a, b))
    return float(mean(vals)) if vals else None


def novelty_vs_pdb(structs, *, foldseek_db=None):
    """Novelty vs PDB (TM-score to the nearest known fold) — **stub** (dev ``05`` §3).

    Foldseek may be absent; rather than block scoring this logs and raises ``NotImplementedError``.
    Callers (e.g. :func:`aggregate`) wrap it best-effort so an absent Foldseek never fails a run.
    """
    raise NotImplementedError(
        "novelty-vs-PDB needs Foldseek (or a TM-score scan vs a reference set), not wired up — "
        "see dev 05_validation_pipeline.md §3; pass eval.score.novelty=false to skip."
    )


# --------------------------------------------------------------------------------------------------
# Aggregation — per (condition, λ): success rate + distributions + Δ(SPA − baseline) (dev 06 §6)
# --------------------------------------------------------------------------------------------------


def _distribution(values) -> Distribution:
    """Summarize a list of metric values, dropping ``None`` and ``NaN``."""
    clean = [float(v) for v in values if v is not None and float(v) == float(v)]
    if not clean:
        return Distribution(n=0)
    return Distribution(n=len(clean), mean=float(mean(clean)), median=float(median(clean)),
                        min=float(min(clean)), max=float(max(clean)))


def aggregate(scores, *, structs_by_name=None, cfg=None, score_cfg=None) -> list[ConditionSummary]:
    """Aggregate per-design :class:`DesignScore`s into per-``(condition, λ)`` :class:`ConditionSummary`.

    Computes the success **rate** (fraction designable), adherence TM/prompt-RMSD and scRMSD
    **distributions**, and — when ``structs_by_name`` maps a design ``name`` to its structure and
    ``score.diversity`` is on — the pairwise-TM **diversity** among that group's designable designs.
    Novelty is left ``None`` (stubbed). Groups are returned sorted by ``(condition, λ)``.
    """
    sc = score_cfg if isinstance(score_cfg, ScoreConfig) else score_config(cfg)
    groups: dict[tuple[str, float], list[DesignScore]] = {}
    for s in scores:
        groups.setdefault((s.condition, float(s.lambda_scale)), []).append(s)

    summaries: list[ConditionSummary] = []
    for (condition, lam), items in sorted(groups.items()):
        scored = [s for s in items if s.designable is not None]
        n_designable = sum(1 for s in scored if s.designable)
        success_rate = (n_designable / len(scored)) if scored else None

        diversity_tm = None
        if sc.diversity and structs_by_name:
            designable_structs = [
                structs_by_name[s.name]
                for s in items
                if s.designable and s.name in structs_by_name
            ]
            if len(designable_structs) >= 2:
                diversity_tm = pairwise_tm_diversity(designable_structs)

        summaries.append(ConditionSummary(
            condition=condition,
            lambda_scale=lam,
            n_designs=len(items),
            n_designable=n_designable,
            success_rate=success_rate,
            tm=_distribution([s.tm_score for s in items]),
            prompt_rmsd=_distribution([s.prompt_rmsd for s in items]),
            scrmsd=_distribution([s.scrmsd for s in items]),
            diversity_tm=diversity_tm,
            novelty=None,
        ))
    return summaries


def delta_vs_baseline(summaries, *, baseline_condition: str = "baseline") -> list[DeltaSummary]:
    """Δ(SPA − baseline) for every non-baseline ``(condition, λ)`` group (dev ``06`` §6 headline).

    The baseline group (``condition == baseline_condition``) is the shared reference; each other
    group's success-rate and mean-TM/mean-prompt-RMSD deltas are reported against it. Returns an
    empty list if no baseline group is present.
    """
    base = next((s for s in summaries if s.condition == baseline_condition), None)
    if base is None:
        return []

    def _sub(a, b):
        return (a - b) if (a is not None and b is not None) else None

    out: list[DeltaSummary] = []
    for s in summaries:
        if s.condition == baseline_condition:
            continue
        out.append(DeltaSummary(
            condition=s.condition,
            lambda_scale=s.lambda_scale,
            d_success_rate=_sub(s.success_rate, base.success_rate),
            d_tm_mean=_sub(s.tm.mean, base.tm.mean),
            d_prompt_rmsd_mean=_sub(s.prompt_rmsd.mean, base.prompt_rmsd.mean),
            spa_success_rate=s.success_rate,
            baseline_success_rate=base.success_rate,
            spa_tm_mean=s.tm.mean,
            baseline_tm_mean=base.tm.mean,
        ))
    return out
