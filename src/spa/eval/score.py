"""Stage 4 of the SPA validation flywheel: score designs (designability + prompt adherence).

Spec: dev ``05_validation_pipeline.md`` §3 ("Metrics") and ``06_experiments.md`` §6 ("Success
criteria"). Consumes Stage-1 :class:`spa.eval.generate.Design` backbones, the Stage-2
:class:`spa.eval.proteinmpnn.SequenceSet` sequences, and the Stage-3 OpenFold3 refolds, and produces
the comparison numbers the poster reports: **designability** (best-of-K self-consistency RMSD) and
**prompt adherence** (does an SPA design honor its structural prompt's fold), aggregated per
``(condition, λ)`` with the headline **Δ(SPA − baseline)**.

Three metric families (the third only for Run-B hard⊕soft motif evals — dev ``14``):

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
- **Motif satisfaction (Run-B hard⊕soft only; dev ``14`` §3)** — when a native motif is scaffolded
  (RFD3 pins it at fixed coords), the Cα **motif-RMSD** over just the motif residues vs the motif's
  source. Two readings: *design-side* (RFD3 backbone vs source — ≈0 by construction, a sanity check the
  pin held, and the headline "hard satisfied, equal across conditions" number) and *refold-side* (the
  OF3 refold vs source — does the designed sequence still realize the motif geometry?). ``None`` for any
  design with no motif supplied, so non-motif evals are unaffected.

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
    "scrmsd_cutoff": 2.0,       # Å — best-of-K Cα scRMSD designability threshold (dev 05 §3 / 06 §6)
    # Atom set for scRMSD. "CA" (default) = this project's historical criterion, paired with the 2.0 Å
    # cutoff above. "N,CA,C" = RFdiffusion3's own criterion, which they pair with 1.5 Å. Set BOTH knobs
    # together when calibrating against their published rates; changing one alone is not comparable.
    "scrmsd_atoms": "CA",
    "plddt_cutoff": 80.0,       # OF3 pLDDT designability gate (applied only when a pLDDT is supplied)
    "tm_norm": "prompt",        # primary TM normalization for adherence: prompt | design | max
    "diversity": True,          # compute pairwise-TM diversity among the designable set
    "novelty": False,           # novelty-vs-PDB (Foldseek) — stub; raises NotImplemented if requested
    "motif_rmsd_cutoff": 1.0,   # Å — Cα RMSD over the motif residues; motif_satisfied iff below (dev 14 §3)
}


@dataclass
class ScoreConfig:
    """Resolved ``eval.score`` knobs (defaults from :data:`DEFAULTS`)."""

    scrmsd_cutoff: float = DEFAULTS["scrmsd_cutoff"]
    scrmsd_atoms: str = DEFAULTS["scrmsd_atoms"]
    plddt_cutoff: float = DEFAULTS["plddt_cutoff"]
    tm_norm: str = DEFAULTS["tm_norm"]
    diversity: bool = DEFAULTS["diversity"]
    novelty: bool = DEFAULTS["novelty"]
    motif_rmsd_cutoff: float = DEFAULTS["motif_rmsd_cutoff"]


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
        scrmsd_atoms=str(sc.get("scrmsd_atoms", DEFAULTS["scrmsd_atoms"])),
        plddt_cutoff=float(sc.get("plddt_cutoff", DEFAULTS["plddt_cutoff"])),
        tm_norm=str(sc.get("tm_norm", DEFAULTS["tm_norm"])),
        diversity=bool(sc.get("diversity", DEFAULTS["diversity"])),
        novelty=bool(sc.get("novelty", DEFAULTS["novelty"])),
        motif_rmsd_cutoff=float(sc.get("motif_rmsd_cutoff", DEFAULTS["motif_rmsd_cutoff"])),
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
    # refold-side ADHERENCE (dev plan/67 §7.8): the same TM as above, recomputed on the best refold
    # rather than the design. Answers "does the fold-steer survive inverse folding?", which the
    # design-side number cannot: it is read before ProteinMPNN ever runs. The refold-side motif
    # metric below has always existed; this is its missing partner on the steering channel, and its
    # absence meant the two conditioning channels were reported at different stages.
    tm_score_refold: float | None = None
    tm_norm_design_refold: float | None = None
    tm_norm_prompt_refold: float | None = None
    # hard-motif satisfaction (Run-B hard⊕soft; dev 14 §3-§4) — all None when no motif was supplied
    motif_rmsd: float | None = None         # design-side: design backbone vs motif source, over the motif residues
    motif_rmsd_refold: float | None = None  # refold-side: best OF3 refold vs motif source (motif survival)
    motif_satisfied: bool | None = None     # design-side motif_rmsd < motif_rmsd_cutoff (the hard pin held)


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
    tm_refold: Distribution = field(default_factory=lambda: Distribution(n=0))          # refold-side adherence TM (steer survival through OF3)
    motif_rmsd: Distribution = field(default_factory=lambda: Distribution(n=0))         # design-side motif Cα-RMSD (≈0 if the hard pin held; dev 14)
    motif_rmsd_refold: Distribution = field(default_factory=lambda: Distribution(n=0))  # refold-side motif Cα-RMSD (motif survival through OF3)
    motif_satisfied_rate: float | None = None  # fraction with design-side motif_rmsd < cutoff (None if no motif scored)
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
    d_motif_rmsd_mean: float | None         # Δ design-side motif-RMSD — expect ≈0 (the hard pin is SPA-independent)
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


# --------------------------------------------------------------------------------------------------
# Selection helpers for the multimer / protein-ligand runs (dev ``90`` §5.0a, items P1 + P3).
#
# ⛔ DELIBERATELY NOT FOLDED INTO ``_ca_array``, and please do not "tidy" them into it.
# ``_ca_array`` is the single entry point for EVERY metric in this module (tm_score, ca_rmsd,
# adherence, motif_rmsd, source_positions, pairwise_tm_diversity, self_consistency, the design-index
# helpers) and for ``scripts/eval/domain_split.py``. Every historical number in ``docs/results/`` was
# produced through it. Adding a filter there, even a default-off one, puts a branch in the hottest
# correctness path in the project for the benefit of two experiments that have not run yet.
#
# Instead these RETURN A FILTERED AtomArray, and ``_as_struct`` passes an AtomArray straight through
# (see above), so a caller writes ``adherence(polymer_only(design), prompt)`` and every existing
# function is reached completely unmodified. Nothing on any historical path changes by construction.
# --------------------------------------------------------------------------------------------------


def polymer_only(struct):
    """Drop non-amino-acid residues (ligands, ions, waters), keeping MODIFIED residues.

    ⚠️ **The obvious implementation is wrong and was measured to be wrong.** Filtering on biotite's
    ``hetero`` flag deletes **selenomethionine (MSE) and phosphoserine (SEP)**, which are genuine
    residues carrying genuine Cα atoms and are common in crystal structures. Verified against the
    installed biotite: on a array of GLY/MSE/SEP plus a Ca²⁺ ion, ``~hetero`` keeps **only GLY**.
    ``filter_amino_acids`` keeps GLY, MSE and SEP and drops the ion, which is the intended behaviour.

    ⭐ **Why this exists at all.** ``_ca_array`` selects on ``atom_name == "CA"``, and a **calcium ion's
    atom name is literally** ``CA``. So a Ca²⁺ cofactor is silently counted as a residue Cα by every
    backbone metric, shifting each later design index by one (dev ``90`` §3.1). Any run whose input
    carries a ligand or an ion should be filtered through here first.

    Args:
        struct: anything :func:`_as_struct` accepts.

    Returns:
        ``AtomArray`` of amino-acid residues only. Pass it straight to any scorer in this module.
    """
    from biotite.structure import filter_amino_acids

    arr = _as_struct(struct)
    return arr[filter_amino_acids(arr)]


def select_chains(struct, chains):
    """Restrict a structure to the named chain(s), so per-chain scoring needs no new scorer.

    ⭐ **Why this exists.** ``_ca_array`` groups nothing by chain: a complex becomes one concatenated
    Cα array, and ``tmtools.tm_align`` then scores two chains as a single sequential polypeptide,
    silently (dev ``90`` §2.1, item M3). ``tmtools`` exposes no chain argument and no MM-align, so
    chain-permutation-aware TM is not reachable through the current dependency. **Scoring one chain at
    a time is therefore the correct move, not a workaround**: for the fixed-partner binder shape the
    question is about the steered chain against its prompt, which is exactly a single-chain comparison.

    Args:
        struct: anything :func:`_as_struct` accepts.
        chains: one chain id, or an iterable of them.

    Returns:
        ``AtomArray`` restricted to those chains.

    Raises:
        ValueError: if any requested chain is absent. A silently empty selection would score a design
            that does not contain the chain as perfect.
    """
    import numpy as np

    arr = _as_struct(struct)
    want = [str(chains)] if isinstance(chains, str) else [str(c) for c in chains]
    present = set(map(str, arr.chain_id))
    missing = [c for c in want if c not in present]
    if missing:
        raise ValueError(
            f"select_chains: chain(s) {missing} absent; present chains are {sorted(present)}"
        )
    return arr[np.isin(arr.chain_id, want)]


def _backbone_array(struct):
    """The N, CA, C backbone ``AtomArray``, ordered canonically within each residue.

    RFdiffusion3 scores designability over **N, CA, C at <= 1.5 A**, whereas this project's default is
    **CA only at < 2.0 A** (a looser criterion on a smaller atom set). Comparing our absolute rates to
    RFD3's published numbers requires scoring on their atom set, which is what this provides.

    Residues missing any of the three are dropped WHOLE (all-or-nothing), so the design and refold keep
    a 1:1 atom correspondence; a partial residue would silently misalign the Kabsch pairing.
    """
    import numpy as np

    arr = _as_struct(struct)
    keep = np.isin(arr.atom_name, ("N", "CA", "C"))
    bb = arr[keep]
    order = {"N": 0, "CA": 1, "C": 2}
    out_idx: list[int] = []
    # group by (chain, res_id, insertion code) so residues stay contiguous and correctly ordered
    seen: dict = {}
    for i in range(len(bb)):
        key = (str(bb.chain_id[i]), int(bb.res_id[i]),
               str(bb.ins_code[i]) if hasattr(bb, "ins_code") else "")
        seen.setdefault(key, {})[str(bb.atom_name[i])] = i
    for key in sorted(seen, key=lambda k: (k[0], k[1], k[2])):
        atoms = seen[key]
        if len(atoms) != 3:      # incomplete backbone -> drop the whole residue
            continue
        out_idx.extend(atoms[a] for a in sorted(atoms, key=lambda a: order[a]))
    return bb[out_idx]


def backbone_rmsd(fixed_struct, mobile_struct) -> float:
    """N/CA/C Kabsch-superposed RMSD (A), RFdiffusion3's designability atom set.

    Same contract as :func:`ca_rmsd` but over three backbone atoms per residue instead of one.
    """
    import biotite.structure as struc

    f = _backbone_array(fixed_struct)
    m = _backbone_array(mobile_struct)
    if len(f) != len(m):
        raise ValueError(
            f"backbone_rmsd: atom-count mismatch ({len(f)} vs {len(m)}) -- no 1:1 correspondence")
    fitted, _ = struc.superimpose(f, m)
    return float(struc.rmsd(f, fitted))


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
# Motif satisfaction — Cα RMSD over the motif residues only (Run-B hard⊕soft; dev 14 §3)
# --------------------------------------------------------------------------------------------------


def motif_rmsd(design_struct, motif_source, motif_residues, *, source_residues=None) -> float:
    """Cα Kabsch RMSD (Å) over the **motif residues only** — "is the hard constraint satisfied?" (dev ``14`` §3).

    Subset both structures to the motif residues' Cα, Kabsch-superpose **over that subset alone**, and
    return its RMSD — so the (diffused) scaffold is irrelevant to the number. ``motif_residues`` are
    **0-based positional indices** into the per-residue Cα array (residues in chain order) of
    ``design_struct``; ``source_residues`` indexes ``motif_source`` and defaults to ``motif_residues``
    (the design-aligned self-prompt case, where design and motif source share length + indexing —
    dev ``14`` §0). Used two ways: *design-side* (``design_struct`` = the RFD3 backbone → ≈0, a sanity
    check the pin held) and *refold-side* (``design_struct`` = the OF3 refold → does the designed
    sequence still realize the motif geometry?).

    Raises ``ValueError`` on an empty/mismatched/out-of-range index set (no silent wrong answer).
    """
    di = list(motif_residues)
    si = list(motif_residues if source_residues is None else source_residues)
    if not di:
        raise ValueError("motif_rmsd: empty motif_residues")
    if len(di) != len(si):
        raise ValueError(f"motif_rmsd: index-count mismatch ({len(di)} design vs {len(si)} source)")
    d = _ca_array(design_struct)
    s = _ca_array(motif_source)
    for arr, idx, who in ((d, di, "design"), (s, si, "source")):
        if min(idx) < 0 or max(idx) >= len(arr):
            raise ValueError(
                f"motif_rmsd: {who} motif index out of range (n={len(arr)}, idx∈[{min(idx)},{max(idx)}])"
            )
    return ca_rmsd(s[si], d[di])


def _residue_atom_coords(struct, chain, resid, atom_names):
    """Coords ``[k, 3]`` (float64) for the named atoms of residue ``(chain, resid)``, in ``atom_names`` order.

    Raises ``ValueError`` if the residue or any requested atom is absent — a missing atom must surface (the
    caller logs + drops the design), never a silent wrong number.
    """
    import numpy as np

    arr = _as_struct(struct)
    sub = arr[(arr.chain_id == str(chain)) & (arr.res_id == int(resid))]
    if len(sub) == 0:
        raise ValueError(f"motif atom: residue {chain}{resid} absent from structure")
    out = []
    for name in atom_names:
        hit = sub[sub.atom_name == str(name)]
        if len(hit) == 0:
            raise ValueError(f"motif atom: {chain}{resid}:{name} absent")
        out.append(hit.coord[0])
    return np.asarray(out, dtype="float64")


def _design_residue_key(design_struct, design_idx):
    """``(chain, res_id)`` of the design's ``design_idx``-th residue (0-based, Cα/file order)."""
    ca = _ca_array(design_struct)
    if design_idx < 0 or design_idx >= len(ca):
        raise ValueError(f"motif atom: design index {design_idx} out of range (n={len(ca)})")
    return str(ca.chain_id[design_idx]), int(ca.res_id[design_idx])


def _kabsch_rmsd(P, Q) -> float:
    """RMSD (Å) after optimal rigid superposition of ``Q`` onto ``P`` (both ``[N, 3]`` float64)."""
    import numpy as np

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Qc.T @ Pc
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T          # rotation mapping Qc onto Pc
    Qr = Qc @ R.T
    return float(np.sqrt(np.mean(np.sum((Qr - Pc) ** 2, axis=1))))


def motif_atom_rmsd(design_struct, motif_source, atom_spec) -> float:
    """RMSD (Å) over the **fixed motif atoms** (tip/all-atom), Kabsch-superposed over just those atoms.

    The right "is the atomic constraint satisfied?" metric when only sidechain *tip* atoms (not Cα) are
    pinned: a tip-only motif does NOT hold Cα, so :func:`motif_rmsd` (Cα) would spuriously report large
    (dev ``26`` §8.6). Measures the *same* atoms RFD3 fixed — matching the paper's AME motif all-atom RMSD
    (§8.1). ``atom_spec`` is the list from :func:`spa.eval.generate.motif_atom_spec`: records
    ``{design_idx (0-based positional), chain, resid, atoms:[names]}``. Design atoms are located by
    positional residue index (:func:`_design_residue_key`), source atoms by author ``(chain, resid)`` — so a
    non-self-aligned source PDB (the enzyme H) is scored against the right atoms. Design-side (backbone) it
    checks the pin held; refold-side it checks the redesigned sequence still realizes the motif geometry.
    Raises ``ValueError`` on an empty spec or a missing/mismatched atom (never a silent wrong answer).
    """
    import numpy as np

    if not atom_spec:
        raise ValueError("motif_atom_rmsd: empty atom_spec")
    d_blocks, s_blocks = [], []
    for rec in atom_spec:
        ch_d, rid_d = _design_residue_key(design_struct, int(rec["design_idx"]))
        d_blocks.append(_residue_atom_coords(design_struct, ch_d, rid_d, rec["atoms"]))
        s_blocks.append(_residue_atom_coords(motif_source, rec["chain"], rec["resid"], rec["atoms"]))
    D = np.concatenate(d_blocks, axis=0)
    S = np.concatenate(s_blocks, axis=0)
    if len(D) != len(S) or len(D) < 3:
        raise ValueError(f"motif_atom_rmsd: matched-atom count {len(D)} vs {len(S)} (need equal, ≥3)")
    return _kabsch_rmsd(S, D)                         # superpose the design's fixed atoms onto the source's


def source_positions(source_struct, chain_resids) -> list[int]:
    """Positional Cα indices in ``source_struct`` for each ``(chain_id, author_resid)`` ref (dev ``14`` §3).

    Builds ``{(str(chain), int(resid)): position}`` over the source's per-residue Cα array (residues in
    file order) and returns the position for each requested ``(chain, resid)``. This is the bridge from a
    contig's **author-numbered** motif residues (e.g. 1CTT ``A102``) to ``motif_rmsd``'s **positional**
    source indices — so a non-self-aligned / non-1-numbered / multi-chain source PDB is scored against the
    *right* residues, not the design-frame positions (review #1). Raises ``ValueError`` naming any ref
    absent from the source.
    """
    ca = _ca_array(source_struct)
    index = {(str(c), int(r)): pos for pos, (c, r) in enumerate(zip(ca.chain_id, ca.res_id))}
    out, missing = [], []
    for chain, resid in chain_resids:
        key = (str(chain), int(resid))
        (out.append(index[key]) if key in index else missing.append(key))
    if missing:
        raise ValueError(f"source_positions: motif refs not found in source: {missing}")
    return out


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


def self_consistency(design_struct, refolds, *, atoms: str = "CA") -> SelfConsistency:
    """Best-of-K scRMSD between a design backbone and its K OF3 refolds (dev ``05`` §3).

    ``atoms`` selects the atom set: ``"CA"`` (default, this project's historical criterion, normally
    paired with a 2.0 Å cutoff) or ``"N,CA,C"`` (RFdiffusion3's own, paired with 1.5 Å). The default
    keeps every previously computed number bit-identical.

    ``refolds`` is any iterable of refold structures (``AtomArray`` / path / Design). A refold whose
    Cα count differs from the design's contributes ``nan`` (skipped from the best-of). The minimum
    over the valid refolds is the design's scRMSD.
    """
    use_bb = str(atoms).upper().replace(" ", "") in ("N,CA,C", "BACKBONE", "BB")
    _arr = _backbone_array if use_bb else _ca_array
    _rmsd = backbone_rmsd if use_bb else ca_rmsd
    d = _arr(design_struct)
    per: list[float] = []
    for rf in refolds:
        try:
            r = _arr(rf)
            if len(r) != len(d):
                per.append(float("nan"))
                continue
            per.append(_rmsd(d, r))
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


def score_design(design, *, prompt=None, refolds=None, plddt=None, motif=None, cfg=None, score_cfg=None) -> DesignScore:
    """Score one design: adherence (if ``prompt``) + designability (if ``refolds``) + motif (if ``motif``).

    Args:
        design: a :class:`spa.eval.generate.Design` (uses ``.atom_array``), an ``AtomArray``, or a PDB path.
        prompt: the prompt's source structure (AtomArray / path) for adherence, or ``None`` to skip.
        refolds: iterable of OF3 refold structures for designability (best-of-K), or ``None`` to skip.
        plddt: optional OF3 pLDDT for the best refold (gates designability when supplied).
        motif: for the Run-B hard⊕soft motif-RMSD (dev ``14`` §3), or ``None`` (default — non-motif evals
            untouched). Either a **2-tuple** ``(source_struct, residues)`` — self-aligned, source indexed by
            the same design-frame ``residues`` — or a **3-tuple** ``(source_struct, design_residues,
            source_residues)`` for a non-self-aligned source (e.g. 1CTT; the flywheel builds it via
            :func:`source_positions`). Fills design-side ``motif_rmsd`` + ``motif_satisfied``, and
            ``motif_rmsd_refold`` when ``refolds`` is also given.
        cfg / score_cfg: thresholds (a Hydra cfg with ``eval.score`` or a :class:`ScoreConfig`).
    """
    sc = score_cfg if isinstance(score_cfg, ScoreConfig) else score_config(cfg)
    refolds = list(refolds) if refolds is not None else None  # may be indexed twice (scRMSD + refold-side motif)
    name, condition, lam, n_res = _design_meta(design)
    ds = DesignScore(name=name, condition=condition, lambda_scale=lam, n_residues=n_res)

    if prompt is not None:
        adh = adherence(design, prompt, tm_norm=sc.tm_norm)
        ds.tm_score = adh.tm_score
        ds.tm_norm_design = adh.tm_norm_design
        ds.tm_norm_prompt = adh.tm_norm_prompt
        ds.prompt_rmsd = adh.prompt_rmsd

    if refolds is not None:
        res = self_consistency(design, refolds, atoms=sc.scrmsd_atoms)
        ds.scrmsd = res.scrmsd
        ds.best_refold_idx = res.best_refold_idx
        ds.plddt = plddt
        ds.designable = is_designable(
            res.scrmsd, plddt, scrmsd_cutoff=sc.scrmsd_cutoff, plddt_cutoff=sc.plddt_cutoff
        )
        # Refold-side adherence (dev plan/67 §7.8). Same `adherence()` and the SAME `tm_norm`, so the
        # pair (tm_score, tm_score_refold) is a like-for-like comparison and their difference is the
        # steer lost through the round trip. Scored on the scRMSD-selected best refold, matching
        # `motif_rmsd_refold` below; note that refold is chosen for reproducing the BACKBONE, not for
        # matching the target, which is a property to state rather than to correct here.
        # ⛔ Whole-design only. Regional steering needs slice TM, max-normalisation and a refolded
        # baseline arm; see plan/67 §7.8a before extending this to a per-region number.
        if prompt is not None and res.best_refold_idx is not None and res.best_refold_idx >= 0:
            try:
                adh_r = adherence(refolds[res.best_refold_idx], prompt, tm_norm=sc.tm_norm)
                ds.tm_score_refold = adh_r.tm_score
                ds.tm_norm_design_refold = adh_r.tm_norm_design
                ds.tm_norm_prompt_refold = adh_r.tm_norm_prompt
            except Exception as e:  # a length-mismatched or unparsable refold must not abort the run
                print(f"[score] refold-side adherence failed for {name}: {e}")

    if motif is not None:
        atom_spec = None
        if isinstance(motif, dict):                  # atomic (tip-atom) motif payload (dev 26 §8.6)
            source = motif["source"]
            design_res = motif.get("design_residues")
            source_res = motif.get("source_residues")
            atom_spec = motif.get("atom_spec")
        elif len(motif) == 3:                        # (source, design_residues, source_residues)
            source, design_res, source_res = motif
        else:                                        # (source, residues) — self-aligned (source_res defaults)
            source, design_res = motif
            source_res = None

        def _mrmsd(struct):                          # tip/all-atom RMSD when atom_spec given, else Cα (§4)
            if atom_spec:
                return motif_atom_rmsd(struct, source, atom_spec)
            return motif_rmsd(struct, source, design_res, source_residues=source_res)

        try:                                         # review #5: a bad source index must not abort the run
            ds.motif_rmsd = _mrmsd(design)
            ds.motif_satisfied = bool(ds.motif_rmsd < sc.motif_rmsd_cutoff)
        except Exception as e:
            print(f"[score] design-side motif_rmsd failed for {name}: {e}")
            ds.motif_rmsd, ds.motif_satisfied = None, None
        if refolds is not None and ds.best_refold_idx is not None and ds.best_refold_idx >= 0:
            try:  # the best refold may differ in length from the motif source (rare) -> leave None
                ds.motif_rmsd_refold = _mrmsd(refolds[ds.best_refold_idx])
            except Exception as e:
                print(f"[score] refold-side motif_rmsd failed for {name}: {e}")  # review #7: log, don't swallow
                ds.motif_rmsd_refold = None
    return ds


# --------------------------------------------------------------------------------------------------
# Diversity (pairwise TM among the designable set) + novelty stub (dev 05 §3)
# --------------------------------------------------------------------------------------------------
# Tier-1 structural metrics for the multimer / protein-ligand runs (dev ``90`` §5.0a, items P4 + P5).
#
# ⭐ These are deliberately REFOLD-FREE: both are geometry on the generated backbone, so they cost
# nothing beyond generation and can gate the expensive designability tier (dev ``90`` §5.1). Neither
# is called by any existing path — every function below is additive, so all historical numbers are
# byte-identical.
# --------------------------------------------------------------------------------------------------


@dataclass
class LigandPocket:
    """Ligand-pocket geometry for one design (dev ``90`` §5.0a P4; Run 1's tier-1 metric).

    RFdiffusion3 holds a supplied ligand's coordinates fixed, so the ligand does not move between the
    reference and the design: what can move is the protein around it. Both fields below therefore measure
    the *protein's* placement relative to a common ligand.

    Attributes:
        n_clash: count of **non-motif backbone atoms (N, CA, C) within ``clash_cutoff`` of the nearest
            ligand atom**. This is RFdiffusion3's own AME-benchmark clash definition (RFD3 supplementary
            §3.5), reused verbatim so the number is comparable to a published criterion rather than
            invented here.
        min_dist: smallest distance (Å) from any protein backbone atom to any ligand atom.
        motif_dists: per-motif-atom distance (Å) to the **nearest ligand atom**, in the order the atoms
            were given. Empty when no motif atoms were supplied.
        motif_dist_rmsd: RMSD (Å) between ``motif_dists`` and the same distances measured on a reference
            structure, or ``None`` when no reference was given. **This is the pocket-preservation
            number**: 0 means the motif sits exactly as far from the ligand as it natively does.
        n_ligand_atoms: ligand atom count actually found.
    """

    n_clash: int
    min_dist: float
    motif_dists: list[float] = field(default_factory=list)
    motif_dist_rmsd: float | None = None
    n_ligand_atoms: int = 0


@dataclass
class InterfaceGeometry:
    """Inter-chain interface geometry for one design (dev ``90`` §5.0a P5; Run 2's tier-1 metric).

    Written for the **fixed-partner binder** shape (dev ``90`` §5.2): the partner chain is revealed from
    the input and held by RFdiffusion3, so it is identical between reference and design, and the steered
    chain is the only thing that can move. That is what makes ``rmsd`` well defined without a refold.

    Attributes:
        n_contacts: count of cross-chain Cα pairs within ``cutoff`` Å.
        n_interface_a: residues of the steered chain participating in at least one contact.
        n_interface_b: residues of the partner chain participating in at least one contact.
        rmsd: Cα RMSD (Å) of the **steered chain's interface residues**, after superposing on the
            **partner** chain, or ``None`` when no reference was given. Superposing on the partner is
            what makes this measure interface displacement rather than whole-complex drift.
        frac_retained: ``n_contacts`` as a fraction of the reference's, or ``None`` without a reference.
            **This is the headline preservation number**; 1.0 means the interface is fully retained.
    """

    n_contacts: int
    n_interface_a: int
    n_interface_b: int
    rmsd: float | None = None
    frac_retained: float | None = None


def _ligand_atoms(struct, ligand_names):
    """Atoms of the named ligand residues. Raises if a requested name is absent (never silently empty)."""
    import numpy as np

    arr = _as_struct(struct)
    names = [str(n).strip() for n in (
        ligand_names.split(",") if isinstance(ligand_names, str) else ligand_names) if str(n).strip()]
    if not names:
        raise ValueError("ligand_pocket: no ligand residue names given")
    mask = np.zeros(len(arr), dtype=bool)
    for name in names:
        hit = arr.res_name == name
        if not hit.any():
            raise ValueError(
                f"ligand_pocket: ligand {name!r} absent from structure "
                f"(present res_names include {sorted(set(map(str, arr.res_name)))[:12]})"
            )
        mask |= hit
    return arr[mask]


def ligand_pocket(design_struct, ligand_names, *, motif_atoms=None, reference=None,
                  clash_cutoff: float = 1.5) -> LigandPocket:
    """Ligand-pocket geometry for one design, refold-free (dev ``90`` §5.0a P4).

    Args:
        design_struct: the generated structure, ligand included.
        ligand_names: residue name(s) of the ligand, e.g. ``"IAI"`` or ``"NAI,ACT"`` (the same string
            RFdiffusion3's ``ligand`` spec key takes), or a list of names.
        motif_atoms: optional ``[(chain, resid, atom_name), ...]`` for the pinned motif atoms whose
            distance to the ligand is the pocket-preservation signal. Typically the same atoms
            ``eval.motif.fixed_atoms`` pinned, so the metric scores what the model actually held.
        reference: optional structure (the native holo complex) to compare ``motif_dists`` against.
            Must expose the same motif atoms and the same ligand.
        clash_cutoff: Å, RFdiffusion3's AME value (1.5) by default.

    Returns:
        :class:`LigandPocket`.

    Raises:
        ValueError: if the ligand or any requested motif atom is absent. A missing atom must surface;
            a silently empty selection would score a broken design as perfect.
    """
    import numpy as np

    lig = _ligand_atoms(design_struct, ligand_names)
    lig_xyz = np.asarray(lig.coord, dtype="float64")

    bb = _as_struct(design_struct)
    bb = bb[np.isin(bb.atom_name, ("N", "CA", "C"))]
    bb_xyz = np.asarray(bb.coord, dtype="float64")

    if len(bb_xyz) == 0 or len(lig_xyz) == 0:
        raise ValueError("ligand_pocket: design has no backbone atoms or no ligand atoms")

    d = np.linalg.norm(bb_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1)   # [n_bb, n_lig]
    per_bb_min = d.min(axis=1)

    motif_keys = set()
    if motif_atoms:
        motif_keys = {(str(c), int(r)) for (c, r, _a) in motif_atoms}
    if motif_keys:
        is_motif = np.array([(str(c), int(r)) in motif_keys
                             for c, r in zip(bb.chain_id, bb.res_id)], dtype=bool)
    else:
        is_motif = np.zeros(len(bb), dtype=bool)

    n_clash = int((per_bb_min[~is_motif] < float(clash_cutoff)).sum())

    def _motif_dists(struct):
        lg = np.asarray(_ligand_atoms(struct, ligand_names).coord, dtype="float64")
        out = []
        for chain, resid, atom in motif_atoms:
            xyz = _residue_atom_coords(struct, chain, resid, [atom])[0]
            out.append(float(np.linalg.norm(lg - xyz[None, :], axis=-1).min()))
        return out

    dists = _motif_dists(design_struct) if motif_atoms else []
    rmsd = None
    if motif_atoms and reference is not None:
        ref = _motif_dists(reference)
        a, b = np.asarray(dists), np.asarray(ref)
        rmsd = float(np.sqrt(((a - b) ** 2).mean()))

    return LigandPocket(
        n_clash=n_clash,
        min_dist=float(per_bb_min.min()),
        motif_dists=dists,
        motif_dist_rmsd=rmsd,
        n_ligand_atoms=int(len(lig_xyz)),
    )


def interface_geometry(design_struct, chain_a, chain_b, *, reference=None,
                       cutoff: float = 8.0) -> InterfaceGeometry:
    """Inter-chain interface geometry for one design, refold-free (dev ``90`` §5.0a P5).

    Args:
        design_struct: the generated complex.
        chain_a: the **steered** chain id (the one SPA conditions).
        chain_b: the **partner** chain id (revealed from the input, held fixed by RFdiffusion3).
        reference: optional structure to compare against, normally the λ=0 design from the same seed.
            Supplies ``frac_retained`` and ``rmsd``.
        cutoff: Å for a cross-chain Cα contact; 8.0 matches ``scripts/eval/domain_split.py``, which is
            the only contact convention already in this repo.

    Returns:
        :class:`InterfaceGeometry`.

    Raises:
        ValueError: if either chain is absent, or if a reference is given whose partner chain does not
            match the design's residue-for-residue. That mismatch means the partner was **not** held
            fixed, which invalidates the superposition this metric depends on.
    """
    import numpy as np

    def _chains(struct):
        ca = _ca_array(struct)
        a = ca[ca.chain_id == str(chain_a)]
        b = ca[ca.chain_id == str(chain_b)]
        if len(a) == 0 or len(b) == 0:
            raise ValueError(
                f"interface_geometry: chain {chain_a!r} has {len(a)} Cα and chain {chain_b!r} has "
                f"{len(b)}; present chains are {sorted(set(map(str, ca.chain_id)))}"
            )
        return a, b

    def _contacts(struct):
        a, b = _chains(struct)
        xa = np.asarray(a.coord, dtype="float64")
        xb = np.asarray(b.coord, dtype="float64")
        d = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
        hit = d < float(cutoff)
        return a, b, hit

    a, b, hit = _contacts(design_struct)
    n_contacts = int(hit.sum())
    iface_a = np.flatnonzero(hit.any(axis=1))
    iface_b = np.flatnonzero(hit.any(axis=0))

    rmsd = None
    frac = None
    if reference is not None:
        ra, rb, rhit = _contacts(reference)
        if len(rb) != len(b):
            raise ValueError(
                f"interface_geometry: partner chain {chain_b!r} has {len(b)} Cα in the design and "
                f"{len(rb)} in the reference. The partner must be held fixed for this metric to mean "
                "anything."
            )
        n_ref = int(rhit.sum())
        frac = (float(n_contacts) / n_ref) if n_ref else None
        if len(ra) == len(a) and len(iface_a) >= 3:
            # Superpose on the PARTNER, then measure the steered chain's interface residues, so the
            # number is interface displacement rather than whole-complex drift.
            xb_d = np.asarray(b.coord, dtype="float64")
            xb_r = np.asarray(rb.coord, dtype="float64")
            cd, cr = xb_d.mean(axis=0), xb_r.mean(axis=0)
            u, _, vt = np.linalg.svd((xb_d - cd).T @ (xb_r - cr))
            sign = np.sign(np.linalg.det(u @ vt))
            rot = u @ np.diag([1.0, 1.0, sign]) @ vt
            xa_d = (np.asarray(a.coord, dtype="float64") - cd) @ rot
            xa_r = np.asarray(ra.coord, dtype="float64") - cr
            diff = xa_d[iface_a] - xa_r[iface_a]
            rmsd = float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    return InterfaceGeometry(
        n_contacts=n_contacts,
        n_interface_a=int(len(iface_a)),
        n_interface_b=int(len(iface_b)),
        rmsd=rmsd,
        frac_retained=frac,
    )


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

        # review #12: derive the satisfied-rate from motif_rmsd vs the cutoff (not the persisted per-design
        # bool), so re-aggregating an existing scores set with a different cutoff stays self-consistent.
        motif_scored = [s for s in items if s.motif_rmsd is not None]
        motif_satisfied_rate = (
            sum(1 for s in motif_scored if s.motif_rmsd < sc.motif_rmsd_cutoff) / len(motif_scored)
            if motif_scored else None
        )

        summaries.append(ConditionSummary(
            condition=condition,
            lambda_scale=lam,
            n_designs=len(items),
            n_designable=n_designable,
            success_rate=success_rate,
            tm=_distribution([s.tm_score for s in items]),
            prompt_rmsd=_distribution([s.prompt_rmsd for s in items]),
            scrmsd=_distribution([s.scrmsd for s in items]),
            motif_rmsd=_distribution([s.motif_rmsd for s in items]),
            motif_rmsd_refold=_distribution([s.motif_rmsd_refold for s in items]),
            tm_refold=_distribution([s.tm_score_refold for s in items]),
            motif_satisfied_rate=motif_satisfied_rate,
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
            d_motif_rmsd_mean=_sub(s.motif_rmsd.mean, base.motif_rmsd.mean),
            spa_success_rate=s.success_rate,
            baseline_success_rate=base.success_rate,
            spa_tm_mean=s.tm.mean,
            baseline_tm_mean=base.tm.mean,
        ))
    return out
