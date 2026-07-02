"""Multi-granularity sub-region sampler for the ``multigranularity`` ("editing") curriculum.

Spec: dev ``17_multigranularity_editing_build.md`` §3–§5; rationale dev ``16`` §9.5/§9.6/§9.7.

Per training example we sample a **granularity** ``g ∈ {global, segment, domain}`` (default mix
0.4 / 0.4 / 0.2) and select a contiguous sub-region ``S ⊆ [0, N)`` of the structure's ``N`` residues.
The cached full-structure ESM3 prompt ``[N,1536]`` is then **masked to the non-S rows** (slice-via-mask,
``17`` §4): the returned boolean *key-padding mask* is ``True`` at rows ``∉ S``, so SPA's cross-attention
attends only to S's rows. This reuses the exact Run-B ``prompt_mask`` machinery
(:meth:`~spa.data.dataset.CDDBPromptDataset._motif_prompt_mask` → ``harness.set_prompt``); the only
difference is *which* rows are masked (non-S here vs the native-motif rows in Run B).

**Framing (``17`` §1 lock).** This is the EDITING use case — condition on a sub-region *of an existing
structure*, training AND inference on the sliced prompt consistently. It is NOT isolated-motif
scaffolding (that would need the paid F-free S-cache; ``17`` §1).

**Variant scope (``17`` §5).** N×1536 (``IdentityProjector``, ``M=N``) ONLY: the mask cleanly restricts
attention to S. Pooled variants (1×1536 mean-pool, 1×32 CLSS) mean-pool the prompt *before* projection
(``projectors.py``), so a key-padding mask on attention does **not** stop the pool from averaging non-S
rows — they need mask-aware pooling or true-slice (``17`` §5, deferred). ``set_prompt`` silently drops a
per-residue mask when the projector changes the token count, so a pooled variant degrades to *global*
(no error) rather than slicing — this sampler assumes the identity projector.

The three granularities:
  - ``global``  → S = all residues → returns ``None`` (no mask; ≡ full-prompt / Run-A behaviour).
  - ``segment`` → S = a contiguous window ``[start, start+len)``, ``len ~ U(min_seg, N)``.
  - ``domain``  → S = one domain of the cleanest contiguous 2-domain split (Cα contact map, cf.
    ``scripts/eval/domain_split.py``); falls back to ``segment`` when there is no clean 2-domain split
    (too short, contact-map degenerate, score above ``domain_score_max``, or Cα-count ≠ prompt N).

All randomness goes through ``rng`` (a ``numpy.random.RandomState`` or the ``numpy.random`` module), so
tests can seed it; production uses the module (per-worker seeded, like ``CoupledIslandCondition``).
"""

from __future__ import annotations

import numpy as np

GRANULARITIES = ("global", "segment", "domain")


def _rng(rng):
    return np.random if rng is None else rng


def sample_granularity(weights, rng=None) -> str:
    """Pick a granularity name from ``weights`` (a ``{name: w}`` mapping — plain dict or OmegaConf
    ``DictConfig`` — or a 3-seq over :data:`GRANULARITIES`)."""
    if hasattr(weights, "keys"):  # dict / OmegaConf DictConfig -> read by granularity name
        w = np.array([float(weights.get(g, 0.0)) for g in GRANULARITIES], dtype=float)
    else:                          # positional sequence aligned to GRANULARITIES
        w = np.array([float(x) for x in weights], dtype=float)
    if w.sum() <= 0:
        raise ValueError(f"granularity weights must be positive; got {weights!r}")
    return str(_rng(rng).choice(GRANULARITIES, p=w / w.sum()))


def segment_region(n: int, min_seg: int, rng=None) -> tuple[int, int]:
    """A contiguous window ``[start, end)`` with ``end-start ~ U(min_seg, n)`` (end exclusive).

    Returns ``(0, n)`` (⇒ the whole structure, i.e. global) when ``n <= min_seg`` or the sampled length
    reaches ``n``. Length is always ``>= 1`` so S is never empty.
    """
    r = _rng(rng)
    if n <= min_seg:
        return 0, n
    seg_len = int(r.randint(min_seg, n + 1))   # inclusive of both min_seg and n
    if seg_len >= n:
        return 0, n
    start = int(r.randint(0, n - seg_len + 1))
    return start, start + seg_len


def detect_two_domains(pdb_path: str, contact_thresh: float = 8.0, min_dom: int = 40) -> dict:
    """Best contiguous 2-domain split + domain-ness score from the Cα contact map.

    Mirrors ``scripts/eval/domain_split.py::detect_domains`` (reusing the same ``spa.eval.score`` Cα
    helpers) so the training-time selector matches the offline analysis exactly. Score =
    ``cross_domain_contacts / min(intra1, intra2)`` — **lower is a cleaner two-domain split** (few
    inter-domain contacts relative to the weaker domain's internal contacts). Returns ``boundary=None``
    when the structure is too short or every candidate boundary is degenerate.
    """
    from spa.eval.score import _as_struct, _ca_array, _coords64

    X = _coords64(_ca_array(_as_struct(pdb_path)))            # [n_ca, 3] Cα
    n = len(X)
    if n < 2 * min_dom:
        return {"n_res": n, "boundary": None, "score": None}

    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    c = (d < contact_thresh).astype(np.float64)
    np.fill_diagonal(c, 0.0)

    best_b, best_score = None, np.inf
    for b in range(min_dom, n - min_dom + 1):
        cross = c[:b, b:].sum()
        intra1 = c[:b, :b].sum() / 2.0
        intra2 = c[b:, b:].sum() / 2.0
        denom = min(intra1, intra2)
        if denom <= 0:
            continue
        score = cross / denom
        if score < best_score:
            best_score, best_b = score, b
    if best_b is None:
        return {"n_res": n, "boundary": None, "score": None}
    return {"n_res": n, "boundary": int(best_b), "score": round(float(best_score), 4)}


def domain_region(pdb_path: str, n: int, *, contact_thresh: float = 8.0, min_dom: int = 40,
                  domain_score_max: float = 0.4, rng=None) -> tuple[int, int] | None:
    """S = one domain of the cleanest 2-domain split, or ``None`` to fall back to segment.

    Returns ``None`` when there is no clean split — too short, degenerate contact map, score above
    ``domain_score_max`` (not cleanly two-domain, e.g. an arbitrary cut through a single fold), or the
    Cα count disagrees with the prompt length ``n`` (so the boundary index would not map to prompt rows,
    mirroring ``_motif_prompt_mask``'s alignment guard). Otherwise picks one of the two domains at
    random (either N- or C-terminal domain is a valid editing sub-region).
    """
    info = detect_two_domains(pdb_path, contact_thresh=contact_thresh, min_dom=min_dom)
    if info["boundary"] is None or info["score"] is None:
        return None
    if info["score"] > domain_score_max or info["n_res"] != n:
        return None
    b = info["boundary"]
    return (0, b) if _rng(rng).rand() < 0.5 else (b, n)


def subregion_pad_mask(
    n: int,
    *,
    weights=(0.4, 0.4, 0.2),
    min_seg: int = 12,
    contact_thresh: float = 8.0,
    min_dom: int = 40,
    domain_score_max: float = 0.4,
    pdb_path: str | None = None,
    rng=None,
) -> tuple[str, np.ndarray | None]:
    """Sample a granularity and return ``(g_effective, pad_mask)``.

    ``pad_mask`` is a ``bool[n]`` array, ``True`` at residues ``∉ S`` (to be masked from SPA's
    cross-attention), or ``None`` for the global case (attend to all rows). ``g_effective`` is the
    granularity actually realized (``domain`` degrades to ``segment`` when no clean split exists, and
    ``segment`` degrades to ``global`` when the sampled window covers the whole structure). S is never
    empty and never all-of-N-when-masked, so the resulting attention is always well-defined.
    """
    g = sample_granularity(weights, rng)
    region: tuple[int, int] | None = None
    if g == "domain":
        if pdb_path is not None:
            region = domain_region(pdb_path, n, contact_thresh=contact_thresh, min_dom=min_dom,
                                    domain_score_max=domain_score_max, rng=rng)
        if region is None:
            g = "segment"  # no clean 2-domain split (or no path) -> fall back to a contiguous segment
    if g == "segment":
        region = segment_region(n, min_seg, rng)
    if g == "global" or region is None or (region[0] == 0 and region[1] >= n):
        return ("global", None)  # S = all residues -> no masking
    start, end = region
    pad = np.ones(n, dtype=bool)   # True everywhere...
    pad[start:end] = False         # ...except S (the kept sub-region)
    return (g, pad)
