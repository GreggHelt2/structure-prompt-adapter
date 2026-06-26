"""Custom RFD3 training conditions for SPA — the coupled-island motif sampler (dev ``10`` §7.3).

RFD3's stock ``IslandCondition`` selects motif islands by **independent** (n_islands, island_len) ranges
(`conditions/island.yaml`: 2–5 islands × 1–12 res). That can yield non-motifs (e.g. two single residues),
never a single contiguous motif, and doesn't couple count↔length. For Run B's hard⊕soft mixing we want
motifs shaped like the standard benchmarks — **MotifBench** (30 cases, single/double/multi-segment;
arXiv:2502.12479) and **Genie2** design25 + 6 multi-motif (arXiv:2405.15489): single & few-segment motifs
dominate, segments ~5–30 res, totals ~10–40 res, leaving the majority as scaffold.

:class:`CoupledIslandCondition` overrides **only** ``sample_motif_tokens``: sample ``n_islands`` (1–5,
weighted toward fewer), sample a **total** motif size (a fraction of the protein, floored/capped), then
**partition** that total across the islands (each ≥ a floor) and place them non-overlapping. So fewer
islands ⇒ longer each, and there are no degenerate 1–2-residue "motifs". Everything else (conditioning
strategy, atom selection) is inherited; downstream (RFD3-native fix + SPA prompt mask) consumes the same
``is_motif_token`` boolean. **No foundry edit** — subclass + a config ``_target_`` swap (dev ``10`` §7.3).
"""
from __future__ import annotations

import numpy as np
from atomworks.ml.utils.token import get_token_starts, spread_token_wise
from rfd3.transforms.training_conditions import IslandCondition


class CoupledIslandCondition(IslandCondition):
    """Motif islands sized by a coupled total-then-partition scheme (dev ``10`` §7.3).

    Bounds (defaults grounded in MotifBench / Genie2): ``n_islands`` 1–5 weighted toward fewer; total
    motif = ``clip(U(frac_min,frac_max)·L, abs_min, abs_max)`` and ≥ ``n_islands × island_floor``;
    per-island ≥ ``island_floor``. Inherits all other ``IslandCondition`` behaviour.
    """

    def __init__(
        self,
        *,
        n_islands_min: int = 1,
        n_islands_max: int = 5,
        n_islands_weights=(0.35, 0.30, 0.20, 0.10, 0.05),  # aligned to range(min, max+1)
        motif_frac_min: float = 0.10,
        motif_frac_max: float = 0.35,
        motif_abs_min: int = 8,
        motif_abs_max: int = 50,
        island_floor: int = 4,
        **island_kwargs,  # the parent contract: name, frequency, island_sampling_kwargs, p_* …
    ) -> None:
        super().__init__(**island_kwargs)
        self.n_islands_min = int(n_islands_min)
        self.n_islands_max = int(n_islands_max)
        self.n_islands_weights = list(n_islands_weights) if n_islands_weights is not None else None
        self.motif_frac_min = float(motif_frac_min)
        self.motif_frac_max = float(motif_frac_max)
        self.motif_abs_min = int(motif_abs_min)
        self.motif_abs_max = int(motif_abs_max)
        self.island_floor = int(island_floor)

    def _sample_n_islands(self, kmax: int) -> int:
        ks = list(range(self.n_islands_min, kmax + 1))
        w = self.n_islands_weights
        if w is not None:
            w = np.asarray(w[: len(ks)], dtype=float)
            w = w / w.sum()
        return int(np.random.choice(ks, p=w))

    def _sample_coupled_islands(self, n: int) -> np.ndarray:
        """``bool[n]`` motif mask over the ``n`` protein tokens (coupled total-then-partition)."""
        mask = np.zeros(n, dtype=bool)
        floor = self.island_floor
        if n < floor + 1:  # chain too short for even one island + scaffold → no motif
            return mask
        # how many islands fit, leaving ≥1 scaffold residue per island
        kmax = min(self.n_islands_max, max(1, (n - 1) // (floor + 1)))
        k = self._sample_n_islands(max(self.n_islands_min, 1) if kmax < self.n_islands_min else kmax)
        # total motif size: fraction of protein, clipped to [abs_min, abs_max] and feasibility
        T = int(round(np.random.uniform(self.motif_frac_min, self.motif_frac_max) * n))
        T = max(T, self.motif_abs_min, k * floor)          # not below the per-island floors
        T = min(T, self.motif_abs_max, n - (k - 1) - 1)    # leave ≥1 between islands + ≥1 scaffold
        if T < k * floor:                                   # cap forced T too small → drop islands
            k = max(1, T // floor)
        # partition T into k segments, each ≥ floor
        seg = np.full(k, floor, dtype=int)
        extra = T - k * floor
        if extra > 0:
            seg += np.random.multinomial(extra, np.full(k, 1.0 / k))
        # scaffold (n − T) into k+1 gaps, with ≥1 between adjacent islands
        gaps = np.zeros(k + 1, dtype=int)
        if k > 1:
            gaps[1:k] = 1
        free = (n - int(seg.sum())) - int(gaps.sum())
        if free > 0:
            gaps += np.random.multinomial(free, np.full(k + 1, 1.0 / (k + 1)))
        # lay out: gap0, seg0, gap1, seg1, …, seg(k−1), gap_k
        pos = 0
        for i in range(k):
            pos += int(gaps[i])
            mask[pos: pos + int(seg[i])] = True
            pos += int(seg[i])
        return mask

    def sample_motif_tokens(self, atom_array):
        token_level_array = atom_array[get_token_starts(atom_array)]
        is_motif_token = np.asarray(~token_level_array.is_protein, dtype=bool).copy()  # non-protein = motif
        is_protein = np.asarray(token_level_array.is_protein, dtype=bool)
        is_motif_token[is_protein] = self._sample_coupled_islands(int(is_protein.sum()))
        return spread_token_wise(atom_array, is_motif_token)
