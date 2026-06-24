"""Variant front-end projectors — the only thing that differs between the three SPA variants.

Kickoff steps 2–3 support; full spec: dev ``03_spa_architecture.md`` §4 and
``02_attachment_points.md`` §7. All three variants attach at the SAME 18 points and share the
per-block Q/out + shared K/V machinery; they differ ONLY in how the raw prompt becomes
``P' ∈ [D, M, 1536]``. Selected by Hydra (``configs/variant/{C,B,A}_*.yaml``).

    Variant C — N×1536 per-residue (PRIMARY): P = ESM3 per-residue [D,N,1536].
        Front-end = identity (M=N), or a Perceiver Resampler compressing N->fixed M to bound
        the quadratic cross-attn cost for long proteins. No CLSS needed (raw ESM3).
    Variant B — 1×1536 global: P = mean-pool of ESM3 [D,1,1536]. Front-end = fan-out to a few
        learned tokens (M≈4) or M=1. Cheapest; global "soft-fold" steering. No CLSS needed.
    Variant A — 1×32 CLSS: P = CLSS-projected [D,1,32] via the frozen CLSS structure_adapter.
        Front-end = fan-out 32->1536->tokens. Most lossy; ONLY this variant loads the CLSS ckpt.
"""

from __future__ import annotations

import torch
from torch import nn


class IdentityProjector(nn.Module):
    """Variant C primary path: pass ESM3 per-residue embeddings through unchanged (M = N)."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError(
            "Front-end projectors are a step-1 scaffold; implement alongside kickoff steps 2–3 "
            "(dev 03_spa_architecture.md §4)."
        )

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:  # [D,N,1536] -> [D,M,1536]
        raise NotImplementedError("kickoff steps 2–3")


# TODO(step 2–3): ResamplerProjector (C long-N), GlobalFanOutProjector (B), CLSSProjector (A).
# Build a `make_projector(variant_cfg) -> nn.Module` factory keyed on configs/variant/*.yaml.
