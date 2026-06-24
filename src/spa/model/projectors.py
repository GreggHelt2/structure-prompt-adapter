"""Variant front-end projectors — the only thing that differs between the three SPA variants.

Spec: dev ``03_spa_architecture.md`` §4 and ``02_attachment_points.md`` §7. All three variants
attach at the SAME 18 points and share the per-block Q/out + shared K/V machinery; they differ
ONLY in how the raw prompt becomes ``P' ∈ [D, M, c_kv]``. Selected by Hydra
(``configs/variant/{C,B,A}_*.yaml``); built via :func:`make_projector`.

    Variant C — N×1536 per-residue (PRIMARY): identity (M=N), or a Resampler to a fixed M.
    Variant B — 1×1536 global: fan the pooled vector out to a few learned tokens.
    Variant A — 1×32 CLSS: fan the CLSS structure embedding 32->c_kv->tokens (loads CLSS).

Implemented now: the Variant-C identity path (the primary/MVP). B and A raise NotImplementedError
until their variants are exercised (Phase B).
"""

from __future__ import annotations

import torch
from torch import nn


class IdentityProjector(nn.Module):
    """Variant C primary path: pass ESM3 per-residue embeddings through unchanged (M = N)."""

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:  # [D,N,c_kv] -> [D,M=N,c_kv]
        return prompt


def make_projector(variant_cfg, c_kv: int = 1536) -> nn.Module:
    """Build the front-end projector for a variant config (``configs/variant/*.yaml``)."""
    name = variant_cfg.projector
    if name == "identity":
        if variant_cfg.get("resampler_tokens", None):
            raise NotImplementedError(
                "Resampler projector (Variant C long-N cost control) — TODO (dev 03 §4/§9.2)."
            )
        return IdentityProjector()
    if name == "global_fanout":
        raise NotImplementedError("Variant B global fan-out projector — TODO (dev 03 §4).")
    if name == "clss":
        raise NotImplementedError("Variant A CLSS projector — TODO (dev 03 §4; loads CLSS ckpt).")
    raise ValueError(f"unknown projector {name!r} (expected identity | global_fanout | clss)")
