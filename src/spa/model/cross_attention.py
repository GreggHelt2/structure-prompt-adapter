"""SPACrossAttention — the decoupled cross-attention term added to each wrapped RFD3 block.

Kickoff step 2. Spec: dev ``03_spa_architecture.md`` §2 and ``02_attachment_points.md`` §4.

Per wrapped block, a standard multi-head cross-attention over the M prompt tokens:

    q   = Wq · A_I          [D, I, 768]   -> [D, I, c_model]   (query = frozen token features)
    K   = Wk · P'           [D, M, 1536]  -> [D, M, c_model]   (shared across blocks; see loader)
    V   = Wv · P'           [D, M, 1536]  -> [D, M, c_model]
    attn = softmax(q Kᵀ/√d_head + key_pad_mask) V             [D, I, c_model]
    spa = Wo · attn         [D, I, c_model] -> [D, I, 768]     (Wo ZERO-INITIALIZED)
    out = λ · spa                                              (λ = inference-tunable scale)

Identity-at-init: ``Wo`` is zero-initialized so ``spa == 0`` at step 0 regardless of q/K/V
init -> the wrapped model reproduces vanilla RFD3 exactly (the standing correctness gate; see
repo CLAUDE.md and ``tests/test_identity_at_init.py``). Optional input RMSNorm on the
pre-final-LayerNorm ESM3 prompt (dev ``01`` §3.3).

This module owns only the per-block query/output projections (Wq, Wo, λ). The shared K/V
projections live once on the loader/context (the prompt is constant across all 18 blocks and
all 200 diffusion steps) — see ``spa.model.wrapper`` / ``spa.model.loader``.
"""

from __future__ import annotations

import torch
from torch import nn


class SPACrossAttention(nn.Module):
    """Per-block decoupled cross-attention. See module docstring for the full spec.

    Args:
        c_query: token-feature width (RFD3 ``c_token`` = 768).
        c_kv: prompt width feeding K/V (ESM3 hidden = 1536).
        c_model: internal attention width (default 768; see dev ``03`` §6 param table).
        n_head: number of attention heads.
        lambda_init: initial value of the inference-tunable scale λ (1.0 for training;
            identity-at-init comes from zero-init Wo, not λ).
        input_rmsnorm: apply RMSNorm to the prompt before K/V (ESM3 is pre-LayerNorm).
    """

    def __init__(
        self,
        c_query: int = 768,
        c_kv: int = 1536,
        c_model: int = 768,
        n_head: int = 8,
        lambda_init: float = 1.0,
        input_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        # TODO(step 2): build Wq (c_query->c_model), Wo (c_model->c_query, ZERO-INIT),
        # optional shared K/V projections, scalar λ buffer/param, optional RMSNorm.
        raise NotImplementedError(
            "SPACrossAttention is a step-1 scaffold; implement in kickoff step 2 "
            "(dev 03_spa_architecture.md §2)."
        )

    def forward(self, query: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the SPA term ``λ · Wo · attn(Wq·query, k, v)`` with shape ``[D, I, c_query]``."""
        raise NotImplementedError("kickoff step 2")
