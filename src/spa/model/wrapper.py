"""SPAWrappedAttention + SPAContext — wrap RFD3 blocks and thread the prompt as a side-channel.

Kickoff step 3. Spec: dev ``02_attachment_points.md`` §2, §5.

RFD3's ``StructureLocalAtomTransformerBlock.forward`` adds the attention output via an existing
residual (``blocks.py:692``)::

    Q_L = Q_L + self.dropout(self.attention_pair_bias(Q_L, C_L, P_LL, f=f, ...))

If we WRAP ``attention_pair_bias`` so its forward returns ``base_attn_out + spa_term``, the
existing residual carries the SPA contribution automatically — exactly IP-Adapter's
``Z = Z_text + λ·Z_image`` pattern, realized by module wrapping (RFD3 has no diffusers-style
attn-processor registry).

Prompt threading (the key constraint): RFD3 carries no forward slot for an external prompt, so
the projected prompt K/V is **stashed on the wrappers** via a shared ``SPAContext`` *before*
invoking RFD3's forward, and reused across all 200 diffusion steps × 18 blocks (the prompt is
constant — ESM3 is run once and cached). With no prompt set, the wrapper returns ``base`` only,
so a wrapped model with no prompt == vanilla RFD3 (the identity invariant / baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SPAContext:
    """Side-channel holding the projected prompt K/V, shared across all wrapped blocks.

    Computed once per design (prompt is constant across blocks and diffusion steps). Set to
    ``None`` / left unset to run wrapped-but-unconditioned (== vanilla RFD3).

    Attributes:
        k: shared prompt keys ``[D, M, c_model]``.
        v: shared prompt values ``[D, M, c_model]``.
        key_padding_mask: ``[D, M]`` mask over the M prompt tokens (variable N per design).
    """

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None


class SPAWrappedAttention(nn.Module):
    """Wraps one frozen ``LocalAttentionPairBias`` and adds the SPA cross-attention term.

    Holds the frozen original (``requires_grad=False``) plus a trainable
    :class:`~spa.model.cross_attention.SPACrossAttention`. ``forward`` mirrors the original
    signature, calls the frozen base, and adds the SPA term when a prompt context is set.

    Args:
        orig: the original ``attention_pair_bias`` module (kept frozen).
        context: shared :class:`SPAContext` (the same instance is set on all 18 wrappers).
        spa: the per-block SPA cross-attention module.
    """

    def __init__(self, orig: nn.Module, context: SPAContext, spa: nn.Module) -> None:
        super().__init__()
        # TODO(step 3): store orig (frozen), context, spa; ensure orig params requires_grad=False.
        raise NotImplementedError(
            "SPAWrappedAttention is a step-1 scaffold; implement in kickoff step 3 "
            "(dev 02_attachment_points.md §5)."
        )

    def forward(self, Q_L: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """``base = orig(Q_L, *args, **kwargs)``; return ``base + spa(Q_L, ctx.k, ctx.v)`` when a
        prompt is set, else ``base`` (so wrapped-no-prompt == vanilla RFD3)."""
        raise NotImplementedError("kickoff step 3")
