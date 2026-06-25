"""SPAWrappedAttention + SPAContext + SPAAdapter — wrap RFD3 blocks; thread the prompt side-channel.

Kickoff step 3. Spec: dev ``02_attachment_points.md`` §2, §5.

RFD3's ``StructureLocalAtomTransformerBlock.forward`` adds the attention output via an existing
residual (``blocks.py:692``)::

    Q_L = Q_L + self.dropout(self.attention_pair_bias(Q_L, C_L, P_LL, f=f, ...))

WRAP ``attention_pair_bias`` so its forward returns ``base_attn_out + λ·spa_term``; the existing
residual then carries the SPA contribution automatically — IP-Adapter's ``Z = Z_text + λ·Z_image``,
realized by module wrapping (RFD3 has no diffusers-style attn-processor registry).

Prompt threading: RFD3 carries no forward slot for an external prompt, so the projected prompt K/V
is **stashed on a shared** :class:`SPAContext` *before* RFD3's forward and reused across all 200
diffusion steps × 18 blocks (the prompt is constant — ESM3 is run once and cached). With no prompt
set (``context.k is None``) the wrapper returns ``base`` only ⇒ **wrapped-no-prompt == vanilla
RFD3** (the identity invariant / baseline). All SPA parameters are bundled in :class:`SPAAdapter`
so only they are optimized and checkpointed (IP-Adapter ``ModuleList`` pattern).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SPAContext:
    """Shared side-channel holding the projected prompt K/V (and key-padding mask).

    Computed once per design (the prompt is constant across blocks and diffusion steps) and read
    by every wrapper. ``k is None`` ⇒ wrapped-but-unconditioned (== vanilla RFD3).

    Attributes:
        k: shared prompt keys ``[D, M, c_model]`` (or ``None``).
        v: shared prompt values ``[D, M, c_model]`` (or ``None``).
        key_padding_mask: ``[D, M]`` boolean (``True`` at PAD prompt positions) or ``None``.
    """

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None


class SPAWrappedAttention(nn.Module):
    """Wraps one frozen ``LocalAttentionPairBias`` and adds the SPA cross-attention term.

    Holds the frozen original (params set ``requires_grad=False``) plus a reference to a per-block
    :class:`~spa.model.cross_attention.SPACrossAttention` and the shared :class:`SPAContext`.
    ``forward`` mirrors the original signature, calls the frozen base, and adds the SPA term only
    when a prompt is set on the context.

    Args:
        orig: the original ``attention_pair_bias`` module (kept frozen).
        context: the shared side-channel (same instance on all 18 wrappers).
        spa: this block's SPA cross-attention module (also owned by :class:`SPAAdapter`).
    """

    def __init__(self, orig: nn.Module, context: SPAContext, spa: nn.Module) -> None:
        super().__init__()
        self.orig = orig
        self.orig.requires_grad_(False)  # belt-and-suspenders; the loader also freezes the host
        self.spa = spa
        self._context = context  # plain attr (not a submodule/param); mutated per design

    def forward(self, Q_L: torch.Tensor, C_L=None, P_LL=None, *args, **kwargs) -> torch.Tensor:
        # RFD3 sets `use_checkpointing` on this module each forward (blocks.py:623); pass it through.
        if "use_checkpointing" in self.__dict__:
            self.orig.use_checkpointing = self.__dict__["use_checkpointing"]

        base = self.orig(Q_L, C_L, P_LL, *args, **kwargs)

        ctx = self._context
        if ctx is None or ctx.k is None:
            return base  # wrapped-no-prompt == vanilla RFD3
        spa = self.spa(Q_L, ctx.k, ctx.v, key_padding_mask=ctx.key_padding_mask)
        return base + spa


class SPAAdapter(nn.Module):
    """Bundles every trainable SPA parameter and owns the shared prompt side-channel.

    ``self.parameters()`` are exactly the SPA params (front-end projector + shared
    :class:`~spa.model.cross_attention.SPAPromptKV` + the per-block
    :class:`~spa.model.cross_attention.SPACrossAttention` ``ModuleList``) — the IP-Adapter
    gather-into-a-ModuleList pattern, so the optimizer and checkpoints see SPA only. The same
    ``cross_attn[i]`` instances are referenced by the wrappers.

    Use:
        ``set_prompt(prompt)`` once per design (before RFD3's forward) to project + stash K/V;
        ``clear_prompt()`` for CFG zero-prompt dropout / the wrapped-no-prompt baseline;
        ``set_scale(λ)`` to tune prompt strength at inference.
    """

    def __init__(self, projector: nn.Module, prompt_kv: nn.Module,
                 cross_attn: nn.ModuleList, context: SPAContext) -> None:
        super().__init__()
        self.projector = projector      # variant front-end: prompt P -> P'  [D, M, c_kv]
        self.prompt_kv = prompt_kv       # shared SPAPromptKV: P' -> (K, V)
        self.cross_attn = cross_attn     # ModuleList[SPACrossAttention], one per wrapped block
        self._context = context          # shared with the wrappers (plain attr)

    @property
    def context(self) -> SPAContext:
        return self._context

    def set_prompt(self, prompt: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> None:
        """Project ``prompt`` ``[D, N, c_kv]`` and stash K/V on the shared context (once per design)."""
        p = self.projector(prompt)
        if key_padding_mask is not None and key_padding_mask.shape[-1] != p.shape[1]:
            # a projector that changes the token count (e.g. variant-A global CLSS: M=n_tokens≠N)
            # invalidates a per-residue mask -> drop it (per-residue non-overlap is variant-C only).
            key_padding_mask = None
        k, v = self.prompt_kv(p)
        self._context.k, self._context.v, self._context.key_padding_mask = k, v, key_padding_mask

    def clear_prompt(self) -> None:
        """Drop the prompt ⇒ wrappers return base only (CFG zero-prompt dropout / baseline)."""
        self._context.k = self._context.v = self._context.key_padding_mask = None

    def set_scale(self, value: float) -> None:
        """Set the inference scale λ on every block (0 ⇒ unconditional == vanilla RFD3)."""
        for ca in self.cross_attn:
            ca.set_scale(value)
