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
class SPAPromptSlot:
    """One prompt in a multi-prompt (two-steer) design: its projected K/V + its region mask.

    The single-prompt path uses :class:`SPAContext`'s ``k``/``v`` directly; the two-steer path
    instead stashes a *list* of these slots (``SPAContext.prompts``) and the wrapper loops-and-sums
    ``Σₖ SPA(A, Gₖ)·profileₖ``. Every experiment so far uses **disjoint** ``profile`` masks (region
    U₁ vs U₂, with the pinned motif + free region 0 in every mask), so each design residue receives
    exactly one prompt's contribution — but that is a **convention, not a constraint**: overlap is
    permitted and is validated nowhere. See :meth:`SPAAdapter.set_prompts` for what overlap does
    (dev ``31`` §3).

    Attributes:
        k: this prompt's keys ``[D, Mₖ, c_model]`` (Mₖ = |Gₖ| tokens; may differ across slots).
        v: this prompt's values ``[D, Mₖ, c_model]``.
        profile: per-residue λ mask ``[I]`` in ``[0, 1]`` selecting this prompt's region (or ``None``
            = every design residue, i.e. a uniform steer).
        key_padding_mask: optional ``[D, Mₖ]`` boolean (``True`` at PAD prompt positions) or ``None``.
    """

    k: torch.Tensor
    v: torch.Tensor
    profile: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None


@dataclass
class SPAContext:
    """Shared side-channel holding the projected prompt K/V (and key-padding mask).

    Computed once per design (the prompt is constant across blocks and diffusion steps) and read
    by every wrapper. ``k is None`` (and ``prompts is None``) ⇒ wrapped-but-unconditioned
    (== vanilla RFD3).

    Attributes:
        k: shared prompt keys ``[D, M, c_model]`` (or ``None``).
        v: shared prompt values ``[D, M, c_model]`` (or ``None``).
        key_padding_mask: ``[D, M]`` boolean (``True`` at PAD prompt positions) or ``None``.
        prompts: optional list of :class:`SPAPromptSlot` for a multi-prompt (two-steer) design.
            When set it takes precedence over ``k``/``v`` and the wrapper sums over the slots;
            ``None`` ⇒ the ordinary single-prompt path.
    """

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None
    prompts: list["SPAPromptSlot"] | None = None


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
        if ctx is None:
            return base  # wrapped-no-prompt == vanilla RFD3
        if ctx.prompts is not None:
            # Multi-prompt (two-steer): sum each prompt's SPA term, gated by its region profile.
            # λ (set_scale) is global; the profiles route each prompt to its own region. Profiles
            # are disjoint by convention, not by enforcement — where they overlap the terms add
            # (dev ``31`` §3; see set_prompts).
            spa_sum = None
            for slot in ctx.prompts:
                term = self.spa(Q_L, slot.k, slot.v,
                                key_padding_mask=slot.key_padding_mask, profile=slot.profile)
                spa_sum = term if spa_sum is None else spa_sum + term
            return base if spa_sum is None else base + spa_sum
        if ctx.k is None:
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
        ``set_prompts([(prompt, profile), ...])`` for a multi-prompt (two-steer) design — one
            steered region per prompt, gated by a disjoint profile (dev: two-steer driver);
        ``set_null_prompt(D)`` for CFG zero-prompt dropout (the learned null token e∅, dev ``11`` §6);
        ``clear_prompt()`` for the wrapped-no-prompt baseline (λ=0 / no prompt available);
        ``set_scale(λ)`` to tune prompt strength at inference;
        ``set_profile(w)`` for optional per-residue region-specific steering (``None`` = uniform).
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
        self._context.prompts = None  # single-prompt path clears any stale multi-prompt state

    def set_prompts(
        self,
        prompts_and_profiles: list[tuple[torch.Tensor, torch.Tensor | None]],
        key_padding_masks: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Stash **several** prompts for a multi-prompt (two-steer) design (once per design).

        Each ``(prompt, profile)`` is projected + turned into its own K/V exactly as
        :meth:`set_prompt` does a single one; the wrapper then sums ``Σₖ SPA(A, Gₖ)·profileₖ``.
        A ``None`` profile means "steer every residue" (only sensible for a single prompt).
        ``λ`` (:meth:`set_scale`) is global across all prompts.

        **Overlapping profiles are permitted and are NOT validated** (dev ``31`` §3). Every
        experiment so far uses **disjoint** masks (region U₁ vs U₂, pinned motif + free region 0 in
        every mask), so each residue gets exactly one prompt's contribution — a convention, not a
        constraint. Where masks *do* overlap the terms simply **add**: each prompt runs its own
        softmax, so the result is a vector **sum of independent steering directions**, not a blend
        and not a competition. Two consequences:

        - **Magnitude grows** — measured ~1.4× for two full-strength, near-orthogonal prompts (they
          add in quadrature). That is an uncontrolled **over-steer**, the regime that costs
          designability. Keep ``Σₖ profileₖ[i] ≤ 1`` unless over-steering deliberately.
        - **The summed direction is not "a fold between G₁ and G₂".** Summation cannot express
          blending at *any* profile setting; that needs per-query key routing (dev ``19``).

        Args:
            prompts_and_profiles: list of ``(prompt [D, Nₖ, c_kv], profile [I] or None)``.
            key_padding_masks: optional per-prompt ``[D, Nₖ]`` masks (default all ``None``).
        """
        slots: list[SPAPromptSlot] = []
        for i, (prompt, profile) in enumerate(prompts_and_profiles):
            p = self.projector(prompt)
            kpm = None if key_padding_masks is None else key_padding_masks[i]
            if kpm is not None and kpm.shape[-1] != p.shape[1]:
                # a token-count-changing projector invalidates a per-residue prompt mask (see
                # set_prompt); non-overlap masking is the identity-projector (variant-C) path anyway.
                kpm = None
            k, v = self.prompt_kv(p)
            prof = None if profile is None else profile.detach().float()
            slots.append(SPAPromptSlot(k=k, v=v, profile=prof, key_padding_mask=kpm))
        # multi-prompt supersedes the single-prompt slot; clear it to avoid double-counting.
        self._context.k = self._context.v = self._context.key_padding_mask = None
        self._context.prompts = slots

    def set_null_prompt(self, batch: int) -> None:
        """Stash the learned null-token K/V on the context — CFG zero-prompt dropout (dev ``11`` §6).

        Use in place of :meth:`clear_prompt` on a *dropped* training step: the SPA cross-attn stays
        **live** on the learned null token ``e∅`` (a single key/value), so a gradient still reaches
        the adapter — clearing the prompt would bypass SPA entirely and zero the gradient (the B1
        crash). ``batch`` is the step's diffusion batch ``D`` so the null K/V matches the query's
        batch dim. The null bypasses the variant projector (prompt-absence is variant-agnostic).
        """
        k, v = self.prompt_kv.null_kv(batch)
        self._context.k, self._context.v, self._context.key_padding_mask = k, v, None
        self._context.prompts = None  # null-prompt is single-slot; clear any multi-prompt state

    def clear_prompt(self) -> None:
        """Drop the prompt ⇒ wrappers return base only (the wrapped-no-prompt baseline: λ=0 / no
        prompt available). NOTE: CFG zero-prompt dropout uses :meth:`set_null_prompt` instead, so a
        gradient still flows to the adapter (dev ``11`` §6)."""
        self._context.k = self._context.v = self._context.key_padding_mask = None
        self._context.prompts = None

    def set_scale(self, value: float) -> None:
        """Set the inference scale λ on every block (0 ⇒ unconditional == vanilla RFD3)."""
        for ca in self.cross_attn:
            ca.set_scale(value)

    def set_profile(self, weights: torch.Tensor | None) -> None:
        """Set a per-residue λ weight ``[I]`` on every block (region-specific steering), or ``None``
        to restore uniform scalar λ. State B=1, state C=0, feathering=a 0→1 ramp (dev: the three-way
        A/B/C masking probe). Effective strength at residue ``i`` is ``set_scale(λ) * weights[i]``."""
        for ca in self.cross_attn:
            ca.set_profile(weights)
