"""Attach SPA to a frozen RFD3 model: wrap the 18 blocks, freeze the host, return the adapter.

Kickoff step 3. Spec: dev ``02_attachment_points.md`` §5; freezing per ``03`` §1 / repo CLAUDE.md.

Verified against source: an ``RFD3`` instance (``model/RFD3.py:18``, a plain ``nn.Module``) exposes
``diffusion_module`` (``RFD3.py:57``) -> ``diffusion_transformer`` (``RFD3_diffusion_module.py:111``,
a ``LocalTokenTransformer``) -> ``blocks`` (``blocks.py:600``), each block's ``attention_pair_bias``
being the ``LocalAttentionPairBias`` SPA wraps (``blocks.py:674``).

``attach_spa`` builds a single :class:`~spa.model.wrapper.SPAAdapter` (front-end projector + shared
``SPAPromptKV`` + a per-block ``SPACrossAttention`` ``ModuleList``), replaces each block's
``attention_pair_bias`` with an :class:`~spa.model.wrapper.SPAWrappedAttention`, freezes the host,
then re-enables the collaterally-frozen SPA cross-attn. Returns the adapter — its ``.parameters()``
are the SPA params (a variant's frozen encoder, e.g. variant-A's CLSS adapter, is included but stays
non-trainable), and it owns the prompt side-channel (``set_prompt`` / ``set_scale``).
"""

from __future__ import annotations

from torch import nn

from .cross_attention import SPACrossAttention, SPAPromptKV
from .projectors import make_projector
from .wrapper import SPAAdapter, SPAContext, SPAWrappedAttention

# Attribute chain from an RFD3 module down to the wrapped ModuleList (dev 02 §1; verified above).
RFD3_BLOCKS_ATTR = "diffusion_module.diffusion_transformer.blocks"
RFD3_ATTENTION_ATTR = "attention_pair_bias"
N_TOKEN_BLOCKS = 18


def _resolve_blocks(model: nn.Module) -> nn.ModuleList:
    """Walk ``RFD3_BLOCKS_ATTR`` on ``model``; raise a clear error if the chain is missing."""
    obj = model
    for attr in RFD3_BLOCKS_ATTR.split("."):
        if not hasattr(obj, attr):
            raise AttributeError(
                f"could not resolve {RFD3_BLOCKS_ATTR!r} on a {type(model).__name__}: missing "
                f"{attr!r}. Pass the RFD3 nn.Module that exposes `.diffusion_module` "
                f"(unwrap any Lightning/engine container first)."
            )
        obj = getattr(obj, attr)
    return obj


def freeze_host(model: nn.Module) -> None:
    """Freeze all parameters of the RFD3 host.

    Note: this also freezes the wrapped SPA submodules as collateral (they live under the wrapped
    blocks); :func:`attach_spa` re-enables them via ``adapter.requires_grad_(True)`` afterward.
    ESM3 / CLSS are not part of ``model`` and are frozen where they are loaded (the prompt
    producer / harness) using the same pattern.
    """
    model.requires_grad_(False)


def attach_spa(model: nn.Module, cfg) -> SPAAdapter:
    """Wrap the RFD3 attention blocks with SPA, freeze the host, and return the SPA adapter.

    Args:
        model: a built, weight-loaded ``RFD3`` module (the frozen host).
        cfg: composed Hydra config; uses the ``model`` and ``variant`` groups.

    Returns:
        :class:`~spa.model.wrapper.SPAAdapter` — ``.parameters()`` are the trainable SPA params;
        owns the prompt side-channel. With no prompt set, the wrapped model is bit-for-bit vanilla
        RFD3 (see ``tests/test_identity_at_init.py``).
    """
    mcfg, vcfg = cfg.model, cfg.variant
    if not mcfg.zero_init_output:
        raise NotImplementedError(
            "zero_init_output=false would break the wrapped-no-prompt == vanilla-RFD3 identity "
            "gate; non-zero-init is not supported (dev 03 §5)."
        )
    if not mcfg.shared_kv:
        raise NotImplementedError(
            "per-block K/V is an ablation (dev 03 §9.1) not yet wired; use shared_kv=true."
        )

    blocks = _resolve_blocks(model)
    n = len(blocks)
    if n != N_TOKEN_BLOCKS:  # informational — design assumes 18 (dev 02 §1)
        import warnings

        warnings.warn(f"expected {N_TOKEN_BLOCKS} token blocks, found {n}; wrapping all {n}.")

    projector = make_projector(vcfg, c_kv=mcfg.c_kv)
    prompt_kv = SPAPromptKV(c_kv=mcfg.c_kv, c_model=mcfg.c_model, input_rmsnorm=mcfg.input_rmsnorm)
    cross_attn = nn.ModuleList(
        SPACrossAttention(
            c_query=mcfg.c_query, c_model=mcfg.c_model,
            n_head=mcfg.n_head, lambda_init=mcfg.lambda_init,
        )
        for _ in range(n)
    )
    context = SPAContext()
    adapter = SPAAdapter(projector=projector, prompt_kv=prompt_kv,
                         cross_attn=cross_attn, context=context)

    for i, block in enumerate(blocks):
        orig = getattr(block, RFD3_ATTENTION_ATTR)
        setattr(block, RFD3_ATTENTION_ATTR,
                SPAWrappedAttention(orig=orig, context=context, spa=cross_attn[i]))

    # freeze_host(model) freezes RFD3 AND — as a side effect — the SPA cross-attn modules, since they
    # now live INSIDE the wrapped RFD3 blocks. Undo exactly that side effect by re-enabling cross_attn;
    # the standalone projector + prompt_kv were never frozen, so their as-built requires_grad is kept
    # (e.g. variant-A's frozen CLSS structure_adapter stays frozen, its trainable fan-out stays trainable).
    freeze_host(model)
    adapter.cross_attn.requires_grad_(True)
    return adapter
