"""Structure Prompt Adapter (SPA).

A parameter-efficient decoupled-cross-attention sidecar for RFdiffusion3, conditioned on
ESM3 structural prompts. RFdiffusion3 and ESM3 stay frozen; only SPA trains.

Design source of truth: ``../structure-prompt-adapter-dev/docs/plan/`` (see repo CLAUDE.md).

Package map:
    spa.model.cross_attention  -- SPACrossAttention (the decoupled cross-attn term)
    spa.model.wrapper          -- SPAWrappedAttention + SPAContext (prompt side-channel)
    spa.model.projectors       -- variant front-ends (C: N×1536, B: 1×1536, A: 1×32 CLSS)
    spa.model.loader           -- attach SPA to a frozen RFD3, freeze, gather trainable params
    spa.prompt.esm3_prompt     -- ESM3 -> per-residue (N,1536) prompt producer + caching
    spa.data.dataset           -- cache-backed dataloader
    spa.train.harness          -- own training loop (grad on SPA only)
    spa.utils.device           -- device selection from config (no hardcoded UUID)
"""

__version__ = "0.0.1.dev0"
