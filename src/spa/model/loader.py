"""Attach SPA to a frozen RFD3 model: wrap the 18 blocks, freeze the host, gather trainables.

Kickoff step 3. Spec: dev ``02_attachment_points.md`` §5; freezing per ``03`` §1 / repo CLAUDE.md.

At load time, after building frozen RFD3 and loading its weights:

  1. Build a shared :class:`~spa.model.wrapper.SPAContext` and the shared prompt K/V projections
     (the prompt is constant across all 18 blocks and 200 steps -> project once).
  2. Iterate ``model.diffusion_module.diffusion_transformer.blocks`` (i = 0..17) and replace each
     ``block.attention_pair_bias`` with ``SPAWrappedAttention(orig=..., context=..., spa=...)``.
  3. Freeze everything that is not SPA: RFD3 + ESM3 (+ CLSS when the 1×32 variant is used).
  4. Gather all SPA submodules into a single ``nn.ModuleList`` so only SPA params are optimized
     and checkpointed (IP-Adapter pattern).

Returns the SPA parameter collection (for the optimizer + checkpointing) and the context
handle (for stashing the per-design prompt before each RFD3 forward).
"""

from __future__ import annotations

from torch import nn

# Path to the wrapped modules inside RFD3 (dev 02 §1):
#   model.diffusion_module.diffusion_transformer.blocks[i].attention_pair_bias   for i in 0..17
RFD3_BLOCKS_ATTR = "diffusion_module.diffusion_transformer.blocks"
RFD3_ATTENTION_ATTR = "attention_pair_bias"
N_TOKEN_BLOCKS = 18


def attach_spa(model: nn.Module, cfg) -> tuple[nn.ModuleList, object]:
    """Wrap the 18 RFD3 attention blocks with SPA, freeze the host, and return SPA trainables.

    Args:
        model: a built, weight-loaded RFdiffusion3 model (frozen host).
        cfg: composed Hydra config (``model`` + ``variant`` groups drive SPA construction).

    Returns:
        ``(spa_params, context)`` — the ``nn.ModuleList`` of SPA submodules to optimize/save, and
        the shared ``SPAContext`` to stash the per-design prompt on before each forward.

    Invariant: with no prompt stashed on ``context``, the wrapped model is bit-for-bit vanilla
    RFD3 (see ``tests/test_identity_at_init.py``).
    """
    # TODO(step 3): implement wrap-freeze-gather per the module docstring.
    raise NotImplementedError(
        "attach_spa is a step-1 scaffold; implement in kickoff step 3 "
        "(dev 02_attachment_points.md §5)."
    )


def freeze_host(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on all non-SPA parameters (RFD3, ESM3, CLSS)."""
    raise NotImplementedError("kickoff step 3")
