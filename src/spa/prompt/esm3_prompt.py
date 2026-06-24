"""ESM3 prompt producer: backbone coords -> per-residue (N, 1536) structural prompt.

Kickoff step 5. Spec: dev ``01_codebase_analysis.md`` §3.5, ``02_attachment_points.md`` §6,
``03`` §1.

ESM3 is LOCAL (HuggingFace weights ``esm3_sm_open_v1``, already cached), frozen, run ONCE per
design (resolved: dev ``07`` C2 — not the Forge API; no ``ESM_API_KEY``)::

    emb = esm3.logits(pt, LogitsConfig(return_embeddings=True)).embeddings   # [D, L+2, 1536]

The ``logits(... return_embeddings=True)`` path is leaner than ``forward_and_sample``. ESM3
prepends BOS / appends EOS, so N = L+2; strip rows 0 and L+1 for a clean per-residue prompt
aligned to RFD3's L residues (recommended; dev ``01`` §3.2). The embeddings are
pre-final-LayerNorm (``01`` §3.3) — apply RMSNorm inside SPA if normalization is wanted.

Because the prompt is SE(3)-invariant and constant across diffusion steps, it is computed once
and cached. A small local cache (a few GB) is enough for A5000 pipeline testing; the full
~251 GB cache is generated on a cloud H100 (dev ``04`` §10, ``scripts/gen_esm3_cache.py``).
"""

from __future__ import annotations

import torch


def esm3_prompt(structure, esm3_model, strip_bos_eos: bool = True) -> torch.Tensor:
    """Return ESM3 per-residue embeddings for one structure.

    Args:
        structure: an ESM3 ``ESMProtein`` / tokenized input for the backbone.
        esm3_model: a loaded, frozen local ESM3 model.
        strip_bos_eos: drop rows 0 and L+1 so the prompt aligns to RFD3's L residues.

    Returns:
        ``[N, 1536]`` (N = L if stripped, else L+2).
    """
    # TODO(step 5): tokenize -> esm3.logits(pt, LogitsConfig(return_embeddings=True)).embeddings
    # under torch.no_grad(); optionally strip BOS/EOS; return on CPU/fp16 for caching.
    raise NotImplementedError(
        "esm3_prompt is a step-1 scaffold; implement in kickoff step 5 "
        "(dev 01_codebase_analysis.md §3.5)."
    )
