"""Variant front-end projectors — the only thing that differs between the three SPA variants.

Spec: dev ``03_spa_architecture.md`` §4 and ``02_attachment_points.md`` §7. All three variants
attach at the SAME 18 points and share the per-block Q/out + shared K/V machinery; they differ
ONLY in how the raw prompt becomes ``P' ∈ [D, M, c_kv]``. Selected by Hydra
(``configs/variant/{C,B,A}_*.yaml``); built via :func:`make_projector`.

    Variant C — N×1536 per-residue (PRIMARY): identity (M=N), or a Resampler to a fixed M.
    Variant B — 1×1536 global: fan the pooled vector out to a few learned tokens.
    Variant A — 1×32 CLSS: mean -> frozen CLSS structure_adapter -> normalize -> learned token fan-out.

Implemented: Variant-C identity (primary/MVP) and Variant-A CLSS (the abstract's "uses the CLSS
encoder"; dev ``10`` / ``01`` §3). Variant B raises NotImplementedError until exercised.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class IdentityProjector(nn.Module):
    """Variant C primary path: pass ESM3 per-residue embeddings through unchanged (M = N)."""

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:  # [D,N,c_kv] -> [D,M=N,c_kv]
        return prompt


def load_clss_structure_adapter(model_name: str = "CLSS-sub.lckpt",
                                repo_id: str = "guyyanai/CLSS") -> nn.Linear:
    """Load CLSS's frozen structure adapter — a single ``Linear(1536, 32)`` (CLSS ``model.py:77``).

    The public CLSS checkpoint (HF ``guyyanai/CLSS``) stores only the adapters + temperature (ESM3 is
    stripped on save), so this is tiny. Used by :class:`CLSSProjector` to turn the mean-pooled ESM3
    embedding into CLSS's 1×32 contrastive output — the *frozen* "output of CLSS" the abstract names.

    We load *just this Linear* rather than instantiate ``CLSSModel``: ``CLSSModel.from_pretrained``
    reconstitutes the 1.4B-param ESM3 structure tower (the ckpt strips it) — wasteful, since SPA already
    runs/caches ESM3 and its tap is bit-identical to CLSS's (dev ``10``). And ``CLSSModel.embed_structures``
    re-runs ESM3 per structure in a Python loop, defeating the cache and the batch dataloader. So only the
    learned CLSS tensor is loaded here; :class:`CLSSProjector` applies CLSS's exact ``mean -> adapter ->
    L2-normalize`` recipe (verified vs ``CLSS model.py:316-348``) to the cached embeddings.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=model_name)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    w, b = sd["structure_adapter.0.weight"], sd["structure_adapter.0.bias"]
    adapter = nn.Linear(w.shape[1], w.shape[0])  # Linear(1536, 32)
    with torch.no_grad():
        adapter.weight.copy_(w)
        adapter.bias.copy_(b)
    adapter.requires_grad_(False)
    adapter.eval()
    return adapter


class CLSSProjector(nn.Module):
    """Variant A front-end: ESM3 per-residue prompt -> CLSS 1×32 -> learned token fan-out.

    Reproduces CLSS's structure embedding (``CLSSModel.embed_structures``, ``CLSS model.py:316-348``):
    mean-pool the ESM3 embeddings, apply the **frozen** ``structure_adapter`` (``Linear(1536,32)``), and
    L2-normalize — the literal "output of CLSS." A **trainable** linear then fans the 32-d vector out to
    ``n_tokens`` prompt tokens of width ``c_kv`` (IP-Adapter image-projection style), feeding the shared
    ``SPAPromptKV`` like any other prompt. SPA's cached ESM3 tap is bit-identical to CLSS's, so the 1×32
    is derived from the existing cache — no CLSS/ESM3 re-run (dev ``10`` / ``01`` §3). Frozen adapter ⇒
    only the fan-out trains.
    """

    def __init__(self, variant_cfg, c_kv: int = 1536) -> None:
        super().__init__()
        self.n_tokens = int(variant_cfg.n_tokens)
        self.c_kv = c_kv
        self.structure_adapter = load_clss_structure_adapter(
            variant_cfg.get("clss_model_name", "CLSS-sub.lckpt")
        )  # frozen Linear(1536, 32)
        if self.structure_adapter.in_features != c_kv:
            raise ValueError(
                f"CLSS structure_adapter expects {self.structure_adapter.in_features}-d input but "
                f"c_kv={c_kv}; variant A consumes the raw 1536-d ESM3 prompt."
            )
        self.fanout = nn.Linear(self.structure_adapter.out_features, self.n_tokens * c_kv)
        self.norm = nn.LayerNorm(c_kv)
        nn.init.xavier_uniform_(self.fanout.weight)
        nn.init.zeros_(self.fanout.bias)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:  # [D,N,c_kv] -> [D, n_tokens, c_kv]
        D = prompt.shape[0]
        z = F.normalize(self.structure_adapter(prompt.mean(dim=1)), dim=-1)  # [D, 32] frozen CLSS output
        return self.norm(self.fanout(z).view(D, self.n_tokens, self.c_kv))   # [D, n_tokens, c_kv]


class GlobalFanoutProjector(nn.Module):
    """Variant B front-end: mean-pool the ESM3 prompt to a 1×1536 global vector, then fan it out to
    ``n_tokens`` learned prompt tokens — like variant A but WITHOUT the CLSS contrastive head (the
    global vector is the raw mean-pooled ESM3 embedding; dev ``03`` §4). All-trainable; no CLSS dep.
    """

    def __init__(self, variant_cfg, c_kv: int = 1536) -> None:
        super().__init__()
        self.n_tokens = int(variant_cfg.n_tokens)
        self.c_kv = c_kv
        self.fanout = nn.Linear(c_kv, self.n_tokens * c_kv)
        self.norm = nn.LayerNorm(c_kv)
        nn.init.xavier_uniform_(self.fanout.weight)
        nn.init.zeros_(self.fanout.bias)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:  # [D,N,c_kv] -> [D, n_tokens, c_kv]
        D = prompt.shape[0]
        return self.norm(self.fanout(prompt.mean(dim=1)).view(D, self.n_tokens, self.c_kv))


def variant_from_checkpoint(ckpt_path, c_kv: int = 1536) -> dict:
    """Infer a variant config from a SPA checkpoint's OWN keys, so it cannot be mis-declared.

    ⭐ WHY THIS EXISTS. :func:`~spa.model.loader.attach_spa` builds the projector from ``cfg.variant``
    *before* ``load_spa`` ever opens the checkpoint, because ``load_state_dict`` needs a target module
    to already exist. So the caller has had to state an architecture the file itself already
    determines. Every checkpoint was trained with exactly one projector, and its key set names that
    projector uniquely:

        no ``projector.*`` keys at all       -> identity        (N×1536; IdentityProjector is PARAMETER-FREE)
        ``projector.structure_adapter.*``    -> clss            (1×32)
        ``projector.fanout.*`` and not above -> global_fanout   (1×1536)

    ``n_tokens`` is recovered arithmetically from ``fanout.weight``, whose shape is
    ``(n_tokens * c_kv, in_features)``.

    ⛔ THE POINT IS TO MAKE A MISMATCH UNREPRESENTABLE, not merely loud. It is already loud:
    ``load_spa`` calls ``load_state_dict`` with default ``strict=True`` and there is no
    ``strict=False`` anywhere in this repo, so a wrong declaration raises on missing or unexpected keys
    before anything is generated. But a flag the caller must set correctly is one more thing to get
    wrong, and reading it off the file removes the degree of freedom entirely. Same reasoning as the
    determinism defaults: an omitted opt-in is indistinguishable from a chosen opt-out.

    ⚠️ ``name`` is COSMETIC on the eval path (only ``spa.train.harness`` reads ``cfg.variant.name``, for
    W&B tags and checkpoint filenames), and is returned in the project's dims convention rather than the
    old A/B/C letters.

    ⚠️ This identifies the ARCHITECTURE, never WHICH TRAINING RUN produced the weights. The three
    N×1536 adapters share key set, shapes and file size (94,526,980 bytes), so ``uncond`` / ``motif`` /
    ``multigran`` are indistinguishable here; sha256 is the only disambiguator.
    """
    import torch

    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "adapter" in obj and "optimizer" in obj:
        obj = obj["adapter"]          # a full-state snapshot keeps the adapter under "adapter"
    keys = set(obj)
    fan = obj.get("projector.fanout.weight")
    n_tokens = None if fan is None else int(fan.shape[0]) // int(c_kv)

    if any(k.startswith("projector.structure_adapter.") for k in keys):
        return {"name": "1x32", "projector": "clss", "n_tokens": n_tokens,
                "use_clss": True, "clss_model_name": "CLSS-sub.lckpt", "strip_bos_eos": True}
    if fan is not None:
        return {"name": "1x1536", "projector": "global_fanout", "n_tokens": n_tokens,
                "strip_bos_eos": True, "use_clss": False}
    stray = sorted(k for k in keys if k.startswith("projector."))
    if stray:
        raise ValueError(f"{ckpt_path}: projector.* keys match no known variant: {stray}")
    return {"name": "Nx1536", "projector": "identity", "resampler_tokens": None,
            "strip_bos_eos": True, "use_clss": False}


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
        return GlobalFanoutProjector(variant_cfg, c_kv=c_kv)
    if name == "clss":
        return CLSSProjector(variant_cfg, c_kv=c_kv)
    raise ValueError(f"unknown projector {name!r} (expected identity | global_fanout | clss)")
