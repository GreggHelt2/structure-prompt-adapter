"""SPA cross-attention: the decoupled cross-attention term added to each wrapped RFD3 block.

Kickoff step 2. Spec: dev ``03_spa_architecture.md`` §2, §5, §6 and ``02_attachment_points.md`` §4.

The module is split into two pieces so the *shared K/V* recommendation (dev ``03`` §6 — the prompt
is identical across all 18 blocks and all 200 diffusion steps, so its K/V can be projected ONCE)
falls out naturally, while the per-block-K/V ablation (``03`` §9.1) is just "instantiate one
:class:`SPAPromptKV` per block":

    SPAPromptKV       prompt P' [D, M, 1536]  --(RMSNorm? + Wk/Wv)-->  K, V [D, M, c_model]
                      (shared: built once per design, reused by every block & step)

    SPACrossAttention query A_I [D, I, 768]   --Wq-->  q [D, I, c_model]
                      attn = softmax(q Kᵀ/√d_head + key_pad_mask) V          [D, I, c_model]
                      spa  = Wo · attn          [D, I, c_model] -> [D, I, 768]   (Wo ZERO-INIT)
                      return  λ · spa
                      (per block: owns Wq, Wo, and the inference scale λ)

**Identity-at-init (the standing correctness gate).** ``Wo`` is zero-initialized, so ``spa == 0``
at step 0 regardless of how Wq/Wk/Wv are initialized -> a wrapped block reproduces vanilla RFD3
exactly. SPA cannot warm-start like IP-Adapter (the prompt is 1536-d, RFD3 K/V are 768-d), so
zero-init is the identity lever (dev ``02`` §4, ``03`` §5). ``λ`` is the inference knob
(IP-Adapter ``set_scale``): a non-learned buffer, 1.0 during training, swept at inference
(0 -> unconditional). Identity comes from ``Wo``, not ``λ``, so λ stays free.
"""

from __future__ import annotations

import torch
from torch import nn


class SPAPromptKV(nn.Module):
    """Project the structural prompt to keys/values: ``[D, M, c_kv] -> [D, M, c_model]`` (×2).

    Shared across the 18 wrapped blocks (computed once per design; the prompt is constant across
    blocks and diffusion steps — dev ``03`` §6). For the per-block-K/V ablation, instantiate one
    per block instead. The optional input RMSNorm normalizes ESM3's *pre-final-LayerNorm*
    embeddings before projection (dev ``01`` §3.3, ``03`` §2).

    Args:
        c_kv: prompt width (ESM3 hidden = 1536).
        c_model: internal attention width (must match :class:`SPACrossAttention`).
        input_rmsnorm: RMSNorm the prompt before K/V projection.
    """

    def __init__(self, c_kv: int = 1536, c_model: int = 768, input_rmsnorm: bool = True) -> None:
        super().__init__()
        self.norm: nn.Module = nn.RMSNorm(c_kv) if input_rmsnorm else nn.Identity()
        self.to_k = nn.Linear(c_kv, c_model, bias=False)
        self.to_v = nn.Linear(c_kv, c_model, bias=False)
        # Normal init — K/V projections are NOT the identity-at-init lever (Wo is).
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # Learned null token e∅ for CFG zero-prompt dropout — the faithful IP-Adapter recipe
        # (dev ``11`` §6, ``07`` Q1.1): on a dropped training step the prompt is REPLACED by this
        # trainable token (the cross-attn stays live), so a gradient still reaches the adapter —
        # unlike bypassing the module, which zeroes the gradient and crashed the B1 smoke. Init
        # NON-zero: a zero token would, with these bias-free K/V projections, give all-zero K/V and
        # no gradient (the "thin null" failure, ``11`` §5). Identity-at-init is preserved by
        # SPACrossAttention's zero-init ``Wo``, not by this token, so e∅ is free to be nonzero.
        self.null_token = nn.Parameter(torch.empty(1, 1, c_kv))
        nn.init.normal_(self.null_token, std=0.02)

    def forward(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``prompt`` ``[D, M, c_kv]`` -> ``(k, v)`` each ``[D, M, c_model]``."""
        p = self.norm(prompt)
        return self.to_k(p), self.to_v(p)

    def null_kv(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V for the learned null prompt e∅ — CFG zero-prompt dropout / a learned unconditional.

        Returns a single null key/value (``M=1``) broadcast over ``batch``: ``(k, v)`` each
        ``[batch, 1, c_model]``. ``batch`` must equal the query's diffusion batch ``D`` — the
        cross-attn reshapes K/V with the query's batch dim, so the null can't rely on implicit
        broadcast there (dev ``11`` §6).
        """
        return self.forward(self.null_token.expand(batch, -1, -1))


class SPACrossAttention(nn.Module):
    """Per-block decoupled multi-head cross-attention over the M prompt tokens.

    Owns the per-block query/output projections (Wq, Wo) and the inference scale λ. Receives
    keys/values already projected to ``c_model`` (from a shared :class:`SPAPromptKV`). Dense
    attention over M prompt tokens — NOT RFD3's 4D local/k-NN self-attention form (dev ``02`` §4).

    Args:
        c_query: token-feature width (RFD3 ``c_token`` = 768).
        c_model: internal attention width (dev ``03`` §6: 768 recommended, 384 if VRAM-tight).
        n_head: number of attention heads (``c_model`` must be divisible by it).
        lambda_init: initial value of the inference scale λ (1.0 for training).

    Forward returns the SPA term ``λ · Wo · attn(Wq·query, k, v)`` with shape ``[D, I, c_query]``;
    it is exactly zero at init (zero-init ``Wo``).
    """

    def __init__(
        self,
        c_query: int = 768,
        c_model: int = 768,
        n_head: int = 8,
        lambda_init: float = 1.0,
    ) -> None:
        super().__init__()
        if c_model % n_head != 0:
            raise ValueError(f"c_model={c_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = c_model // n_head
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(c_query, c_model, bias=True)
        self.to_out = nn.Linear(c_model, c_query, bias=True)
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.zeros_(self.to_q.bias)
        # Zero-init the output projection -> identity-at-init (weight AND bias zero so the term is
        # exactly 0, not merely small). This is the correctness gate; see module docstring.
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # λ is the inference knob (not learned): a buffer so it rides the module to device / into
        # the SPA checkpoint, settable via set_scale().
        self.register_buffer("lambda_scale", torch.tensor(float(lambda_init)))

    def set_scale(self, value: float) -> None:
        """Set the inference scale λ (IP-Adapter ``set_scale``). 0 -> unconditional."""
        self.lambda_scale.fill_(float(value))

    def forward(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the SPA term.

        Args:
            query: token features ``[D, I, c_query]`` (the frozen block's own ``A_I``).
            k: prompt keys ``[D, M, c_model]`` (from :class:`SPAPromptKV`).
            v: prompt values ``[D, M, c_model]``.
            key_padding_mask: optional ``[D, M]`` boolean mask, ``True`` at PAD prompt positions
                to be ignored (PyTorch ``nn.MultiheadAttention`` convention).

        Returns:
            ``[D, I, c_query]`` — ``λ · Wo · attn``; exactly zero at init.
        """
        D, I, _ = query.shape
        M = k.shape[1]

        q = self.to_q(query).view(D, I, self.n_head, self.head_dim).transpose(1, 2)  # [D,h,I,hd]
        k = k.view(D, M, self.n_head, self.head_dim).transpose(1, 2)                 # [D,h,M,hd]
        v = v.view(D, M, self.n_head, self.head_dim).transpose(1, 2)                 # [D,h,M,hd]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale                   # [D,h,I,M]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        # Softmax over the M prompt tokens, in fp32 for bf16 stability.
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)                                                  # [D,h,I,hd]
        out = out.transpose(1, 2).reshape(D, I, self.n_head * self.head_dim)         # [D,I,c_model]
        out = self.to_out(out)                                                       # [D,I,c_query]
        # Cast λ to out's dtype so the term stays in the host's compute dtype (e.g. bf16 under
        # autocast); an fp32 λ would promote base+spa to fp32 and break the identity gate.
        return out * self.lambda_scale.to(out.dtype)
