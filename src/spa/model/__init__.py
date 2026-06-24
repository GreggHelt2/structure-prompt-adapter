"""SPA model components: cross-attention, wrapper/side-channel, projectors, loader."""

from .cross_attention import SPACrossAttention, SPAPromptKV

__all__ = ["SPACrossAttention", "SPAPromptKV"]
