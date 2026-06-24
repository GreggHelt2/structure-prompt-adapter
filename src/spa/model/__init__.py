"""SPA model components: cross-attention, wrapper/side-channel, projectors, loader."""

from .cross_attention import SPACrossAttention, SPAPromptKV
from .loader import attach_spa, freeze_host
from .projectors import IdentityProjector, make_projector
from .wrapper import SPAAdapter, SPAContext, SPAWrappedAttention

__all__ = [
    "SPACrossAttention",
    "SPAPromptKV",
    "SPAWrappedAttention",
    "SPAContext",
    "SPAAdapter",
    "IdentityProjector",
    "make_projector",
    "attach_spa",
    "freeze_host",
]
