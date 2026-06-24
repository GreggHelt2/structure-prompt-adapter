"""Config-driven device selection.

Portability invariant (repo CLAUDE.md / dev ``04`` portability rule): device selection comes
from ``configs/hardware/*.yaml`` (e.g. ``device: cuda:0``), NEVER a hardcoded GPU UUID. On the
local box the A5000 is pinned by ``CUDA_VISIBLE_DEVICES=GPU-...`` *baked into the conda env vars*
— an env-level concern the runtime never sees in code, so after masking the A5000 is just
``cuda:0``. The single cloud H100 is likewise ``cuda:0`` with no masking. Hence the same
``device: cuda:0`` string works on both, and local-A5000 -> cloud-H100 is a config change.

This module is implemented (not a stub) because it *is* the portability boundary.
"""

from __future__ import annotations


def resolve_device(device: str = "cuda:0"):
    """Resolve a config device string to a ``torch.device``.

    Args:
        device: a device string from ``configs/hardware/*.yaml`` (``"cuda:0"``, ``"cpu"``, ...).
            ``"auto"`` picks CUDA when available, else CPU.

    Returns:
        ``torch.device``.

    Raises:
        RuntimeError: a CUDA device was requested but CUDA is unavailable (e.g. the A5000 UUID
            masking was not applied — check the conda env vars).
    """
    import torch  # lazy so importing this module never requires torch

    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requested device {device!r} but CUDA is unavailable. On the local box, "
            "confirm CUDA_VISIBLE_DEVICES=GPU-<A5000-UUID> is set (baked into the conda env)."
        )
    return dev
