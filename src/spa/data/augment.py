"""SE(3) data augmentation for SPA training targets (dev 03 §7, 04).

RFD3 obtains SE(3) behavior by random-rotation augmentation of the target coordinates; the ESM3
prompt is SE(3)-invariant (verified step 5: rotation cosine 0.99998), so it is cached once and
reused across augmentations. These helpers apply a random proper rotation (optionally a
translation) to a coordinate tensor, NaN-safe (atom37 representations use NaN for missing atoms).
"""

from __future__ import annotations

import torch


def random_rotation(generator: torch.Generator | None = None) -> torch.Tensor:
    """A uniformly-random proper rotation (det +1) in SO(3), via QR of a Gaussian. CPU float32 [3,3]."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))   # make the QR sign-unique (proper orthogonal)
    if torch.det(q) < 0:                     # ensure det +1 (rotation, not reflection)
        q[:, 0] = -q[:, 0]
    return q


def apply_se3(coords: torch.Tensor, R: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
    """Apply rotation ``R`` ``[3,3]`` (and optional translation ``t`` ``[3]``) to ``coords[...,3]``, NaN-safe."""
    finite = torch.isfinite(coords)
    R = R.to(coords.device, coords.dtype)
    out = torch.einsum("...c,cd->...d", torch.nan_to_num(coords), R)
    if t is not None:
        out = out + t.to(coords.device, coords.dtype)
    out[~finite] = float("nan")              # preserve missing-atom NaNs
    return out


def augment_coords(coords: torch.Tensor, generator: torch.Generator | None = None,
                   translate: float = 0.0) -> torch.Tensor:
    """Random SE(3) augmentation of a coordinate tensor (random rotation; optional uniform shift)."""
    R = random_rotation(generator)
    t = None
    if translate:
        t = (torch.rand(3, generator=generator) * 2 - 1) * translate
    return apply_se3(coords, R, t)
