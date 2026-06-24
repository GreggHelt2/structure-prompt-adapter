"""SE(3) augmentation utilities (kickoff step 6 support). Pure-CPU; always runs."""

import torch

from spa.data.augment import apply_se3, augment_coords, random_rotation


def test_random_rotation_is_proper_orthogonal():
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        R = random_rotation(g)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)   # orthogonal
        assert abs(torch.det(R).item() - 1.0) < 1e-5              # proper (det +1)


def test_rotation_preserves_pairwise_distances():
    g = torch.Generator().manual_seed(1)
    coords = torch.randn(20, 3)
    rot = apply_se3(coords, random_rotation(g))

    def pdist(x):
        return torch.cdist(x, x)

    assert torch.allclose(pdist(coords), pdist(rot), atol=1e-4)


def test_nan_positions_preserved():
    coords = torch.randn(8, 37, 3)
    coords[3, 20:] = float("nan")          # missing atoms
    out = augment_coords(coords, torch.Generator().manual_seed(2))
    assert torch.equal(torch.isnan(out), torch.isnan(coords))


def test_translation_applied():
    coords = torch.zeros(4, 3)
    R = torch.eye(3)
    out = apply_se3(coords, R, t=torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]).expand(4, 3))
