"""ESM3 prompt producer (kickoff step 5), against the real local ESM3 weights.

Verifies the prompt shape/stripping, that ESM3 is frozen, the SE(3) invariance the cache-once design
relies on (dev W1.1), and the cache driver. Skipped unless ESM (and the cached weights) + a CUDA
device are available, so the suite stays green elsewhere.
"""

import glob
import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("esm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ESM3 prompt test needs CUDA")

TOY_DIR = os.environ.get("SPA_TOY_DIR", os.path.expanduser("~/projects/spa/training_data/toy"))


def _toy_pdb():
    pdbs = sorted(glob.glob(os.path.join(TOY_DIR, "*.pdb")))
    if not pdbs:
        pytest.skip(f"no toy PDBs in {TOY_DIR}")
    return pdbs[0]


@pytest.fixture(scope="module")
def esm3():
    from spa.prompt import load_esm3

    return load_esm3()


def test_prompt_shape_and_stripping(esm3):
    from spa.prompt import esm3_prompt

    pdb = _toy_pdb()
    full = esm3_prompt(pdb, esm3, strip_bos_eos=False)
    stripped = esm3_prompt(pdb, esm3, strip_bos_eos=True)
    assert full.ndim == 2 and full.shape[1] == 1536
    assert stripped.shape[0] == full.shape[0] - 2  # BOS/EOS removed
    assert stripped.shape[1] == 1536


def test_esm3_is_frozen(esm3):
    assert all(not p.requires_grad for p in esm3.parameters())


def test_se3_rotation_invariance(esm3):
    # Cache-once + augment relies on the prompt being SE(3)-invariant (dev W1.1): cosine > 0.999.
    from esm.sdk.api import ESMProtein
    from spa.prompt import esm3_prompt

    coords = ESMProtein.from_pdb(_toy_pdb(), chain_id="detect").coordinates
    g = torch.Generator().manual_seed(0)
    R = torch.linalg.qr(torch.randn(3, 3, generator=g))[0]
    finite = torch.isfinite(coords)
    rot = torch.einsum("lac,cd->lad", torch.nan_to_num(coords), R)
    rot[~finite] = float("nan")

    base = esm3_prompt(coords, esm3).float()
    rotated = esm3_prompt(rot, esm3).float()
    cos = torch.nn.functional.cosine_similarity(base, rotated, dim=-1)
    assert cos.mean().item() > 0.999


def test_build_cache(esm3, tmp_path):
    from spa.prompt import build_cache

    cfg = OmegaConf.create(
        {
            "data": {"root": TOY_DIR, "length_cap": 4096},
            "hardware": {"device": "cuda:0"},
            "out_dir": str(tmp_path / "esm3_cache"),
            "dtype": "float16",
            "strip_bos_eos": True,
        }
    )
    stats = build_cache(cfg, esm3_model=esm3)
    assert stats["n_done"] > 0
    files = sorted(glob.glob(os.path.join(cfg.out_dir, "*.pt")))
    assert len(files) == stats["n_done"]
    emb = torch.load(files[0])
    assert emb.dtype == torch.float16 and emb.ndim == 2 and emb.shape[1] == 1536

    # Re-running skips already-cached structures.
    again = build_cache(cfg, esm3_model=esm3)
    assert again["n_skipped"] >= stats["n_done"] and again["n_done"] == 0
