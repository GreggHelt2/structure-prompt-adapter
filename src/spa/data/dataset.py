"""Cache-backed dataset: pairs a target structure with its precomputed ESM3 prompt.

Kickoff step 6. Spec: dev ``04_training_strategy.md`` §5–§6.

Each training example is ``(target_structure, esm3_prompt)`` where the prompt was precomputed by
``spa.prompt.esm3_prompt`` / ``scripts/gen_esm3_cache.py`` and read from the cache rather than
recomputed (running ESM3 live each step would be prohibitively slow). SE(3) augmentation is
applied to the TARGET coordinates at load time (the prompt is invariant, so it is reused across
rotations — dev ``03`` §7 / ``04``). CFG zero-prompt dropout is applied in the harness, not here.

Locations come from ``configs/data/*.yaml`` + ``configs/paths/default.yaml`` (toy vs cddb;
local small cache vs NVMe-mounted cloud cache) — never hardcoded.
"""

from __future__ import annotations

from torch.utils.data import Dataset


class SPADataset(Dataset):
    """Yields ``(target, prompt)`` from a manifest + ESM3 prompt cache.

    Args:
        cfg: composed Hydra config (``data`` + ``paths`` groups give manifest + cache roots).
        split: one of ``{"train", "validate", "test"}``.
    """

    def __init__(self, cfg, split: str = "train") -> None:
        # TODO(step 6): read split manifest; resolve cache paths from cfg; set up augmentation.
        raise NotImplementedError(
            "SPADataset is a step-1 scaffold; implement in kickoff step 6 "
            "(dev 04_training_strategy.md §5–§6)."
        )

    def __len__(self) -> int:
        raise NotImplementedError("kickoff step 6")

    def __getitem__(self, idx: int):
        raise NotImplementedError("kickoff step 6")
