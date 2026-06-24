"""Hydra entry point for ESM3 prompt-cache generation. Kickoff step 5 (local) / step 9 (cloud).

    conda run -n spa-dev python scripts/gen_esm3_cache.py data=toy          # small local cache
    conda run -n spa-dev python scripts/gen_esm3_cache.py data=cddb hardware=cloud_h100

Runs frozen local ESM3 over a dataset split and writes per-residue (N,1536) prompts to the cache
(fp16). On the A5000 this produces only a small test cache; the full ~251 GB cache is generated
on a cloud H100 -> GCS (dev ``04`` §10). See ``spa.prompt.esm3_prompt``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="cache")
def main(cfg: DictConfig) -> None:
    # lazy import so `--cfg job` works pre-install
    raise NotImplementedError(
        "ESM3 cache generation is a step-1 scaffold; implement in kickoff step 5 "
        "(dev 01_codebase_analysis.md §3.5, 04_training_strategy.md §10)."
    )


if __name__ == "__main__":
    main()
