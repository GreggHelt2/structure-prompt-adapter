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
    from spa.prompt import build_cache  # lazy import so `--cfg job` works pre-install

    stats = build_cache(cfg)
    mb = stats["bytes"] / 1e6
    print(
        f"ESM3 cache -> {stats['out_dir']}: {stats['n_done']} written "
        f"({mb:.1f} MB), {stats['n_skipped']} skipped, {stats['n_too_long']} over length cap, "
        f"{stats['n_failed']} failed, in {stats['seconds']}s."
    )


if __name__ == "__main__":
    main()
