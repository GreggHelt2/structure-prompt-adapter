"""Hydra entry point: build leakage-safe train/validate/test splits over CDDB (kickoff step 8 Part A).

    conda run -n spa-dev python scripts/build_splits.py
    conda run -n spa-dev python scripts/build_splits.py data.split.seed=1 data.split.ratios.train=0.9

CDDB is one structure per Foldseek (structural) cluster -> already structurally de-duplicated, so a
simple length-stratified random split is leakage-safe (dev ``04`` §2 / ``07`` I.9). Writes a master
manifest + per-split manifests + ``split_meta.json`` under ``training_data/``. NO clustering / NO
MMseqs2. See ``spa.data.splits``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="splits")
def main(cfg: DictConfig) -> None:
    from spa.data.splits import build_splits  # lazy import so `--cfg job` works pre-install

    out = build_splits(cfg)
    c = out["counts"]
    total = sum(c.values())
    print(f"splits -> {out['master']}")
    print(
        f"  train={c['train']} ({c['train']/total:.1%})  "
        f"validate={c['validate']} ({c['validate']/total:.1%})  "
        f"test={c['test']} ({c['test']/total:.1%})  total={total}"
    )
    print(f"  meta -> {out['meta']}")


if __name__ == "__main__":
    main()
