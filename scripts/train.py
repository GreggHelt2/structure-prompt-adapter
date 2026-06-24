"""Hydra entry point for SPA training. Kickoff steps 6–7.

    conda run -n spa-dev python scripts/train.py --cfg job          # inspect composed config
    conda run -n spa-dev python scripts/train.py variant=C_n_by_1536 hardware=local_a5000

All knobs (variant, hardware, batch, lr, CFG drop-rate, λ, paths) are Hydra overrides — see
``configs/train.yaml`` and the config groups it composes.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from spa.train.harness import train  # lazy import so `--cfg job` works pre-install

    train(cfg)


if __name__ == "__main__":
    main()
