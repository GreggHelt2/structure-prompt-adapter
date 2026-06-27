"""Hydra entry point for SPA Stage-1 design generation (dev 05_validation_pipeline.md Stage 0).

Mirrors ``scripts/train.py``: a lazy import keeps ``--cfg job`` working pre-install.

    conda run -n spa-dev python scripts/eval/generate.py --cfg job          # inspect composed config
    # baseline (vanilla RFD3) + SPA, λ-sweep, K=8 designs each, from one prompt structure:
    conda run -n spa-dev python scripts/eval/generate.py \
        variant=C_n_by_1536 eval.ckpt=/path/spa_C_final.pt \
        eval.prompt_pdb=/path/prompt.pdb 'eval.conditions=[baseline,spa]' \
        'eval.lambda_scale=[0.5,1.0]' eval.num_designs=8 eval.length=100

All knobs (variant, K, λ, length, sampler steps, prompt, ckpt, out_dir, paths) are Hydra overrides —
see ``configs/eval.yaml`` and the ``configs/eval/default.yaml`` group it composes.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from spa.eval.generate import generate  # lazy import so `--cfg job` works pre-install

    designs = generate(cfg)
    print(f"generated {len(designs)} design(s) under {cfg.eval.out_dir}")


if __name__ == "__main__":
    main()
