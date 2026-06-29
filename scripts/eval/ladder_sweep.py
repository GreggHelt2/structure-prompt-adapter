"""Hydra entry point for the checkpoint-step adherence ladder (dev ``13_longer_training_decision.md``).

Evaluates a ladder of EXISTING SPA snapshots on one held-out prompt — adherence only, no OF3, no new
training — to plot adherence vs. training step (plateau ⇒ a longer run won't help; still rising ⇒
extend). Same sampling path as the flywheel driver (re-seeded per rollout), just run efficiently
(engine + ESM3 prompt computed once; only the adapter weights swap per checkpoint).

    conda run -n spa-dev python scripts/eval/ladder_sweep.py variant=C_n_by_1536 \\
        eval.prompt_pdb=/path/prompt.pdb eval.length=105 eval.num_designs=8 \\
        'eval.lambda_scale=[0.5,1,2]' \\
        'eval.ladder=[/tmp/spa_ladder/spa_C_step1000.pt,/tmp/spa_C_runA_step3000.pt,/tmp/spa_ladder/spa_C_step5000.pt,/tmp/spa_ladder/spa_C_step10000.pt,/tmp/spa_ladder/spa_C_step20000.pt,/tmp/spa_C_runA_final.pt]' \\
        eval.out_dir=/abs/out
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from spa.eval.ladder import run_ladder  # lazy import so `--cfg job` works pre-install

    OmegaConf.set_struct(cfg, False)  # run_ladder mutates eval.{conditions,lambda_scale,out_dir,...}
    out = run_ladder(cfg)
    print(f"ladder complete: {len(out['points'])} point(s) -> {out['results_path']}")


if __name__ == "__main__":
    main()
