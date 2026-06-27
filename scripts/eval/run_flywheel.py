"""Hydra entry point for the full SPA validation flywheel (dev 05_validation_pipeline.md §1–§4).

Mirrors ``scripts/eval/generate.py``: a lazy import keeps ``--cfg job`` working pre-install. Wires
Stage 1 (generate RFD3 ± SPA) → Stage 2 (ProteinMPNN inverse fold) → Stage 3 (OpenFold3 refold —
**pluggable + stubbed**: skipped unless ``eval.flywheel.refolder`` is configured) → Stage 4 (score
adherence + designability, aggregate, Δ vs baseline), writing ``eval.out_dir/flywheel_results.json``.

    conda run -n spa-dev python scripts/eval/run_flywheel.py --cfg job        # inspect composed config
    # full flywheel for a trained adapter: baseline + SPA, λ-sweep, K designs, adherence vs a prompt:
    conda run -n spa-dev python scripts/eval/run_flywheel.py \
        variant=C_n_by_1536 eval.ckpt=/path/spa_C_final.pt \
        eval.prompt_pdb=/path/prompt.pdb 'eval.conditions=[baseline,spa]' \
        'eval.lambda_scale=[0.5,1.0]' eval.num_designs=8 eval.length=100

OF3 designability stays off until a refolder is wired (``eval.flywheel.refolder=<_target_ spec>`` or
the ``run_flywheel(cfg, refolder=...)`` API). All other knobs (variant, K, λ, length, sampler steps,
prompt, ckpt, ProteinMPNN, score thresholds, out_dir, paths) are Hydra overrides — see
``configs/eval/default.yaml``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from spa.eval.flywheel import run_flywheel  # lazy import so `--cfg job` works pre-install

    out = run_flywheel(cfg)
    print(f"flywheel complete: {len(out['designs'])} design(s) -> results {out['results_path']}")


if __name__ == "__main__":
    main()
