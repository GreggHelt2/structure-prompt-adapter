"""Hydra entry point for SPA Stage-2 inverse folding (dev 05_validation_pipeline.md Stage 2).

Mirrors ``scripts/eval/generate.py``: a lazy import keeps ``--cfg job`` working pre-install.
Reads the ``eval.proteinmpnn`` knobs and inverse-folds the Stage-1 design PDBs (by default the
``*.pdb`` under ``eval.out_dir``) into N sequences each with ProteinMPNN, one FASTA per backbone.

    conda run -n spa-dev python scripts/eval/inverse_fold.py --cfg job        # inspect composed config
    # inverse-fold every Stage-1 design under eval.out_dir, 8 seqs each at temp 0.1:
    conda run -n spa-dev python scripts/eval/inverse_fold.py \
        eval.proteinmpnn.num_seqs=8 eval.proteinmpnn.sampling_temp=0.1
    # or a specific dir of design PDBs:
    conda run -n spa-dev python scripts/eval/inverse_fold.py \
        eval.proteinmpnn.design_dir=/path/designs eval.proteinmpnn.out_dir=/path/seqs

All knobs (num_seqs, sampling_temp, model_name, weights_dir, designs/design_dir, out_dir, conda_env)
are Hydra overrides — see ``configs/eval/default.yaml`` (the ``proteinmpnn`` block) and
``configs/paths/default.yaml`` (``proteinmpnn_repo``).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from spa.eval.proteinmpnn import inverse_fold  # lazy import so `--cfg job` works pre-install

    results = inverse_fold(cfg)
    total = sum(len(r.sequences) for r in results)
    print(f"inverse-folded {len(results)} design(s) -> {total} sequence(s)")


if __name__ == "__main__":
    main()
