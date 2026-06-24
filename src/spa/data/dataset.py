"""Cache-backed training dataset — the seam between featurization and the harness.

Kickoff step 8 (the real CDDB featurization). The harness (``spa.train.harness``) consumes per-step
**example dicts**; a dataset's job is to produce them. The contract (matching RFD3's own training
example + the SPA prompt; see ``spa.train.harness.spa_training_step``):

    {
      "feats": dict,                       # RFD3 features `f` (atom_to_token_map, is_polar,
                                           #   is_ligand, is_virtual, is_sidechain, is_dna, is_rna, …)
      "t": Tensor[D],                      # EDM timestep / sigma
      "X_noisy_L": Tensor[D, L, 3],        # noised atom coords (= coord_to_be_noised + noise)
      "X_gt_L_in_input_frame": Tensor[D, L, 3],   # ground-truth atom coords (target)
      "crd_mask_L": BoolTensor[L],         # resolved-atom mask
      "is_original_unindexed_token": BoolTensor[I],
      "prompt": Tensor[D, N, 1536] | None, # ESM3 structural prompt (from the cache), or None (CFG)
    }

For the LOCAL overfit, ``spa.data.synthetic.capture_synthetic_example`` produces one such dict from
a real captured RFD3 batch. The real dataset below — reading processed CDDB structures + the ESM3
prompt cache (``spa.prompt.build_cache``) and applying RFD3's training featurization + EDM noising —
is built in **kickoff step 8** once the processed dataset + leakage-safe splits exist (dev ``04``).
"""

from __future__ import annotations

from torch.utils.data import Dataset


class CDDBPromptDataset(Dataset):
    """Yields harness example dicts from processed CDDB structures + the ESM3 prompt cache.

    Args:
        cfg: composed Hydra config (``data`` + ``paths`` groups give manifest + cache roots).
        split: one of ``{"train", "validate", "test"}``.
    """

    def __init__(self, cfg, split: str = "train") -> None:
        raise NotImplementedError(
            "real CDDB featurization (RFD3 training transform + EDM noising) is kickoff step 8; "
            "use spa.data.synthetic.capture_synthetic_example for the local overfit (step 6/7)."
        )

    def __len__(self) -> int:
        raise NotImplementedError("kickoff step 8")

    def __getitem__(self, idx: int):
        raise NotImplementedError("kickoff step 8")
