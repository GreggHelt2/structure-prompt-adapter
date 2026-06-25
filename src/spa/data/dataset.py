"""Cache-backed CDDB training dataset — real RFD3 featurization (kickoff step 8 Part B).

The harness (``spa.train.harness``) consumes per-step **example dicts**; this dataset produces them
from real CDDB structures, replacing the local stand-in (``spa.data.synthetic``). Per item it:

1. parses the PDB (atomworks ``GenericDFParser`` — the same loader RFD3 training uses),
2. runs RFD3's **training featurization pipeline** (reconstructed from the checkpoint ``train_cfg`` +
   the foundry CDDB-monomer train yamls; MSA-free, unconditional / full-structure denoise),
3. assembles the example dict via RFD3's trainer mapping (``trainer/rfd3.py`` ``_assemble_network_inputs``
   / ``_assemble_loss_extra_info``), and
4. joins the cached ESM3 structural **prompt** for that structure.

**Live featurization** (dev ``09`` §8): featurization is ~98% deterministic but cheap (~150–380 ms)
and hides behind the GPU step with a couple of dataloader workers, so we run it live (fresh EDM noise
each access) rather than maintaining a second cache. Returns **CPU** tensors — the training loop moves
them to device (CUDA can't cross ``num_workers`` fork). The 6 Part-B foundry friction fixes are in
``build_train_transform`` (dev ``09`` §8).

**Augmentation note:** the pipeline applies RFD3-native SE(3) augmentation (``MotifCenterRandomAugmentation``
+ ``AugmentNoise``), so the harness's own ``se3_augment`` must be **off** for ``data=cddb`` (else double
augmentation). **Prompt alignment:** the cached prompt is the full-structure ESM3 embedding; it aligns to
the featurized tokens only when no crop occurs, i.e. ``length_cap <= crop_size`` (384). The 256 default
is safe; longer caps would need per-crop prompts.

The contract this yields (consumed by ``spa.train.harness.spa_training_step``):

    {
      "feats": dict, "t": Tensor[D], "X_noisy_L": Tensor[D,L,3],
      "X_gt_L_in_input_frame": Tensor[D,L,3], "crd_mask_L": BoolTensor, "is_original_unindexed_token": BoolTensor,
      "prompt": Tensor[D,N,1536],
    }
"""

from __future__ import annotations

import functools
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

_CIF_ARGS = {"cache_dir": None, "load_from_cache": False, "save_to_cache": False, "add_missing_atoms": False}


def _remap_targets(d):
    """Checkpoint cfg references ``projects.aa_design.*``; OSS foundry has them at ``rfd3.transforms.*``."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "_target_" and isinstance(v, str):
                d[k] = v.replace(
                    "projects.aa_design.transforms.training_conditions",
                    "rfd3.transforms.training_conditions",
                )
            else:
                _remap_targets(v)
    return d


@functools.lru_cache(maxsize=2)
def build_train_transform(rfd3_ckpt: str, foundry_train_cfg_dir: str, sigma_data: int = 16):
    """Reconstruct RFD3's CDDB-monomer **training** featurization pipeline (dev ``09`` §6/§8).

    Returns a Compose transform: parsed-structure dict -> example dict (``feats``, ``t``, ``noise``,
    ``coord_atom_lvl_to_be_noised``, ``ground_truth``). Unconditional (full-structure denoise), MSA-free.
    Cached per process (lru_cache) so it builds once per dataloader worker. The 6 Part-B friction fixes
    (dev ``09`` §8) are applied here.

    NOTE: loads the checkpoint once per process to read ``train_cfg`` (the small config, not for weights).
    TODO(opt): pre-extract ``train_cfg`` to a small file to avoid the per-worker ckpt read.
    """
    import hydra.utils
    from omegaconf import OmegaConf

    ck = torch.load(rfd3_ckpt, map_location="cpu", weights_only=False)
    tc = ck["train_cfg"]
    base = OmegaConf.load(os.path.join(foundry_train_cfg_dir, "pdb/base_transform_args.yaml"))
    mono = OmegaConf.load(os.path.join(foundry_train_cfg_dir, "rfd3_monomer_distillation.yaml"))
    tf = OmegaConf.merge(base.dataset.transform, mono.monomer_distillation.dataset.transform)

    node = OmegaConf.create(OmegaConf.to_container(tf, resolve=False))
    node._target_ = "rfd3.transforms.pipelines.build_atom14_base_pipeline"
    node.residue_cache_dir = None  # MACE-OFF23 cache absent (warning only; runs self-contained)
    node.b_factor_min = None       # CDDB B-factors are 0-1; the cfg's 70 would drop EVERY atom
    node.sigma_data = sigma_data   # avoid the ${model...sigma_data} interpolation
    node.return_atom_array = False  # training consumes tensors, not the atom array
    root = OmegaConf.create({"datasets": tc.datasets, "model": tc.model, "tf": node})
    cont = _remap_targets(OmegaConf.to_container(root, resolve=True)["tf"])
    # SPA MVP denoises the FULL structure -> unconditional path only; drop motif conditions
    # (island / tipatom / ppi / seq) and the external-binary hbond feature (hbplus).
    cont["train_conditions"] = {"unconditional": cont["train_conditions"]["unconditional"]}
    cont["meta_conditioning_probabilities"]["calculate_hbonds"] = 0.0
    return hydra.utils.instantiate(OmegaConf.create(cont))


@functools.lru_cache(maxsize=1)
def _parser():
    from atomworks.ml.datasets.parsers import GenericDFParser

    return GenericDFParser(pn_unit_iid_colnames=None)


def load_structure(example_id: str, pdb_path: str) -> dict:
    """Parse a PDB into the pipeline's input dict (matches RFD3 training ``__getitem__``)."""
    from atomworks.ml.datasets.parsers import load_example_from_metadata_row

    row = pd.Series({"example_id": example_id, "path": str(pdb_path), "assembly_id": "1"})
    d = load_example_from_metadata_row(row, _parser(), cif_parser_args=dict(_CIF_ARGS))
    aa = d["atom_array"]
    if "atom_id" in aa.get_annotation_categories():
        aa.del_annotation("atom_id")  # the train pipeline re-derives it; AddGlobalAtomIdAnnotation won't overwrite
    return d


def move_to_device(obj, device):
    """Recursively move torch tensors in a (possibly nested) example dict to ``device``; leave the rest."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


class CDDBPromptDataset(Dataset):
    """Yields harness example dicts from processed CDDB structures + the ESM3 prompt cache.

    Args:
        cfg: composed Hydra config (``data`` + ``paths`` groups give manifest, PDB, cache, ckpt roots).
        split: one of ``{"train", "validate", "test"}``.
    """

    def __init__(self, cfg, split: str = "train") -> None:
        self.cfg = cfg
        self.split = split
        self.pdb_dir = cfg.data.pdb_dir
        self.cache_dir = cfg.paths.esm3_cache_dir
        self.length_cap = cfg.data.get("length_cap", None)

        df = pd.read_parquet(os.path.join(cfg.data.splits_root, split, "manifest.parquet"))
        if self.length_cap is not None:
            df = df[df["length"] <= self.length_cap]
        if cfg.data.get("require_cached_prompt", False):
            df = df[df["pdb_file"].map(
                lambda f: os.path.exists(os.path.join(self.cache_dir, f"{os.path.splitext(f)[0]}.pt"))
            )]
        self.rows = df.reset_index(drop=True)

    @property
    def transform(self):
        return build_train_transform(self.cfg.paths.rfd3_ckpt, self.cfg.paths.foundry_train_cfg_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        out = self.transform(load_structure(row["id"], os.path.join(self.pdb_dir, row["pdb_file"])))

        coords = out["coord_atom_lvl_to_be_noised"]  # [D, L, 3] clean (augmented), tiled
        gt = out["ground_truth"]
        return {
            "feats": out["feats"],
            "t": out["t"],  # [D]
            "X_noisy_L": coords + out["noise"],  # [D, L, 3]
            "X_gt_L_in_input_frame": coords,  # no-align target (trainer rfd3.py:292)
            "crd_mask_L": gt["mask_atom_lvl"].bool(),  # bool (lDDT does an in-place bool multiply)
            "is_original_unindexed_token": gt["is_original_unindexed_token"].bool(),
            # prompt cache is keyed by the PDB-file stem (what spa.prompt.build_cache writes)
            "prompt": self._load_prompt(os.path.splitext(row["pdb_file"])[0], coords.shape[0]),  # [D, N, 1536]
        }

    def _load_prompt(self, pdb_stem: str, D: int) -> torch.Tensor:
        p = torch.load(os.path.join(self.cache_dir, f"{pdb_stem}.pt"), weights_only=True).float()  # [N, 1536]
        return p.unsqueeze(0).expand(D, -1, -1).contiguous()
