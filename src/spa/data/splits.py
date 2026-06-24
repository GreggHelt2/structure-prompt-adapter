"""Build leakage-safe train/validate/test splits over CDDB (kickoff step 8 Part A).

CDDB is one structure per Foldseek (structural) cluster representative of AFDB -> already
structurally de-duplicated by construction (dev ``04`` §2 / ``07`` I.9). So a simple
length-stratified random partition is *already* a leakage-safe, cluster-level split at the
structural level -- **NO clustering / NO MMseqs2**. Filtering is deliberately light (dev ``04``
§1): pLDDT/length were applied at construction, and the ``rmsd_ca`` designability cut is SKIPPED
by default (the CDDB paper's own ablation shows it hurts diversity & co-designability). The length
cap is a *runtime* dataloader view, NOT baked into the splits, so one partition serves both the
A5000 and the H100.

Outputs (under ``training_data/``, outside the repos):
  - master:    ``<master_manifest>``                 every id + metadata + assigned split
  - per-split: ``<splits_root>/{split}/manifest.parquet``
  - meta:      ``<processed>/split_meta.json``        seed, ratios, filters, counts, timestamp
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_SPLITS = ("train", "validate", "test")


def _stem(meta_id: str) -> str:
    """metadata id ``AF-<acc>-F1-model_v4_pmpnn_esmfold`` -> file stem ``AF-<acc>-F1-model_v4``."""
    return meta_id.replace("_pmpnn_esmfold", "")


def load_metadata(cfg) -> pd.DataFrame:
    """Read the CDDB metadata parquet and add the PDB-join columns (``stem``, ``pdb_file``)."""
    df = pd.read_parquet(cfg.data.metadata_parquet).copy()
    df["stem"] = df["id"].map(_stem)
    df["pdb_file"] = df["stem"] + "_esmfold_v1.pdb"
    return df


def assign_splits(
    df: pd.DataFrame,
    *,
    seed: int,
    ratios: dict,
    stratify_by: str | None = "length",
    n_length_bins: int = 20,
    filter_rmsd_ca_max: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assign each row a ``split`` label deterministically. Returns ``(df_with_split, filter_notes)``.

    Leakage-safe by construction (one structure per structural cluster). Stratifying by length
    makes the train/validate/test ratio hold *within* every length band, so any length-capped
    runtime view stays ~balanced. ``test = 1 - train - validate`` (the remainder).
    """
    n0 = len(df)
    if filter_rmsd_ca_max is not None:
        df = df[df["rmsd_ca"] <= filter_rmsd_ca_max]
    if min_length is not None:
        df = df[df["length"] >= min_length]
    if max_length is not None:
        df = df[df["length"] <= max_length]
    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)  # deterministic input order
    filt = {
        "n_input": int(n0), "n_kept": int(len(df)), "n_dropped": int(n0 - len(df)),
        "filter_rmsd_ca_max": filter_rmsd_ca_max, "min_length": min_length, "max_length": max_length,
    }

    r_tr, r_va = float(ratios["train"]), float(ratios["validate"])
    rng = np.random.RandomState(int(seed))

    if stratify_by == "length":
        ranks = df["length"].rank(method="first").to_numpy()
        bins = np.minimum((ranks - 1) / len(df) * n_length_bins, n_length_bins - 1e-9).astype(int)
        groups = [np.where(bins == b)[0] for b in range(n_length_bins)]
    elif stratify_by is None:
        groups = [np.arange(len(df))]
    else:
        raise ValueError(f"unknown stratify_by={stratify_by!r} (use 'length' or null)")

    split = np.empty(len(df), dtype=object)
    for idx in groups:
        idx = idx.copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * r_tr))
        n_va = int(round(n * r_va))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "validate"
        split[idx[n_tr + n_va:]] = "test"

    df = df.copy()
    df["split"] = split
    return df, filt


def write_manifests(df: pd.DataFrame, cfg, filt: dict) -> dict:
    """Write the master manifest, the per-split manifests, and ``split_meta.json``."""
    cols = ["id", "stem", "pdb_file", "length", "plddt_avg", "plddt_std", "rmsd_ca", "split"]
    master = df[cols]
    os.makedirs(os.path.dirname(cfg.data.master_manifest), exist_ok=True)
    master.to_parquet(cfg.data.master_manifest, index=False)

    counts = {}
    for s in _SPLITS:
        sub = master[master["split"] == s].reset_index(drop=True)
        d = os.path.join(cfg.data.splits_root, s)
        os.makedirs(d, exist_ok=True)
        sub.to_parquet(os.path.join(d, "manifest.parquet"), index=False)
        counts[s] = int(len(sub))

    meta = {
        "n_total": int(len(master)),
        "counts": counts,
        "ratios": {k: float(v) for k, v in cfg.data.split.ratios.items()},
        "seed": int(cfg.data.split.seed),
        "stratify_by": cfg.data.split.get("stratify_by"),
        "n_length_bins": int(cfg.data.split.n_length_bins),
        "filters": filt,
        "pdb_dir": str(cfg.data.pdb_dir),
        "metadata_parquet": str(cfg.data.metadata_parquet),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = os.path.join(os.path.dirname(cfg.data.master_manifest), "split_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"counts": counts, "master": str(cfg.data.master_manifest), "meta": meta_path}


def build_splits(cfg) -> dict:
    """Compose the full Part-A pipeline from a Hydra config (see ``scripts/build_splits.py``)."""
    sp = cfg.data.split
    df = load_metadata(cfg)
    df, filt = assign_splits(
        df,
        seed=int(sp.seed),
        ratios={k: float(v) for k, v in sp.ratios.items()},
        stratify_by=sp.get("stratify_by"),
        n_length_bins=int(sp.n_length_bins),
        filter_rmsd_ca_max=None if sp.get("filter_rmsd_ca_max") is None else float(sp.filter_rmsd_ca_max),
        min_length=None if sp.get("min_length") is None else int(sp.min_length),
        max_length=None if sp.get("max_length") is None else int(sp.max_length),
    )
    return write_manifests(df, cfg, filt)
