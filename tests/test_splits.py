"""Tests for CDDB split construction (kickoff step 8 Part A). CPU-only; no GPU/ckpt needed.

Core logic is tested on a synthetic DataFrame (fast, deterministic). One test against the real
CDDB metadata is skipped unless the parquet is present.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spa.data.splits import _stem, assign_splits

RATIOS = {"train": 0.8, "validate": 0.1, "test": 0.1}

_META = (
    "/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/"
    "atomistica_data_release/metadata/metadata_atomistica_syn_release.parquet"
)
_PDB = (
    "/home/user1/projects/spa/training_data/proteina-atomistica_data_vrelease/"
    "atomistica_data_release/pdb"
)


def _toy_df(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "id": [f"AF-X{i:06d}-F1-model_v4_pmpnn_esmfold" for i in range(n)],
        "length": rng.randint(33, 512, size=n),
        "plddt_avg": rng.uniform(80, 99, size=n),
        "plddt_std": rng.uniform(1, 16, size=n),
        "rmsd_ca": rng.uniform(0.2, 30, size=n),
    })


def test_stem_join():
    assert _stem("AF-A0A009E921-F1-model_v4_pmpnn_esmfold") == "AF-A0A009E921-F1-model_v4"


def test_deterministic_given_seed():
    df = _toy_df()
    a, _ = assign_splits(df, seed=0, ratios=RATIOS)
    b, _ = assign_splits(df, seed=0, ratios=RATIOS)
    pd.testing.assert_series_equal(a.set_index("id")["split"], b.set_index("id")["split"])


def test_seed_changes_assignment():
    df = _toy_df()
    a = assign_splits(df, seed=0, ratios=RATIOS)[0].set_index("id")["split"]
    b = assign_splits(df, seed=1, ratios=RATIOS)[0].set_index("id")["split"]
    assert (a != b).mean() > 0.1  # a meaningful fraction move splits


def test_disjoint_and_complete():
    df = _toy_df()
    out, _ = assign_splits(df, seed=0, ratios=RATIOS)
    assert set(out["split"]) == {"train", "validate", "test"}
    assert len(out) == len(df)
    assert out["id"].is_unique           # no id in two splits (leakage guard)
    assert out["split"].notna().all()    # every id assigned


def test_ratios_within_tolerance():
    out, _ = assign_splits(_toy_df(n=10000), seed=0, ratios=RATIOS)
    frac = out["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.02
    assert abs(frac["validate"] - 0.1) < 0.02
    assert abs(frac["test"] - 0.1) < 0.02


def test_stratify_balances_within_length_bands():
    out, _ = assign_splits(_toy_df(n=10000), seed=0, ratios=RATIOS, stratify_by="length", n_length_bins=10)
    q = pd.qcut(out["length"], 4, labels=False, duplicates="drop")
    for ql in np.unique(q):
        sub = out[q == ql]
        assert abs((sub["split"] == "train").mean() - 0.8) < 0.05


def test_rmsd_filter_drops_rows():
    out, filt = assign_splits(_toy_df(), seed=0, ratios=RATIOS, filter_rmsd_ca_max=2.0)
    assert (out["rmsd_ca"] <= 2.0).all()
    assert filt["n_dropped"] > 0
    assert filt["filter_rmsd_ca_max"] == 2.0


def test_no_filter_keeps_all():
    out, filt = assign_splits(_toy_df(), seed=0, ratios=RATIOS)
    assert filt["n_dropped"] == 0
    assert filt["n_kept"] == filt["n_input"]


@pytest.mark.skipif(not os.path.exists(_META), reason="CDDB metadata parquet not present")
def test_real_metadata_join_sample():
    ids = pd.read_parquet(_META, columns=["id"]).head(50)["id"]
    for mid in ids:
        f = _stem(mid) + "_esmfold_v1.pdb"
        assert os.path.exists(os.path.join(_PDB, f)), f
