"""Stage-2 inverse-folding smoke test (dev 05_validation_pipeline.md "Stage 2 — Inverse fold").

Exercises ``spa.eval.proteinmpnn`` end-to-end against the real ProteinMPNN: build a trivial poly-G
backbone PDB (no RFD3 / no CUDA needed), run ProteinMPNN for N=2 sequences via the config-driven
``inverse_fold`` (which calls the ``run_proteinmpnn`` subprocess worker), and assert the file-based
PDB->FASTA handoff produced N valid sequences of the right length.

Gated exactly like the other real-tool tests (``test_eval_generate`` etc.): skipped unless the
read-only ProteinMPNN repo + its bundled ``v_48_020`` weights are present and torch/numpy/biotite
import — so the suite stays green where ProteinMPNN isn't installed. ProteinMPNN runs on CPU for a
tiny backbone, so no GPU is required. FAST by construction: one 12-residue backbone, N=2, a single
ProteinMPNN invocation shared across all assertions (module-scoped fixture).
"""

import os
from pathlib import Path

import pytest

REPO = Path(os.environ.get("SPA_PROTEINMPNN_REPO", "/home/user1/projects/spa/needed_repos/ProteinMPNN"))
RUN = REPO / "protein_mpnn_run.py"
WEIGHTS = REPO / "vanilla_model_weights" / "v_48_020.pt"
L = 12          # backbone length (residues) — tiny for speed; ProteinMPNN clamps k-NN to length
N = 2           # sequences to design (the smoke N; the real run uses ~8)
CANONICAL = set("ACDEFGHIKLMNPQRSTVWYX")


def _have_deps() -> bool:
    if not (RUN.exists() and WEIGHTS.exists()):
        return False
    try:
        import biotite  # noqa: F401
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _have_deps(),
    reason="ProteinMPNN repo/weights or torch/numpy/biotite not available",
)


def _write_polyg_backbone(path: Path, length: int) -> None:
    """Write a trivial L-residue poly-glycine backbone (N, CA, C, O) PDB via biotite (the same writer
    Stage-1 ``generate.write_pdb`` uses), so ProteinMPNN's ``parse_PDB`` reads one chain of ``length``
    GLY residues. Geometry is arbitrary (non-colinear) — only counts/lengths matter for the smoke."""
    import numpy as np
    from biotite.structure import Atom, array
    from biotite.structure.io.pdb import PDBFile

    atoms = []
    serial_offsets = (("N", "N", 0.0), ("CA", "C", 1.0), ("C", "C", 2.0), ("O", "O", 2.6))
    for i in range(length):
        x = 3.8 * i
        y = 1.5 * np.sin(i)        # keep residues non-colinear so k-NN distances are well-defined
        for name, element, dz in serial_offsets:
            atoms.append(
                Atom(
                    [x, y, dz],
                    chain_id="A", res_id=i + 1, res_name="GLY",
                    atom_name=name, element=element,
                )
            )
    pdb = PDBFile()
    pdb.set_structure(array(atoms))
    path.parent.mkdir(parents=True, exist_ok=True)
    pdb.write(str(path))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """Build one poly-G backbone and inverse-fold it once (config-driven path → subprocess worker)."""
    from omegaconf import OmegaConf

    from spa.eval.proteinmpnn import inverse_fold

    tmp = tmp_path_factory.mktemp("spa_pmpnn")
    backbone = tmp / "designs" / "smoke_design_0.pdb"
    _write_polyg_backbone(backbone, L)

    cfg = OmegaConf.create(
        {
            "paths": {"proteinmpnn_repo": str(REPO)},
            "eval": {
                "out_dir": str(tmp / "designs"),
                "proteinmpnn": {
                    "num_seqs": N,
                    "sampling_temp": 0.1,
                    "seed": 0,
                    "batch_size": 1,
                    "model_name": "v_48_020",
                    "weights_dir": None,        # -> bundled vanilla_model_weights
                    "ca_only": False,
                    "conda_env": None,          # current interpreter (spa-dev)
                    "designs": [str(backbone)],  # explicit single design
                    "design_dir": None,
                    "out_dir": str(tmp / "seqs"),
                },
            },
        }
    )
    results = inverse_fold(cfg)
    assert len(results) == 1
    return results[0]


def test_returns_n_sequences(result):
    assert len(result.sequences) == N
    assert len(result.scores) == N


def test_each_sequence_has_backbone_length(result):
    for seq in result.sequences:
        assert len(seq.replace("/", "")) == L
    assert result.n_residues == L


def test_sequences_are_canonical_amino_acids(result):
    for seq in result.sequences:
        assert seq and set(seq.replace("/", "")) <= CANONICAL


def test_valid_fasta_written(result):
    # The FASTA exists, sits at <out_dir>/seqs/<design_stem>.fa, and holds native + N designed records.
    from spa.eval.proteinmpnn import _parse_fasta, parse_proteinmpnn_fasta

    assert result.fasta_path.exists()
    assert result.fasta_path.name == f"{result.name}.fa"
    text = result.fasta_path.read_text()
    assert text.startswith(">")

    all_records = _parse_fasta(result.fasta_path)
    assert len(all_records) == N + 1          # 1 native (input) record + N designed
    designed, scores = parse_proteinmpnn_fasta(result.fasta_path)
    assert len(designed) == N and len(scores) == N
    assert designed == result.sequences       # round-trips to what the SequenceSet returned
