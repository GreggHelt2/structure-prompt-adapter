"""SPA validation flywheel — Stage 1 (design generation).

Spec: dev ``05_validation_pipeline.md`` §1–§2 (the self-consistency flywheel, "Stage 0 — Generate
(RFD3 ± SPA)"). This package drives a frozen RFD3 inference engine with a trained SPA adapter to
generate backbone designs under a chosen condition (``baseline`` == wrapped-no-prompt == vanilla
RFD3; ``spa`` == prompted + λ-scaled), writing each design to disk as a PDB ready for the
downstream ProteinMPNN → OpenFold3 stages (the dev ``05`` F1.5.2 CIF→PDB handoff, done in-memory
from the RFD3 AtomArray).
"""

from .generate import Design, generate
from .proteinmpnn import SequenceSet, inverse_fold, run_proteinmpnn
from .score import (
    Adherence,
    ConditionSummary,
    DeltaSummary,
    DesignScore,
    Refolder,
    ScoreConfig,
    SelfConsistency,
    adherence,
    aggregate,
    ca_rmsd,
    delta_vs_baseline,
    is_designable,
    pairwise_tm_diversity,
    score_design,
    self_consistency,
    tm_score,
)

__all__ = [
    "Design",
    "generate",
    "SequenceSet",
    "inverse_fold",
    "run_proteinmpnn",
    # scoring (Stage 4)
    "Adherence",
    "ConditionSummary",
    "DeltaSummary",
    "DesignScore",
    "Refolder",
    "ScoreConfig",
    "SelfConsistency",
    "adherence",
    "aggregate",
    "ca_rmsd",
    "delta_vs_baseline",
    "is_designable",
    "pairwise_tm_diversity",
    "score_design",
    "self_consistency",
    "tm_score",
]
