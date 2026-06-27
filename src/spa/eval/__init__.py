"""SPA validation flywheel — Stage 1 (design generation).

Spec: dev ``05_validation_pipeline.md`` §1–§2 (the self-consistency flywheel, "Stage 0 — Generate
(RFD3 ± SPA)"). This package drives a frozen RFD3 inference engine with a trained SPA adapter to
generate backbone designs under a chosen condition (``baseline`` == wrapped-no-prompt == vanilla
RFD3; ``spa`` == prompted + λ-scaled), writing each design to disk as a PDB ready for the
downstream ProteinMPNN → OpenFold3 stages (the dev ``05`` F1.5.2 CIF→PDB handoff, done in-memory
from the RFD3 AtomArray).
"""

from .generate import Design, generate

__all__ = ["Design", "generate"]
