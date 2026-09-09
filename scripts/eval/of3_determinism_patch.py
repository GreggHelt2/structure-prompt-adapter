"""Make OpenFold3 inference bitwise reproducible: a runtime monkeypatch, NO edit to the read-only OF3
dep (dev ``docs/plan/91`` §5; audited 2026-09-08).

THE PROBLEM. OF3 is not bit-identical run to run even at ``batch_size=1`` with a fixed seed. Six
default-mode runs of one 76 aa MSA-free monomer produced **six distinct output files**: all 601 ATOM
records differ, and every other line is byte-identical (the OF3 cif carries no timestamp or version
line, so nothing needs excluding from the diff).

THE CAUSE IS ONE OP. ``Tensor.scatter_add_`` on CUDA, in the atom-to-token pooling at
``openfold3/core/utils/atomize_utils.py:183`` (the float feature aggregation) and ``:195`` (the
per-token atom count). It accumulates with CUDA atomics, so the summation order varies run to run: on
the A5000 it gives **30 distinct bit patterns in 30 identical calls**. Measured **402 calls per fold**,
independent of length. Only ``:183`` actually matters, since ``:195`` sums 0/1 masks, which is exact
in float regardless of order; patching the method covers both harmlessly.

⭐ This is the SAME STRUCTURAL BUG as RFdiffusion3's, differing only in which torch op does the
accumulation (RFD3 uses ``index_reduce``; see ``spa.eval.determinism``). Both are float atomics
pooling atoms into tokens.

THE MAGNITUDE IS TINY AND CHANGES NO PUBLISHED RESULT. Cα-RMSD **0.0017 to 0.0028 Å** superposed
between runs, max per-atom displacement 0.036 Å, ``avg_plddt`` moving 78.71051 to 78.708778. That is
about 1000x below the 2.0 Å scRMSD designability threshold, so it cannot flip a designability call.
It breaks bit-exact provenance, not conclusions.

WHY NOT THE TORCH FLAGS, which do work here (unlike RFD3). Measured ablation, per fold:

===========================================================  ===============  =============
route                                                        bit-identical?   per-fold time
===========================================================  ===============  =============
default, as shipped                                          no, 4 distinct   16.9 to 19.3 s
``cudnn.deterministic`` + ``CUBLAS_WORKSPACE_CONFIG`` only    no, 2 distinct   27.0 s
full torch determinism flags                                 yes, 6 runs      28.1 to 30.1 s
**this shim**                                                **yes**          see below
===========================================================  ===============  =============

The flags cost **~1.6x**, and essentially all of it is ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` alone
(27.0 s by itself). Since OF3 refolding is ~90.5% of local wall-clock (dev ``plan/85``), the flag
route taxes the pipeline's dominant stage. This shim instead scopes determinism to the one op that
needs it, so cuBLAS is never constrained and no environment variable is set.

HOW. ``torch.Tensor.scatter_add_`` is wrapped so that PyTorch's **own** deterministic route
(``_scatter_via_index_put``, a sort-based accumulation) is selected for that call only. We do not
reimplement the reduction: the arithmetic is torch's, which is why there is no correctness risk and
no assumption about index shape or batching. Op-level cost is 604.7 us against 26.6 us stock, which
over 402 calls is ~232 ms per fold, roughly +1.3% on a 17.7 s fold.

⛔ WHAT THIS DOES NOT BUY.

- **It does not reproduce past runs.** Every route produces different structures from the default
  build, so archived refolds stay as recorded. Go-forward only.
- ⚠️ **It is not known to be sufficient for LIGANDS.** OF3 PR #320 (open, unreviewed since 2026-07-18)
  independently diagnoses this same ``scatter_add_`` and names a **second** source we did not
  exercise: ambient Python/NumPy/Torch RNG during on-the-fly feature creation (ligand conformer
  generation, ``ref_pos``), not pinned to the datapoint seed. Our audit used an MSA-free monomer with
  no ligand, where that path likely never fires. **Test a ligand case before claiming completeness**
  (dev ``plan/91`` §5.3).
- **Batch size stays part of the identity.** ``bs>1`` is not bit-identical to ``bs=1`` for non-first
  rows regardless of this shim, because ``predict_step`` reseeds once per batch
  (``runner.py:923-926``, "TODO bs=1"). See dev ``plan/23`` §7.5 and ``of3_batch_patch.py``.
- **Untested:** longer chains, complexes, MSA-supplied refolds, the H100, and the triton (cloud)
  config. ``ChunkSizeTuner`` (``core/utils/chunk_utils.py:351-388``) does an OOM-driven binary search,
  so its chunk choice is memory-pressure dependent; believed benign, not isolated.

Run OF3 through this shim instead of the bare console script::

    conda run -n spa-verify-of3 python of3_determinism_patch.py predict --query-json … …

``OF3Refolder(deterministic=True)`` does this automatically.
"""

from __future__ import annotations

import sys


def apply_patches() -> None:
    import torch

    _orig_scatter_add_ = torch.Tensor.scatter_add_

    def _deterministic_scatter_add_(self, dim, index, src):
        """Select torch's own deterministic scatter route, for this call only.

        Scoping matters. Leaving ``use_deterministic_algorithms(True)`` on globally is what forces
        ``CUBLAS_WORKSPACE_CONFIG`` (the first cuBLAS matmul otherwise raises), and that env var is
        where ~all of the 1.6x flag-route penalty lives. Nothing inside ``scatter_add_`` calls cuBLAS,
        so enabling it here is free of that constraint. The previous state is always restored, including
        its ``warn_only`` setting, so a caller that had its own determinism configuration keeps it.
        """
        prev = torch.are_deterministic_algorithms_enabled()
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return _orig_scatter_add_(self, dim, index, src)
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn)

    torch.Tensor.scatter_add_ = _deterministic_scatter_add_

    print(
        "[of3_determinism_patch] scatter_add_ routed through torch's deterministic kernel. "
        "Refolds will NOT match default-build runs at the same seed.",
        file=sys.stderr,
        flush=True,
    )


def main():
    apply_patches()
    from openfold3.run_openfold import cli

    cli()


if __name__ == "__main__":
    main()
