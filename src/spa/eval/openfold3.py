"""Stage 3 of the SPA validation flywheel: refold ProteinMPNN sequences with OpenFold3.

Spec: dev ``05_validation_pipeline.md`` §1 ("Stage 3 — Refold (OpenFold3)") + §4 (file-based handoff
across env boundaries) and dev ``07`` F1.5.4 (the no-kernel runner-yaml). This is the concrete
implementation of the :class:`spa.eval.score.Refolder` protocol that the flywheel's Stage 3 injection
point expects — turning a Stage-2 :class:`~spa.eval.proteinmpnn.SequenceSet` (the N designed sequences
for one backbone) into N OpenFold3 **refold** structures for the best-of-K self-consistency scRMSD
(designability) metric.

How OF3 is driven (verified in Task 1.5 / dev ``05`` Stage 3):

- OF3 ships an ``run_openfold`` console entry point (an entry point in the ``spa-verify-of3`` env, NOT
  importable here — it has its own heavy deps). We invoke it via **subprocess**, exactly the
  invocation dev ``05`` verified: ``run_openfold predict --query-json q.json --use-msa-server=False
  --inference-ckpt-path of3.pt --runner-yaml of3_nokernel.yml --num-diffusion-samples 1 --output-dir
  out`` (run in ``spa-verify-of3`` via ``conda run``).
- **MSA-free** (designed sequences have no meaningful MSA) and the **no-kernel runner-yaml**
  (F1.5.4: disables the DeepSpeed evo-attention / triton / cueq kernels that aren't installed → stock
  PyTorch attention). Both the CLI flag and the yaml set ``use_msa_server=false`` (belt-and-suspenders).
- **One subprocess per backbone, all N sequences batched into one multi-query JSON** so the ~2.3 GB
  OF3 model loads ONCE per backbone, then folds the N sequences sequentially (peak VRAM = a single
  fold = length-driven, independent of N — dev ``05`` measured 2.2 GB at 76 res, MSA-free).
- **GPU targeting is inherited, never hardcoded** (dev root ``CLAUDE.md`` portability rule): the
  subprocess inherits the parent's ``CUDA_VISIBLE_DEVICES`` (the A5000 UUID locally; unset on the
  single-GPU H100). ``cuda_visible_devices`` can override per-call if ever needed.
- **File-based handoff** (dev ``05`` §4): sequences in → ``.cif`` out; nothing in-memory crosses the
  env boundary, so OF3 can live in its own env/machine. Refold ``.cif`` files are returned as
  **paths**; :func:`spa.eval.score.self_consistency` loads them (biotite auto-detects mmCIF) for scRMSD.

OF3 writes ``{out}/of3/{design}/q{i}/seed_{seed}/q{i}_seed_{seed}_sample_1_model.cif`` per sequence
(``writer.py``; one ``seed`` from the runner-yaml ``seeds: [42]``, one sample from
``--num-diffusion-samples 1``). We use simple ``q{i}`` query ids (the design name is the run dir) so
the output path is reconstructed unambiguously regardless of any query-id sanitization.

All knobs (ckpt, runner-yaml, conda env, #samples, seed, structure format, out dir) are config/CLI —
nothing hardware- or cost-specific is hardcoded.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


class OF3Refolder:
    """Refold ProteinMPNN sequences with OpenFold3 (the Stage-3 :class:`spa.eval.score.Refolder`).

    Args:
        ckpt_path: OF3 inference checkpoint (``paths.openfold3_ckpt``).
        runner_yaml: the no-kernel runner-yaml (``paths.openfold3_runner_yaml``; F1.5.4).
        out_dir: root for refold outputs (refolds land under ``<out_dir>/of3/<design>/``).
        conda_env: env with the ``run_openfold`` entry point (default ``spa-verify-of3``); ``None`` ->
            current interpreter.
        num_diffusion_samples: OF3 diffusion samples per sequence (1 = one refold/sequence, the
            best-of-K self-consistency unit; OF3's own default is 5).
        seed: the single model seed (must match the runner-yaml ``seeds: [seed]`` — drives the
            ``seed_{seed}`` output dir).
        structure_format: OF3 output structure format (``cif`` default; ``cif.gz`` / ``pdb``).
        cuda_visible_devices: optional explicit device mask; ``None`` inherits the parent env
            (portable — A5000 UUID locally, unset on the single-GPU H100).
        use_msa_server: keep ``False`` (MSA-free); passed on the CLI to match dev ``05``.
    """

    def __init__(
        self,
        *,
        ckpt_path,
        runner_yaml,
        out_dir,
        conda_env: str | None = "spa-verify-of3",
        num_diffusion_samples: int = 1,
        seed: int = 42,
        structure_format: str = "cif",
        cuda_visible_devices: str | None = None,
        use_msa_server: bool = False,
    ) -> None:
        self.ckpt_path = str(ckpt_path)
        self.runner_yaml = str(runner_yaml)
        self.out_dir = Path(str(out_dir))
        self.conda_env = conda_env
        self.num_diffusion_samples = int(num_diffusion_samples)
        self.seed = int(seed)
        self.structure_format = str(structure_format)
        self.cuda_visible_devices = cuda_visible_devices
        self.use_msa_server = bool(use_msa_server)

    # ----------------------------------------------------------------------------------------------
    # Query JSON + command assembly + output-path reconstruction
    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def _chain(seq) -> dict:
        """One single-chain protein query body (cleaned sequence; drop ProteinMPNN '/' chain seps)."""
        clean = str(seq).replace("/", "").strip()
        return {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": clean}]}

    def _build_query_json(self, sequences: list[str]) -> dict:
        """One single-chain protein query per sequence, keyed ``q{i}`` (dev ``05`` schema)."""
        return {"queries": {f"q{i}": self._chain(s) for i, s in enumerate(sequences)}}

    def _build_command(self, query_json: Path, run_dir: Path) -> list[str]:
        cmd = [
            "run_openfold", "predict",
            "--query-json", str(query_json),
            "--output-dir", str(run_dir),
            "--inference-ckpt-path", self.ckpt_path,
            "--runner-yaml", self.runner_yaml,
            "--num-diffusion-samples", str(self.num_diffusion_samples),
            f"--use-msa-server={self.use_msa_server}",  # dev 05 verified `=False` form
        ]
        if self.conda_env:
            cmd = ["conda", "run", "-n", str(self.conda_env)] + cmd
        return cmd

    def _cif_path(self, run_dir: Path, qid: str) -> Path:
        """The cif OF3 writes for query ``qid`` (writer.py: ``{id}/seed_{S}/{id}_seed_{S}_sample_1_*``)."""
        return run_dir / qid / f"seed_{self.seed}" / f"{qid}_seed_{self.seed}_sample_1_model.{self.structure_format}"

    def _run_openfold(self, query_payload: dict, run_dir: Path) -> None:
        """Write the multi-query JSON, run ONE ``run_openfold`` subprocess (model loads once), raise on
        failure. Shared by :meth:`refold` (one backbone) and :meth:`refold_all` (the whole matrix)."""
        run_dir.mkdir(parents=True, exist_ok=True)
        query_json = run_dir / "queries.json"
        with open(query_json, "w") as fh:
            json.dump(query_payload, fh)
        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        cmd = self._build_command(query_json, run_dir)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenFold3 refold failed (exit {proc.returncode}) for {run_dir.name}.\n"
                f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )

    # ----------------------------------------------------------------------------------------------
    # Refolder protocol
    # ----------------------------------------------------------------------------------------------

    def refold(self, sequence_set) -> list:
        """Refold ONE backbone's sequences (one ``run_openfold`` subprocess; model loads once). Returns
        the cif paths that got written (a missing one is dropped + warned, so best-of-K just has fewer
        candidates than poisoning the scRMSD with a bad path)."""
        name = getattr(sequence_set, "name", "design")
        sequences = list(getattr(sequence_set, "sequences", []) or [])
        if not sequences:
            print(f"[of3] {name}: no sequences to refold -> skipping.")
            return []
        run_dir = self.out_dir / "of3" / name
        self._run_openfold({"queries": {f"q{i}": self._chain(s) for i, s in enumerate(sequences)}}, run_dir)
        refolds: list[str] = []
        missing = 0
        for i in range(len(sequences)):
            cif = self._cif_path(run_dir, f"q{i}")
            if cif.exists():
                refolds.append(str(cif))
            else:
                missing += 1
        if missing:
            print(f"[of3] {name}: {missing}/{len(sequences)} refold(s) missing on disk (dropped).")
        print(f"[of3] {name} -> {len(refolds)} refold(s) in {run_dir}")
        return refolds

    def refold_all(self, sequence_sets) -> dict:
        """**Batched** refold across MANY backbones in ONE ``run_openfold`` subprocess — the model loads
        **once for the whole matrix**, amortizing the ~45 s/process startup (import+CUDA+build) that
        dominates per-backbone refolding (≈ halves eval wall-time at scale). Returns
        ``{design_name: [refold cif paths]}``. The flywheel uses this when present (else per-design
        :meth:`refold`). **Peak VRAM is unchanged** — OF3 still folds queries sequentially (peak = a
        single fold, length-driven). Query ids ``d{i}_q{j}`` map back to (design i, seq j) for grouping.
        """
        sets = list(sequence_sets)
        queries: dict = {}
        names: list[str] = []
        nseq: list[int] = []
        for i, ss in enumerate(sets):
            names.append(getattr(ss, "name", f"design{i}"))
            seqs = list(getattr(ss, "sequences", []) or [])
            nseq.append(len(seqs))
            for j, s in enumerate(seqs):
                queries[f"d{i}_q{j}"] = self._chain(s)
        out: dict = {nm: [] for nm in names}
        if not queries:
            print("[of3] refold_all: no sequences across any backbone -> skipping.")
            return out
        run_dir = self.out_dir / "of3_batch"
        self._run_openfold({"queries": queries}, run_dir)
        total = 0
        for i, nm in enumerate(names):
            for j in range(nseq[i]):
                cif = self._cif_path(run_dir, f"d{i}_q{j}")
                if cif.exists():
                    out[nm].append(str(cif))
                    total += 1
        print(f"[of3] refold_all: {total} refold(s) for {len(sets)} backbone(s) in ONE run -> {run_dir}")
        return out
