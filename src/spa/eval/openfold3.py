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
        batch_patch_shim: str | None = None,
        ligand_ccd: str | None = None,
        ligand_smiles: str | None = None,
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
        # When set, run OF3 via `python <shim> predict …` instead of the bare `run_openfold` console
        # script — the shim monkeypatches OF3's 3 bs=1 guards so a runner-yaml with
        # data_module_args.batch_size>1 works for same-length batches (dev 23; scripts/eval/of3_batch_patch.py).
        self.batch_patch_shim = str(batch_patch_shim) if batch_patch_shim else None
        # Ligand for the refold query (dev ``90`` §3.1 L2). Both None => no ligand chain is emitted and
        # every query is byte-identical to before this existed. `ligand_smiles` wins if both are given.
        self.ligand_ccd = str(ligand_ccd) if ligand_ccd else None
        self.ligand_smiles = str(ligand_smiles) if ligand_smiles else None

    # ----------------------------------------------------------------------------------------------
    # Query JSON + command assembly + output-path reconstruction
    # ----------------------------------------------------------------------------------------------

    #: Chain ids handed to OF3, in ProteinMPNN's own chain order.
    _CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def _chain(self, seq) -> dict:
        """One query body: **one OF3 chain per ProteinMPNN chain**, plus any configured ligand.

        ⛔ **This used to do ``str(seq).replace("/", "")``, which was silently WRONG for a complex.**
        ProteinMPNN separates chains with ``/``; deleting it fused them into a single covalent chain, and
        because the total residue count was preserved, ``score.self_consistency``'s equal-length guard
        **passed** and scRMSD was computed against a chimera with no error, no warning and no NaN. Every
        complex would have come back systematically non-designable for a reason invisible in the output
        (dev ``90`` §2.1 item M2). A monomer has no ``/``, so its query is byte-identical to before.

        ⭐ **Ligand.** When :attr:`ligand_ccd` is set, a ``molecule_type: "ligand"`` chain is appended
        using OF3's own schema (``ccd_codes``, or ``smiles`` via :attr:`ligand_smiles`), so the refold
        oracle sees the same small molecule RFdiffusion3 designed around (dev ``90`` §3.1 item L2).
        Without it the ligand is absent from the refold and self-consistency silently scores an
        apo prediction against a holo design.
        """
        parts = [p.strip() for p in str(seq).split("/") if p.strip()]
        if len(parts) > len(self._CHAIN_IDS):
            raise ValueError(f"refold: {len(parts)} chains exceeds the {len(self._CHAIN_IDS)} ids available")
        chains = [{"molecule_type": "protein", "chain_ids": [self._CHAIN_IDS[i]], "sequence": p}
                  for i, p in enumerate(parts)]
        if self.ligand_ccd or self.ligand_smiles:
            lig = {"molecule_type": "ligand", "chain_ids": [self._CHAIN_IDS[len(parts)]]}
            if self.ligand_smiles:
                lig["smiles"] = str(self.ligand_smiles)
            else:
                lig["ccd_codes"] = str(self.ligand_ccd)
            chains.append(lig)
        return {"chains": chains}

    def _build_query_json(self, sequences: list[str]) -> dict:
        """One single-chain protein query per sequence, keyed ``q{i}`` (dev ``05`` schema)."""
        return {"queries": {f"q{i}": self._chain(s) for i, s in enumerate(sequences)}}

    def _build_command(self, query_json: Path, run_dir: Path) -> list[str]:
        head = ["python", self.batch_patch_shim] if self.batch_patch_shim else ["run_openfold"]
        cmd = head + [
            "predict",
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
        self._last_proc = proc  # kept so callers can surface OF3 output on a silent (exit-0) empty result
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenFold3 refold failed (exit {proc.returncode}) for {run_dir.name}.\n"
                f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )

    def _surface_missing(self, run_dir: Path, n_missing: int) -> None:
        """Diagnose a SILENT shortfall (exit 0 but cifs missing). OF3's ``predict_step`` wraps
        forward+confidence in a try/except that logs to ``<run>/logs/predict_err_rank*.log`` and returns
        ``None`` — so a real failure (esp. a bs>1 ragged-atom crash) looks like "no folds" with no error.
        Surface the swallowed traceback(s) + the tail of OF3's own stderr so it stops being invisible."""
        print(f"[of3] ⚠️  {n_missing} refold(s) MISSING despite exit 0 — OF3 likely SWALLOWED an exception.")
        logs = sorted(run_dir.glob("**/predict_err_rank*.log"))
        for lg in logs:
            print(f"[of3]    --- swallowed traceback: {lg} ---")
            try:
                print(lg.read_text()[-2500:])
            except Exception as exc:  # pragma: no cover
                print(f"[of3]    (could not read {lg}: {exc})")
        proc = getattr(self, "_last_proc", None)
        if proc is not None and (proc.stderr or "").strip():
            print(f"[of3]    --- OF3 stderr tail ---\n{proc.stderr[-2000:]}")
        if not logs and not (proc and (proc.stderr or "").strip()):
            print(f"[of3]    (no predict_err log under {run_dir} and no stderr — check batch_size/kernel yaml)")

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
        self._run_openfold(self._build_query_json(sequences), run_dir)
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
            self._surface_missing(run_dir, missing)
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
        expected = sum(nseq)
        if total < expected:
            self._surface_missing(run_dir, expected - total)
        print(f"[of3] refold_all: {total} refold(s) for {len(sets)} backbone(s) in ONE run -> {run_dir}")
        return out


# ------------------------------------------------------------------------------------------------
# Confidence outputs (dev ``90`` §5.0l step 1)
# ------------------------------------------------------------------------------------------------
#
# ⛔ **OF3 has always written confidence and this module has always thrown it away.** Beside every
# ``*_model.cif`` OF3 writes two more files, and until 2026-09-04 nothing in this project opened
# either:
#
#   ``*_confidences_aggregated.json``  scalars + per-chain dicts: ``iptm``, ``ptm``, ``chain_ptm``,
#                                     ``chain_pair_iptm``, ``bespoke_iptm``, ``avg_plddt``,
#                                     ``has_clash``, ``sample_ranking_score``, ``disorder``, ``gpde``
#   ``*_confidences.json``            the matrices: ``pae`` and ``pde`` (token x token) and per-atom
#                                     ``plddt``
#
# ⭐ **Why it matters.** ``score.is_designable`` accepts a pLDDT gate that has never fired anywhere in
# this project, because no caller could supply a pLDDT. More importantly, RFdiffusion3's own
# multimeric-binder criterion is *confidence-gated* (interface minimum pAE, binder pTM, target-aligned
# RMSD; dev ``90`` §3a), so a criterion shaped like theirs was uncomputable from our artifacts even
# though the inputs were on disk the whole time. Same defect class as the silent chain fusion
# (dev ``90`` §2.1 M2): the capability was present and nothing read it.
#
# ⚠️ **This is a READER, not a criterion.** Nothing here changes ``is_designable``, ``self_consistency``
# or any number already reported. It only makes the confidence available to callers that ask.
#
# ⚠️ **OF3 is not AlphaFold3.** Reading ipTM and pAE from OF3 does not make our numbers comparable to
# RFdiffusion3's published ones, which are AlphaFold3 quantities (dev ``90`` §3a). It makes an
# *internally* consistent confidence-gated criterion possible, applied to both arms of our own runs.


def confidence_paths(cif_path) -> tuple[Path, Path]:
    """``(aggregated_json, full_json)`` beside a refold cif. Neither is checked for existence."""
    cif = Path(str(cif_path))
    stem = cif.name
    for suffix in (".cif.gz", ".cif", ".pdb"):
        if stem.endswith("_model" + suffix):
            stem = stem[: -len("_model" + suffix)]
            break
    else:
        stem = cif.stem.replace("_model", "")
    return (cif.parent / f"{stem}_confidences_aggregated.json",
            cif.parent / f"{stem}_confidences.json")


def read_confidence(cif_path, *, with_matrices: bool = False) -> dict | None:
    """Confidence scalars for one refold, or ``None`` when OF3 wrote none.

    Returns the aggregated JSON as-is (``iptm``, ``ptm``, ``chain_ptm``, ``chain_pair_iptm``,
    ``avg_plddt``, ``has_clash``, ...). With ``with_matrices=True`` the token-level ``pae`` / ``pde``
    and per-atom ``plddt`` are merged in under their own keys, which is a much larger read (a 215-token
    complex carries a 215x215 ``pae``), so it is off by default.
    """
    agg_p, full_p = confidence_paths(cif_path)
    if not agg_p.exists():
        return None
    with open(agg_p) as fh:
        out = dict(json.load(fh))
    if with_matrices and full_p.exists():
        with open(full_p) as fh:
            out.update(json.load(fh))
    return out


def interface_pae(pae, chain_sizes) -> dict:
    """Cross-chain pAE summaries for a two-chain complex, from the token-level ``pae`` matrix.

    ``chain_sizes`` is the token count per chain **in the order OF3 emitted them**, which for our
    complex refolds is ``[len(designed), len(target)]`` because :meth:`OF3Refolder._chain` assigns
    chain ids A, B, ... in ProteinMPNN's own chain order. For an all-protein complex a token is a
    residue, so Ca counts per chain are the right sizes; a ligand contributes per-atom tokens and must
    be counted as such.

    Returns ``min``, ``mean`` and ``median`` over the union of the two off-diagonal blocks, plus
    ``min_ab`` / ``min_ba`` for the individual directions (pAE is not symmetric: entry ``(i, j)`` is
    the expected error at token ``i`` when the prediction is aligned on token ``j``).

    ⭐ **Which one is RFdiffusion3's.** Their gate is the interface **minimum** pAE at 1.5 A
    (dev ``90`` §3a), which is a permissive statistic over thousands of pairs. The binder-design
    literature more often gates a cross-chain **mean**, so both are returned and any report must name
    which it used (`WORKING_AGREEMENTS` §2.7).
    """
    if not pae or len(chain_sizes) < 2:
        return {}
    n_a = int(chain_sizes[0])
    n_ab = n_a + int(chain_sizes[1])
    ab = [v for row in pae[:n_a] for v in row[n_a:n_ab]]
    ba = [v for row in pae[n_a:n_ab] for v in row[:n_a]]
    both = ab + ba
    if not both:
        return {}
    s = sorted(both)
    return {
        "min": float(min(both)),
        "mean": float(sum(both) / len(both)),
        "median": float(s[len(s) // 2]),
        "min_ab": float(min(ab)) if ab else float("nan"),
        "min_ba": float(min(ba)) if ba else float("nan"),
        "n_pairs": len(both),
    }
