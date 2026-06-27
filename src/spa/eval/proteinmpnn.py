"""Stage 2 of the SPA validation flywheel: inverse-fold a design backbone with ProteinMPNN.

Spec: dev ``05_validation_pipeline.md`` §1–§2 ("Stage 2 — Inverse fold (ProteinMPNN)") and §4
(file-based handoff across env boundaries). Consumes a Stage-1 :class:`spa.eval.generate.Design`
(its ``.path`` is a written PDB — the dev ``05`` F1.5.2 CIF→PDB handoff already done in-memory by
``generate.write_pdb``) and runs ProteinMPNN to produce **N sequences** for that fixed backbone.

How ProteinMPNN is driven (verified in Task 1.5; dev ``07`` smoke results + ``01`` §5):

- ProteinMPNN ships as a **script** (``needed_repos/ProteinMPNN/protein_mpnn_run.py``), not a clean
  importable package — its ``from protein_mpnn_utils import ...`` only resolves when the script's
  own directory is ``sys.path[0]``. We therefore invoke it via **subprocess** (running the script
  adds its dir to ``sys.path`` automatically), exactly the invocation Task 1.5 verified:
  ``python protein_mpnn_run.py --pdb_path bb.pdb --out_folder out --num_seq_per_target N
  --sampling_temp T --seed S`` with bundled default weights ``vanilla_model_weights/v_48_020.pt``.
- **File-based PDB → FASTA handoff** (dev ``05`` §4): PDB in, FASTA out — no in-memory tensors cross
  the boundary, so ProteinMPNN can run in its own env/machine. By default it runs under the **current
  interpreter** (``spa-dev``, where Task 1.5 verified it); set ``proteinmpnn.conda_env`` to wrap the
  call in ``conda run -n <env>`` if it must live elsewhere.
- ProteinMPNN writes one FASTA per backbone at ``<out_folder>/seqs/<pdb_stem>.fa``. Its **first**
  record is the input (native) sequence (header ``>name, score=...``); the following N records are
  the designed sequences (headers ``>T=..., sample=k, score=..., global_score=..., seq_recovery=...``,
  ``protein_mpnn_run.py:384,403``). We parse the designed records back out and return them.

Cost knobs (``proteinmpnn.num_seqs`` N, ``sampling_temp``, ``model_name``/``weights_dir``,
``out_dir``, ``seed``, ``batch_size``, ``ca_only``, ``conda_env``) are all config/CLI — nothing
hardware- or cost-specific is hardcoded (dev root ``CLAUDE.md`` portability rule). N≈8 is the
conventional best-of-8 self-consistency set, but it is the primary cost lever and stays a knob.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SequenceSet:
    """The N inverse-folded sequences ProteinMPNN designed for one backbone (dev ``05`` Stage 2).

    Attributes:
        name: the design stem (== Stage-1 PDB stem == FASTA basename; the output-name group id).
        design_path: the input backbone PDB (a :attr:`spa.eval.generate.Design.path`).
        fasta_path: the written FASTA (ProteinMPNN ``<out_dir>/seqs/<name>.fa``; native + N designs).
        sequences: the N designed sequences (native record excluded). Multi-chain seqs keep
            ProteinMPNN's ``/`` chain separator; a monomer is a plain string.
        n_residues: residue count of the backbone (== ``len(seq)`` with any ``/`` removed).
        scores: per-sequence ProteinMPNN ``score`` (neg-log-prob; lower = more confident) parsed from
            the FASTA headers, aligned with :attr:`sequences` — for downstream best-of-K (dev ``05`` §3).
    """

    name: str
    design_path: Path
    fasta_path: Path
    sequences: list[str]
    n_residues: int
    scores: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# Config resolution (mirror generate.py: read the `eval` group, default sensibly)
# --------------------------------------------------------------------------------------------------


def _pmpnn_cfg(cfg):
    """The ``eval.proteinmpnn`` sub-config (a plain dict view) with the Stage-2 knobs."""
    return cfg.eval.get("proteinmpnn") or {}


def _weights_dir(cfg, pm) -> Path:
    """Resolve the ProteinMPNN weights folder: ``proteinmpnn.weights_dir`` or the bundled default
    ``<proteinmpnn_repo>/vanilla_model_weights`` (dev ``05`` Stage 2 / ``07`` Task 1.5)."""
    if pm.get("weights_dir"):
        return Path(str(pm["weights_dir"]))
    return _repo_dir(cfg) / "vanilla_model_weights"


def _repo_dir(cfg) -> Path:
    """The read-only ProteinMPNN repo (``paths.proteinmpnn_repo``); overridable via env for tests."""
    env = os.environ.get("SPA_PROTEINMPNN_REPO")
    if env:
        return Path(env)
    return Path(str(cfg.paths.proteinmpnn_repo))


# --------------------------------------------------------------------------------------------------
# FASTA parsing (ProteinMPNN's output format; protein_mpnn_run.py:317,384,403)
# --------------------------------------------------------------------------------------------------


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a FASTA into ``[(header_without_'>', sequence), ...]`` (whitespace-robust)."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header, parts = line[1:].strip(), []
        elif line.strip():
            parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def _header_field(header: str, key: str):
    """Pull ``key=<value>`` out of a ProteinMPNN FASTA header (comma-separated ``k=v`` fields)."""
    for tok in header.split(","):
        tok = tok.strip()
        if tok.startswith(f"{key}="):
            return tok[len(key) + 1:].strip()
    return None


def parse_proteinmpnn_fasta(path: Path) -> tuple[list[str], list[float]]:
    """Split a ProteinMPNN FASTA into its (designed sequences, scores), dropping the native record.

    The native (input) record is the first one and its header has no ``sample=`` field; every designed
    record's header carries ``sample=k`` (``protein_mpnn_run.py:403``). We select on that so the result
    is exactly the N designed sequences regardless of ordering.
    """
    seqs: list[str] = []
    scores: list[float] = []
    for header, seq in _parse_fasta(path):
        if "sample=" not in header:  # native/input record -> skip
            continue
        seqs.append(seq)
        sc = _header_field(header, "score")
        scores.append(float(sc) if sc is not None else float("nan"))
    return seqs, scores


def _residue_count(seq: str) -> int:
    """Backbone residue count from a (possibly multi-chain) ProteinMPNN sequence."""
    return len(seq.replace("/", ""))


# --------------------------------------------------------------------------------------------------
# Invocation (subprocess on protein_mpnn_run.py — it is a script, not cleanly importable)
# --------------------------------------------------------------------------------------------------


def _build_command(
    *,
    repo_dir: Path,
    pdb_path: Path,
    out_dir: Path,
    num_seqs: int,
    sampling_temp,
    seed: int,
    batch_size: int,
    weights_dir: Path,
    model_name: str,
    ca_only: bool,
    conda_env: str | None,
) -> list[str]:
    """Assemble the ``protein_mpnn_run.py`` argv (the exact Task-1.5 invocation; dev ``05`` Stage 2)."""
    cmd = [
        sys.executable,
        str(repo_dir / "protein_mpnn_run.py"),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(int(num_seqs)),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(int(seed)),
        "--batch_size", str(int(batch_size)),
        "--path_to_model_weights", str(weights_dir),
        "--model_name", str(model_name),
    ]
    if ca_only:
        cmd.append("--ca_only")
    if conda_env:  # run ProteinMPNN in its own conda env (dev 05 §4 cross-env handoff)
        cmd = ["conda", "run", "-n", str(conda_env)] + cmd
    return cmd


def run_proteinmpnn(
    pdb_path,
    *,
    repo_dir,
    out_dir,
    num_seqs: int = 8,
    sampling_temp=0.1,
    seed: int = 0,
    batch_size: int = 1,
    weights_dir=None,
    model_name: str = "v_48_020",
    ca_only: bool = False,
    conda_env: str | None = None,
) -> SequenceSet:
    """Inverse-fold one backbone PDB → N designed sequences (the low-level Stage-2 worker).

    Runs ``protein_mpnn_run.py`` as a subprocess (dev ``05`` Stage 2), then parses the FASTA it writes
    at ``<out_dir>/seqs/<pdb_stem>.fa`` and returns the N designed sequences as a :class:`SequenceSet`.
    All knobs are explicit so a driver/test can call this without a full Hydra config.

    Raises:
        ValueError: if ``num_seqs`` is not a positive multiple of ``batch_size`` (ProteinMPNN computes
            ``NUM_BATCHES = num_seqs // batch_size`` and would silently produce fewer sequences).
        RuntimeError: if the ProteinMPNN subprocess fails or writes no FASTA.
    """
    pdb_path = Path(pdb_path)
    repo_dir = Path(repo_dir)
    out_dir = Path(out_dir)
    num_seqs, batch_size = int(num_seqs), int(batch_size)
    if num_seqs <= 0 or batch_size <= 0 or num_seqs % batch_size != 0:
        raise ValueError(
            f"num_seqs ({num_seqs}) must be a positive multiple of batch_size ({batch_size}); "
            "ProteinMPNN drops the remainder (NUM_BATCHES = num_seqs // batch_size)."
        )
    if weights_dir is None:
        weights_dir = repo_dir / "vanilla_model_weights"

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(
        repo_dir=repo_dir, pdb_path=pdb_path, out_dir=out_dir, num_seqs=num_seqs,
        sampling_temp=sampling_temp, seed=seed, batch_size=batch_size,
        weights_dir=Path(weights_dir), model_name=model_name, ca_only=ca_only, conda_env=conda_env,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(
            f"ProteinMPNN failed (exit {proc.returncode}) for {pdb_path}.\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    # ProteinMPNN names the FASTA after the PDB stem (parse_PDB: name = basename[:-4]).
    fasta_path = out_dir / "seqs" / f"{pdb_path.stem}.fa"
    if not fasta_path.exists():
        raise RuntimeError(
            f"ProteinMPNN reported success but no FASTA at {fasta_path}.\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    sequences, scores = parse_proteinmpnn_fasta(fasta_path)
    n_res = _residue_count(sequences[0]) if sequences else 0
    return SequenceSet(
        name=pdb_path.stem, design_path=pdb_path, fasta_path=fasta_path,
        sequences=sequences, n_residues=n_res, scores=scores,
    )


# --------------------------------------------------------------------------------------------------
# Config-driven entry (mirror generate.generate: compose-config in, list of records out)
# --------------------------------------------------------------------------------------------------


def _resolve_design_paths(cfg, pm) -> list[Path]:
    """Resolve the Stage-1 design PDBs to inverse-fold (explicit list / dir / Stage-1 ``out_dir``)."""
    from .generate import _resolve_out_dir

    if pm.get("designs"):
        return [Path(str(p)) for p in pm["designs"]]
    src = pm.get("design_dir") or cfg.eval.out_dir
    d = _resolve_out_dir(src)
    return sorted(d.glob("*.pdb"))


def inverse_fold(cfg, *, designs=None) -> list[SequenceSet]:
    """Inverse-fold a batch of Stage-1 backbones with ProteinMPNN from a composed config.

    Mirrors :func:`spa.eval.generate.generate`: reads the ``eval.proteinmpnn`` knobs, resolves the
    set of design PDBs (an explicit ``designs`` arg of :class:`~spa.eval.generate.Design` / paths, else
    ``proteinmpnn.designs`` / ``proteinmpnn.design_dir`` / the Stage-1 ``eval.out_dir`` glob), runs
    ProteinMPNN on each, and returns the per-design :class:`SequenceSet` records.

    Args:
        cfg: composed config (``eval.proteinmpnn`` knobs + ``paths.proteinmpnn_repo``).
        designs: an explicit iterable of :class:`~spa.eval.generate.Design` (``.path``) or PDB paths to
            inverse-fold (a driver injection point); falls back to the config-resolved set if None.
    """
    pm = _pmpnn_cfg(cfg)
    repo_dir = _repo_dir(cfg)
    weights_dir = _weights_dir(cfg, pm)
    from .generate import _resolve_out_dir

    out_dir = _resolve_out_dir(pm.get("out_dir") or "./outputs/eval/seqs")

    if designs is not None:
        paths = [Path(str(getattr(d, "path", d))) for d in designs]
    else:
        paths = _resolve_design_paths(cfg, pm)
    if not paths:
        raise ValueError(
            "inverse_fold: no design PDBs found — pass designs=, or set eval.proteinmpnn.designs / "
            "eval.proteinmpnn.design_dir, or generate Stage-1 designs into eval.out_dir first."
        )

    results: list[SequenceSet] = []
    for pdb_path in paths:
        res = run_proteinmpnn(
            pdb_path,
            repo_dir=repo_dir,
            out_dir=out_dir,
            num_seqs=int(pm.get("num_seqs", 8)),
            sampling_temp=pm.get("sampling_temp", 0.1),
            seed=int(pm.get("seed", 0)),
            batch_size=int(pm.get("batch_size", 1)),
            weights_dir=weights_dir,
            model_name=str(pm.get("model_name", "v_48_020")),
            ca_only=bool(pm.get("ca_only", False)),
            conda_env=pm.get("conda_env"),
        )
        results.append(res)
        print(f"[inverse_fold] {res.name} -> {len(res.sequences)} seq(s) -> {res.fasta_path}")

    print(f"[inverse_fold] inverse-folded {len(results)} design(s) into {out_dir}/seqs")
    return results
