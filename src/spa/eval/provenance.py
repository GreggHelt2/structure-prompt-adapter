"""Write a self-describing ``RUN_PROVENANCE.json`` next to a run's outputs.

WHY THIS EXISTS. An audit on 2026-07-31 of every run this project has ever produced found that the
as-run configuration survived only by accident:

- Of 104 local run directories, **26 carried no recoverable config at all** (no Hydra snapshot, no
  embedded ``config`` block), and the artifacts cannot be regenerated to recover it because RFD3's
  CUDA sampler is not bitwise reproducible.
- The single best provenance format in the project was a hand-written ``RUN_PROVENANCE.json``, which
  is exactly why it existed for **2 of those 104 runs**. Anything that depends on remembering does
  not happen; anything a driver writes automatically survives.
- Of 93 cloud jobs, none pinned a code revision or an image digest, and the job stdout logs that
  echoed the resolved config were already past retention when we went looking.

So this module exists to make the record a by-product of running, not a thing to remember. It is
deliberately **best-effort and non-fatal**: a provenance failure must never take down a run that has
already spent GPU time. Every collector is individually guarded.

WHAT IT RECORDS, and why each field is here rather than assumed:

``code``        git sha AND dirty flag for both repos. Cloud jobs now pin a sha at submit
                (``scripts/cloud/_pin_run_env.sh``); local runs had nothing, and a dirty tree means
                the sha alone does not identify what ran.
``checkpoint``  path plus **sha256** of the adapter. `results/17` had to state its checkpoint hash by
                hand to make the claim checkable. The RFD3 host is recorded by path/size/mtime rather
                than hash, because hashing 2.6 GB on every run is not worth eight seconds.
``prompts``     id, path, length and ⭐ **split**. `results/25` ran 80 prompts of which 68 were
                train-split and nobody noticed for three days, by which time the claim was in the
                paper. Looking the split up at run time turns that into a warning at second zero.
``sampler``     the requested knobs. ``generate.py`` separately reads the values back off the live
                sampler and aborts on mismatch, so requested == effective by construction.
``env``         host, GPU, conda env, python/torch, because "which box" has silently mattered before.
``timing``      start and end in **Pacific**, per the project convention.

The two fields a machine cannot supply are ``purpose`` and ``scope``, and ``scope`` is the one that
prevents later over-reading: "ProteinMPNN and OpenFold3 were NOT run, so no designability numbers
exist here". Callers should pass them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FILENAME = "RUN_PROVENANCE.json"
_PACIFIC = "America/Los_Angeles"


def _now_pacific() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(_PACIFIC)).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _git(repo: str | os.PathLike) -> dict[str, Any]:
    out: dict[str, Any] = {"repo": str(repo)}
    try:
        out["sha"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                             stderr=subprocess.DEVNULL, timeout=15).decode().strip()
        rc = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD"],
                            stderr=subprocess.DEVNULL, timeout=15).returncode
        out["dirty"] = bool(rc)
    except Exception:
        out["sha"] = None
    return out


def _sha256(path: str | os.PathLike, limit_mb: int = 512) -> str | None:
    """Hash a file, but skip anything above ``limit_mb`` so a 2.6 GB host checkpoint is not rehashed
    on every run. Returns None when skipped or unreadable."""
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size > limit_mb * 1024 * 1024:
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _file_id(path) -> dict[str, Any] | None:
    try:
        p = Path(str(path))
        st = p.stat()
        return {"path": str(p), "bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "sha256": _sha256(p)}
    except Exception:
        return {"path": str(path), "exists": False} if path else None


def lookup_split(accession: str, splits_root: str | os.PathLike | None) -> str | None:
    """Which split of the pinned partition an accession belongs to, or None if unresolvable.

    Best-effort by design: a public user has no split manifests, and a run must not fail because a
    provenance nicety could not be computed.
    """
    if not splits_root or not accession:
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    for split in ("train", "validate", "test"):
        try:
            man = Path(str(splits_root)) / split / "manifest.parquet"
            if not man.exists():
                continue
            key = _split_cache(str(man))
            if accession in key:
                return split
        except Exception:
            continue
    return None


_SPLIT_CACHE: dict[str, set] = {}


def _split_cache(manifest: str) -> set:
    if manifest not in _SPLIT_CACHE:
        import pandas as pd
        df = pd.read_parquet(manifest, columns=["stem"])
        _SPLIT_CACHE[manifest] = {s.split("-")[1] for s in df["stem"] if "-" in s}
    return _SPLIT_CACHE[manifest]


def describe_prompt(path_or_id, splits_root=None) -> dict[str, Any]:
    """One prompt's record: id, path, and the split it came from (warning loudly if not held out)."""
    s = str(path_or_id) if path_or_id is not None else ""
    stem = Path(s).stem
    acc = stem.split("-")[1] if stem.startswith("AF-") and "-" in stem else stem
    rec: dict[str, Any] = {"id": acc, "source": s or None}
    rec["split"] = lookup_split(acc, splits_root)
    return rec


def collect(cfg=None, *, prompts=None, purpose=None, scope=None,
            extra: dict | None = None, started: str | None = None) -> dict[str, Any]:
    """Assemble the provenance record. Never raises."""
    rec: dict[str, Any] = {
        "schema": "spa-run-provenance/1",
        "purpose": purpose,
        "scope": scope,
        "started_pacific": started,
        "written_pacific": _now_pacific(),
        "anomalies": [],
    }
    try:
        here = Path(__file__).resolve().parents[3]
        rec["code"] = {"public": _git(here), "dev": _git(here.parent / "structure-prompt-adapter-dev")}
    except Exception:
        pass
    try:
        import torch
        rec["env"] = {
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    except Exception:
        rec["env"] = {"host": socket.gethostname()}

    if cfg is not None:
        try:
            ev = cfg.eval
            rec["eval"] = {
                "conditions": _plain(ev.get("conditions", None)),
                "lambda_scale": _plain(ev.get("lambda_scale", None)),
                "num_designs": _plain(ev.get("num_designs", None)),
                "length": _plain(ev.get("length", None)),
                "seed": _plain(ev.get("seed", None)),
                "out_dir": str(ev.get("out_dir", "")),
            }
            rec["sampler_requested"] = {k: _plain(ev.get(k, None))
                                        for k in ("num_timesteps", "gamma_0", "step_scale")}
            rec["proteinmpnn"] = {k: _plain(ev.proteinmpnn.get(k, None))
                                  for k in ("seed", "num_seqs", "sampling_temp")} \
                if ev.get("proteinmpnn", None) is not None else None
            rec["checkpoint"] = _file_id(ev.get("ckpt", None)) if ev.get("ckpt", None) else None
            try:
                rec["rfd3_ckpt"] = _file_id(cfg.paths.rfd3_ckpt)
            except Exception:
                pass
            splits_root = None
            try:
                splits_root = cfg.data.get("splits_root", None)
            except Exception:
                pass
            if prompts is None:
                prompts = [p for p in (ev.get("prompt_pdb", None), ev.get("prompt_cache", None)) if p]
            rec["prompts"] = [describe_prompt(p, splits_root) for p in (prompts or [])]
        except Exception as exc:  # config shapes vary across drivers; record rather than fail
            rec["anomalies"].append(f"config capture partial: {type(exc).__name__}: {exc}")
    elif prompts:
        rec["prompts"] = [describe_prompt(p) for p in prompts]

    not_heldout = [p["id"] for p in rec.get("prompts", []) if p.get("split") not in (None, "test")]
    if not_heldout:
        rec["anomalies"].append(
            "PROMPTS NOT HELD OUT: " + ", ".join(f"{p['id']}={p['split']}"
                                                 for p in rec["prompts"]
                                                 if p.get("split") not in (None, "test")))
    if extra:
        rec["extra"] = _plain(extra)
    return rec


def _plain(v):
    """OmegaConf containers and Paths to plain JSON-able values."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(v):
            return OmegaConf.to_container(v, resolve=True)
    except Exception:
        pass
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def write(out_dir, cfg=None, **kw) -> str | None:
    """Write ``RUN_PROVENANCE.json`` into ``out_dir``. Returns the path, or None on failure.

    Never raises: a provenance failure must not kill a run that has already spent GPU time. If the
    file exists (a multi-stage run writing into one directory), numbered siblings are added rather
    than overwriting, because the earlier stage's record is not redundant.
    """
    try:
        d = Path(str(out_dir))
        d.mkdir(parents=True, exist_ok=True)
        path = d / FILENAME
        n = 1
        while path.exists():
            path = d / f"{Path(FILENAME).stem}.{n}.json"
            n += 1
        rec = collect(cfg, **kw)
        path.write_text(json.dumps(rec, indent=2, default=str))
        warn = [a for a in rec.get("anomalies", []) if a.startswith("PROMPTS NOT HELD OUT")]
        for w in warn:
            print(f"[provenance] ⚠️  {w}")
        print(f"[provenance] wrote {path}")
        return str(path)
    except Exception as exc:
        print(f"[provenance] WARNING: could not write provenance ({type(exc).__name__}: {exc})")
        return None
