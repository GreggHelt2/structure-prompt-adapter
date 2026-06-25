"""ESM3 prompt producer: structure -> per-residue (N, 1536) structural prompt, + a cache driver.

Kickoff step 5. Spec: dev ``01_codebase_analysis.md`` §3.5, ``02_attachment_points.md`` §6, ``03`` §1.

ESM3 is LOCAL (HuggingFace weights ``esm3_sm_open_v1``, already cached), frozen, run ONCE per
design (resolved: dev ``07`` C2 — not the Forge API; no ``ESM_API_KEY``). The canonical SPA prompt is
**structure-only** (``ESMProtein(coordinates=coords)`` with no sequence, dev ``01`` §3.5), so it
conditions on topology rather than sequence identity::

    pt  = esm3.encode(ESMProtein(coordinates=coords))            # structure tokens (VQ-VAE)
    emb = esm3.logits(pt, LogitsConfig(return_embeddings=True)).embeddings   # [1, L+2, 1536] fp32

ESM3 prepends BOS / appends EOS, so N = L+2; strip rows 0 and L+1 for a per-residue prompt aligned
to RFD3's L residues (recommended; dev ``01`` §3.2). Embeddings are pre-final-LayerNorm (``01`` §3.3)
— apply RMSNorm inside SPA (``SPAPromptKV``) if normalization is wanted.

Verified on the A5000 (toy PDB): structure-only -> (39,1536) -> stripped (37,1536); SE(3) rotation
cosine mean 0.99998 (the W1.1 invariance that justifies caching once and reusing across rotational
augmentations). A small local cache is enough for A5000 pipeline testing; the full ~251 GB cache is
generated on a cloud H100 (dev ``04`` §10).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_esm3(device=None):
    """Load the local, frozen ESM3 (``esm3_sm_open_v1``) on ``device`` (eval, requires_grad=False)."""
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
    # Force-load the lazy structure-token encoder (VQ-VAE) so it gets frozen too — it is created on
    # the first encode() with requires_grad=True otherwise (esm3.py:252 get_structure_encoder).
    model.get_structure_encoder()
    model.requires_grad_(False)
    model.eval()
    return model


def _coords_and_seq(structure, chain_id: str = "detect"):
    """Resolve ``structure`` (PDB path | ESMProtein | coordinates tensor) -> (coords[L,37,3], seq|None)."""
    from esm.sdk.api import ESMProtein

    if isinstance(structure, (str, Path)):
        prot = ESMProtein.from_pdb(str(structure), chain_id=chain_id)
        return prot.coordinates, prot.sequence
    if isinstance(structure, ESMProtein):
        return structure.coordinates, structure.sequence
    if torch.is_tensor(structure):
        return structure, None
    raise TypeError(f"unsupported structure input: {type(structure).__name__}")


def esm3_prompt(structure, esm3_model, strip_bos_eos: bool = True,
                use_sequence: bool = False, chain_id: str = "detect") -> torch.Tensor:
    """Return ESM3 per-residue embeddings for one structure.

    Args:
        structure: a PDB path, an ``ESMProtein``, or an ``[L,37,3]`` atom37 coordinate tensor.
        esm3_model: a loaded, frozen local ESM3 model.
        strip_bos_eos: drop rows 0 and L+1 so the prompt aligns to RFD3's L residues.
        use_sequence: also condition on sequence (default False = structure-only, dev ``01`` §3.5).
        chain_id: chain selection when ``structure`` is a PDB path ("detect" = first chain).

    Returns:
        ``[N, 1536]`` (N = L if stripped, else L+2), float32, on the model's device.
    """
    from esm.sdk.api import ESMProtein, LogitsConfig

    coords, seq = _coords_and_seq(structure, chain_id=chain_id)
    device = next(esm3_model.parameters()).device
    prot = ESMProtein(coordinates=coords.to(device), sequence=seq if use_sequence else None)
    with torch.no_grad():
        pt = esm3_model.encode(prot)
        emb = esm3_model.logits(pt, LogitsConfig(return_embeddings=True)).embeddings
    emb = emb.squeeze(0)  # [L+2, 1536]
    return emb[1:-1] if strip_bos_eos else emb


def build_cache(cfg, esm3_model=None) -> dict:
    """Generate a per-residue ESM3 prompt cache over a dataset split (kickoff step 5 / cloud step 9).

    Reads PDBs under ``cfg.data.pdb_dir`` (CDDB) or ``cfg.data.root`` (toy), writes one ``<stem>.pt`` per
    structure to ``cfg.out_dir`` in ``cfg.dtype`` (skipping existing). ``cfg.limit`` caps the number of
    structures (e.g. the cloud ~1k benchmark). Returns summary stats.
    """
    from spa.utils.device import resolve_device

    device = resolve_device(cfg.hardware.device)
    model = esm3_model or load_esm3(device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = DTYPES[cfg.dtype]
    length_cap = cfg.data.get("length_cap", None)
    limit = cfg.get("limit", None)  # cap #structures (e.g. the cloud ~1k benchmark); None = all

    # toy uses `root`; CDDB uses `pdb_dir` (overridable to the cloud untar location).
    pdb_root = cfg.data.get("pdb_dir") or cfg.data.get("root")
    if pdb_root is None:
        raise ValueError("data config must set `pdb_dir` (CDDB) or `root` (toy)")
    pdbs = sorted(Path(pdb_root).glob("*.pdb"))
    if limit is not None:
        pdbs = pdbs[: int(limit)]
    stats = {"n_done": 0, "n_skipped": 0, "n_too_long": 0, "n_failed": 0, "bytes": 0, "out_dir": str(out_dir)}
    total = len(pdbs)
    prog_sec = float(cfg.get("progress_every_sec", 60))  # live [progress] cadence; stdout -> Cloud Logging
    t0 = time.time()
    last_log = t0
    print(f"[progress] start: {total} structures, length_cap={length_cap}, out={out_dir}", flush=True)
    for i, pdb in enumerate(pdbs):
        dst = out_dir / f"{pdb.stem}.pt"
        if dst.exists():
            stats["n_skipped"] += 1
        else:
            try:
                emb = esm3_prompt(pdb, model, strip_bos_eos=cfg.strip_bos_eos)
                if length_cap is not None and emb.shape[0] > length_cap:
                    stats["n_too_long"] += 1
                else:
                    emb = emb.to("cpu", save_dtype).contiguous()
                    tmp = dst.with_suffix(".pt.tmp")
                    torch.save(emb, tmp)
                    tmp.replace(dst)  # atomic: dst.exists() <=> complete (safe skip + resume; no partial files)
                    stats["n_done"] += 1
                    stats["bytes"] += dst.stat().st_size
            except Exception as e:  # one bad/OOM structure must NOT kill a multi-hour run
                stats["n_failed"] += 1
                if stats["n_failed"] <= 20:
                    print(f"[warn] failed {pdb.name}: {type(e).__name__}: {e}", flush=True)
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        now = time.time()
        if now - last_log >= prog_sec or i + 1 == total:
            done, el = i + 1, now - t0
            rate = done / el if el > 0 else 0.0
            eta_h = (total - done) / rate / 3600 if rate > 0 else 0.0
            print(
                f"[progress] {done}/{total} ({100.0 * done / total:.1f}%) | "
                f"cached={stats['n_done']} toolong={stats['n_too_long']} skip={stats['n_skipped']} "
                f"fail={stats['n_failed']} | {rate:.1f} prot/s | elapsed {el / 60:.1f} min | "
                f"ETA {eta_h:.2f} h | cache {stats['bytes'] / 1e9:.2f} GB",
                flush=True,
            )
            last_log = now
    stats["seconds"] = round(time.time() - t0, 2)
    return stats
