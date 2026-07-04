"""Stage 1 of the SPA validation flywheel: generate RFD3 ± SPA designs from a trained adapter.

Spec: dev ``05_validation_pipeline.md`` §1–§2 ("Stage 0 — Generate (RFD3 ± SPA)") and the
identity-gate invariant of dev ``02``/``03`` (wrapped-no-prompt == vanilla RFD3). This is the
**inference** path — RFD3's real multi-step diffusion sampler (``RFD3InferenceEngine.run``; 100-step
default per the rfd3 ``edm.yaml``, NOT the 200 its docs claim — see dev ``07`` I.10), NOT the
single training forward the harness uses.

How generation is driven (vs the training harness):

- The harness runs ONE denoising step under grad for a loss; here we run the FULL sampler under
  ``no_grad`` via ``engine.run(inputs=None, out_dir=None)`` — exactly the path ``rfd3 design`` and
  ``tests/test_identity_at_init.py`` exercise — to actually roll out a design. ``inputs=None``
  requests an unconditional design of ``eval.length`` residues (the ``specification`` knob).
- **K designs in one shot:** the engine's diffusion batch ``D = eval.num_designs`` ⇒ one
  ``engine.run`` rolls out K independent designs (K initial-noise draws), returned as K
  ``RFD3Output`` objects (one cleaned biotite ``AtomArray`` each).
- SPA attaches to the EMA ``shadow`` net inference actually uses (``harness.frozen_rfd3_net``); the
  wrapped blocks read the shared prompt side-channel. A **condition** selects the side-channel:
  ``baseline`` → :meth:`SPAAdapter.clear_prompt` (wrappers return base only ⇒ vanilla RFD3, the
  identity gate); ``spa`` → :meth:`SPAAdapter.set_prompt` (ESM3 prompt) + :meth:`set_scale` (λ).
- **Reproducibility / paired noise:** the RFD3 sampler draws its Gaussian noise from the *global*
  torch RNG (``inference_sampler.py`` ``torch.normal``), and ``BaseInferenceEngine`` only seeds at
  construction. We re-seed (``seed_everything(eval.seed)``) immediately before *every* sampler run,
  so (a) attaching/loading the adapter — which consumes RNG via random inits — cannot perturb the
  noise, and (b) every (condition, λ) run starts from the *same* initial noise. That makes the
  baseline↔vanilla comparison and the λ-sweep clean paired comparisons, and is what makes the
  identity gate bit-for-bit (SPA consumes no RNG during the forward, so wrapped-no-prompt draws the
  identical noise sequence as vanilla).

Cost knobs (``eval.num_designs`` K, ``eval.lambda_scale`` λ, ``eval.length``, ``eval.num_timesteps``,
``eval.out_dir``, ``variant``, ``eval.ckpt``) are all config/CLI — nothing hardware- or
cost-specific is hardcoded (local A5000 → cloud H100 is a config change; dev root ``CLAUDE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Design:
    """One generated backbone (dev ``05`` Stage 0 output) plus its provenance.

    Attributes:
        prompt_id: output-name group id (prompt source stem or ``eval.prompt_id``).
        condition: ``"baseline"`` (vanilla RFD3) or ``"spa"`` (prompted).
        lambda_scale: the SPA strength λ used (0.0 for baseline).
        idx: index within the diffusion batch (0..K-1).
        path: the written PDB file.
        n_residues: residue count of the design (== ``eval.length`` for an unconditional monomer).
        atom_array: the RFD3 cleaned biotite ``AtomArray`` (kept in-memory; ``.coord`` for scoring).
    """

    prompt_id: str
    condition: str
    lambda_scale: float
    idx: int
    path: Path
    n_residues: int
    atom_array: Any = None


# --------------------------------------------------------------------------------------------------
# Engine + adapter setup
# --------------------------------------------------------------------------------------------------


def build_eval_engine(cfg):
    """Build + initialize the RFD3 inference engine for generation (loads frozen host weights).

    Mirrors ``harness.build_engine`` but reads the ``eval`` group: ``num_designs`` becomes the
    diffusion batch (== K designs/run), ``length``/``specification`` set the design spec, and
    ``num_timesteps`` (if given) overrides the sampler's 100-step default (rfd3 ``edm.yaml``) — the run-time cost knob.
    """
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    ev = cfg.eval
    spec = dict(ev.get("specification") or {})
    if ev.get("length") is not None:
        spec.setdefault("length", int(ev.length))
    sampler: dict = {}
    if ev.get("num_timesteps") is not None:
        sampler["num_timesteps"] = int(ev.num_timesteps)

    engine = RFD3InferenceEngine(
        **RFD3InferenceConfig(
            ckpt_path=cfg.paths.rfd3_ckpt,
            diffusion_batch_size=int(ev.num_designs),
            specification=spec,
            inference_sampler=sampler,
            seed=int(ev.get("seed", 0)),
            # OPT-IN trajectory dump (feature-flagged, default off; dev prototype). When
            # +eval.dump_trajectory=true, the engine builds per-step AtomArrayStacks onto each
            # RFD3Output (see generate() for the multi-MODEL PDB write). Off => byte-identical.
            dump_trajectories=bool(ev.get("dump_trajectory", False)),
        )
    )
    engine.initialize()
    return engine


def load_adapter(net, cfg, device):
    """Attach SPA to the frozen host, (optionally) load a trained checkpoint, match host dtype.

    Returns the :class:`~spa.model.wrapper.SPAAdapter`. With ``eval.ckpt=null`` the adapter is left
    at zero-init (identity) — useful for the baseline-only path and the smoke test. The adapter is
    cast to the host net's parameter dtype so it composes whatever precision the Fabric engine runs
    in: under ``bf16-mixed`` the host params stay float32 and Fabric's autocast handles compute
    (adapter stays float32, exactly as training); only the rare ``*-true`` half-precision host needs
    the explicit cast to avoid a dtype mismatch.
    """
    import torch

    from ..model import attach_spa

    adapter = attach_spa(net, cfg).to(device)
    if cfg.eval.get("ckpt"):
        from ..train.harness import load_spa

        load_spa(adapter, cfg.eval.ckpt)
    host_dtype = next(net.parameters()).dtype
    if host_dtype != torch.float32:
        adapter.to(dtype=host_dtype)
    return adapter


# --------------------------------------------------------------------------------------------------
# Prompt resolution (reuse the existing ESM3 producer — do not reinvent ESM3)
# --------------------------------------------------------------------------------------------------


def resolve_prompt(cfg, device):
    """Produce the structural prompt ``[N, c_kv]`` for the ``spa`` condition (dev ``05`` Stage 0).

    Two sources, both ending in the same ``[N, 1536]`` tensor :meth:`SPAAdapter.set_prompt` expects:

    - ``eval.prompt_cache``: a precomputed ``.pt`` (the training/cloud ESM3 cache format) — no ESM3
      load (matches how training reads cached prompts; cheap + the fast path for the smoke test).
    - ``eval.prompt_pdb``: a structure file → the existing :func:`spa.prompt.esm3_prompt.esm3_prompt`
      producer (the *same* structure-only ESM3 tap training used). ESM3 is loaded, run once, then
      freed (``del`` + ``empty_cache``) so it does not co-reside with RFD3 during the sampler —
      "ESM3 is run once and cached" (dev ``02`` §5, prompt is constant across all steps/blocks).
    """
    import torch

    ev = cfg.eval
    if ev.get("prompt_cache"):
        p = torch.load(ev.prompt_cache, weights_only=True).float().to(device)
        return p.squeeze(0) if p.dim() == 3 else p
    if ev.get("prompt_pdb"):
        from ..prompt.esm3_prompt import esm3_prompt, load_esm3

        model = load_esm3(device)
        try:
            p = esm3_prompt(
                ev.prompt_pdb, model,
                strip_bos_eos=bool(cfg.variant.get("strip_bos_eos", True)),
                use_sequence=bool(ev.get("use_sequence", False)),
            ).detach().float().to(device)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return p
    raise ValueError(
        "condition='spa' requires a prompt: set eval.prompt_pdb (a structure file) or "
        "eval.prompt_cache (a precomputed [N,1536] .pt)."
    )


def _prompt_id(cfg) -> str:
    ev = cfg.eval
    if ev.get("prompt_id"):
        return str(ev.prompt_id)
    for key in ("prompt_pdb", "prompt_cache"):
        if ev.get(key):
            return Path(str(ev[key])).stem
    return "design"


# --------------------------------------------------------------------------------------------------
# Native motif (hard conditioning) — Run-B hard⊕soft (dev 14 §1); absent ⇒ unconditional, unchanged
# --------------------------------------------------------------------------------------------------


def _parse_contig_motif(contig: str) -> list[tuple[int, str, int]]:
    """``(design_index, chain, source_resid)`` for each motif residue in an RFD3 contig (dev ``14`` §1).

    Grammar (the RFD3 dialect-2 subset we use; e.g. ``"59,A60-71,79"`` or 1CTT
    ``"74,A102,1,A104,24,A129,2,A132,75"``): comma-separated tokens walked left→right over the design
    sequence — a **bare integer** is a diffused scaffold gap of that many residues (advances the design
    cursor); a token starting with a **chain letter** (``A102`` or ``A60-71``) is a fixed motif segment
    pulled from the input. Each motif residue maps its **design-frame position** to its **source (chain,
    author-resid)** — the SPA prompt-mask consumes the design positions, while ``motif_rmsd`` consumes the
    *source positions* (via :func:`spa.eval.score.source_positions`), so a non-self-aligned /
    non-1-numbered / multi-chain source is scored correctly (review #1). Rejects variable-length
    (``min-max``) gaps and a motif-free contig (the design length must be fixed + well-defined).
    """
    import re

    out: list[tuple[int, str, int]] = []
    cursor = 0
    for tok in (t.strip() for t in str(contig).split(",")):
        if not tok:
            continue
        if tok[0].isalpha():                                   # motif segment from the input chain
            m = re.fullmatch(r"([A-Za-z]+)(\d+)(?:-(\d+))?", tok)
            if not m:
                raise ValueError(f"contig: cannot parse motif token {tok!r}")
            chain = m.group(1)
            start = int(m.group(2))
            end = int(m.group(3)) if m.group(3) else start
            if end < start:
                raise ValueError(f"contig: bad motif range {tok!r} (end < start)")
            for resid in range(start, end + 1):
                out.append((cursor, chain, resid))
                cursor += 1
        elif tok.isdigit():                                    # fixed-length diffused scaffold gap
            cursor += int(tok)
        else:                                                  # variable 'min-max' gap or junk
            raise ValueError(
                f"contig: token {tok!r} unsupported for motif eval — use fixed-int gaps + chain-prefixed "
                f"motif segments (no variable 'min-max' gaps; dev 14 §1)."
            )
    if not out:
        raise ValueError(f"contig {contig!r} has no motif segments")
    return out


def _parse_contig_motif_indices(contig: str) -> list[int]:
    """Design-frame 0-based indices of the motif residues (see :func:`_parse_contig_motif`)."""
    return [d for (d, _chain, _resid) in _parse_contig_motif(contig)]


def _contig_length(contig: str) -> int:
    """Total design length a contig implies (Σ gap lengths + motif residue counts = the final cursor).

    Used to assert the SPA prompt and the motif contig are the same length (dev ``14`` §0/§2; review #3/#6).
    """
    import re

    cursor = 0
    for tok in (t.strip() for t in str(contig).split(",")):
        if not tok:
            continue
        if tok[0].isalpha():
            m = re.fullmatch(r"([A-Za-z]+)(\d+)(?:-(\d+))?", tok)
            if not m:
                raise ValueError(f"contig: cannot parse motif token {tok!r}")
            start, end = int(m.group(2)), (int(m.group(3)) if m.group(3) else int(m.group(2)))
            if end < start:
                raise ValueError(f"contig: bad motif range {tok!r} (end < start)")
            cursor += end - start + 1
        elif tok.isdigit():
            cursor += int(tok)
        else:
            raise ValueError(f"contig: token {tok!r} unsupported for motif eval (dev 14 §1).")
    return cursor


def build_motif(cfg):
    """Build the native motif spec + its design-frame indices for the Run-B hard⊕soft eval (dev ``14`` §1).

    Reads ``eval.motif`` (``source_pdb``, ``contig``, optional ``fixed_atoms``); returns
    ``(DesignInputSpecification, motif_residues)``, or ``(None, None)`` when no motif is configured — the
    default, so the unconditional path stays byte-identical. The spec is what ``engine.run(inputs=…)``
    consumes directly (a ``DesignInputSpecification``; the engine's ``diffusion_batch_size`` still yields
    K designs for it). ``motif_residues`` are the 0-based design indices the SPA prompt-mask (§2) and
    ``motif_rmsd`` use. **Length comes from the contig** — ``eval.length`` is ignored when a motif is
    active (warned), per ``_canonicalize_inputs`` not merging ``specification_overrides`` onto a spec.
    """
    m = cfg.eval.get("motif")
    if not m:
        return None, None
    from rfd3.inference.input_parsing import DesignInputSpecification

    contig = str(m["contig"])
    residues = _parse_contig_motif_indices(contig)
    spec = DesignInputSpecification(
        input=str(m["source_pdb"]),
        contig=contig,
        select_fixed_atoms=m.get("fixed_atoms", True),
    )
    if cfg.eval.get("length") is not None:
        print("[generate] motif active -> design length is set by eval.motif.contig; eval.length ignored.")
    print(f"[generate] motif: {len(residues)} fixed residues at design indices "
          f"[{min(residues)}..{max(residues)}] from {m['source_pdb']} (contig {contig!r})")
    return spec, residues


# --------------------------------------------------------------------------------------------------
# Sub-region "scaffolding" mask (soft-only; dev 17 §7 / 16 §9.5) — SPA conditions on a sub-region S
# of the prompt only. This is the OPPOSITE polarity to the Run-B motif mask (which masks the motif so
# SPA attends to the scaffold): here we KEEP S and mask its complement, so SPA attends to S's rows
# only. No native RFD3 motif is placed — the design is unconditional-length (== N == prompt length).
# --------------------------------------------------------------------------------------------------


def subregion_keep(cfg) -> list[int] | None:
    """Sorted 0-based indices of the kept sub-region S (``None`` if ``eval.subregion`` unset).

    Two forms: ``eval.subregion.keep`` = an explicit index list, or ``eval.subregion.keep_range`` =
    ``[start, end)`` (compact — the contiguous S the domain/segment samplers produce; avoids a long
    CLI list on the cloud). Exactly one must be given.
    """
    sr = cfg.eval.get("subregion")
    if not sr:
        return None
    keep = sr.get("keep") if hasattr(sr, "get") else None
    krange = sr.get("keep_range") if hasattr(sr, "get") else None
    if keep is None and krange is None:
        raise ValueError("eval.subregion is set but has neither `keep` (index list) nor `keep_range` [start,end).")
    if krange is not None:
        lo, hi = int(krange[0]), int(krange[1])
        if hi <= lo:
            raise ValueError(f"eval.subregion.keep_range must be [start,end) with end>start; got {list(krange)}")
        idxs = range(lo, hi)
    else:
        idxs = keep
    out = sorted({int(i) for i in idxs})
    if not out:
        raise ValueError("eval.subregion sub-region S is empty — S must contain at least one residue.")
    return out


def subregion_key_padding_mask(keep, N: int, K: int, device):
    """``[K, N]`` bool key-padding mask, ``True`` at rows ``∉ keep`` (masked), for the sub-region eval.

    Returns ``None`` when ``keep`` spans all N rows (⇒ no masking, the global/full-prompt control).
    Guards the indices against the prompt length ``N`` (a keep index ≥ N is a driver/length bug).
    """
    import torch

    if min(keep) < 0 or max(keep) >= N:
        raise ValueError(f"eval.subregion.keep index out of range for prompt length N={N}: "
                         f"kept∈[{min(keep)},{max(keep)}] (need eval.length == N).")
    if len(keep) >= N:
        print(f"[generate] subregion: keep spans all N={N} rows -> no mask (global/full-prompt control).")
        return None
    mask = torch.ones(K, N, dtype=torch.bool, device=device)
    mask[:, keep] = False   # attend to S's rows only
    print(f"[generate] subregion mask: SPA attends to {len(keep)}/{N} rows "
          f"[{min(keep)}..{max(keep)}] (masked {N - len(keep)} non-S rows).")
    return mask


# --------------------------------------------------------------------------------------------------
# Output (F1.5.2 CIF→PDB, done in-memory from the RFD3 AtomArray)
# --------------------------------------------------------------------------------------------------


def _resolve_out_dir(out_dir) -> Path:
    """Resolve ``eval.out_dir``; a relative path resolves against the ORIGINAL cwd under Hydra
    (Hydra chdir's into its run dir), else against the current cwd (direct calls / tests)."""
    p = Path(str(out_dir)).expanduser()
    if p.is_absolute():
        return p
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.utils import get_original_cwd

        if HydraConfig.initialized():
            return Path(get_original_cwd()) / p
    except Exception:
        pass
    return Path.cwd() / p


def _fmt_lambda(value: float) -> str:
    return f"{float(value):g}"


def write_pdb(atom_array, path: Path) -> int:
    """Write an RFD3 biotite ``AtomArray`` to PDB (the dev ``05`` F1.5.2 CIF→PDB role, in-memory —
    RFD3's native dump is mmCIF; ProteinMPNN's ``parse_PDB`` wants PDB). Returns the residue count.

    Uses biotite directly (present in ``spa-dev``; gemmi is not) — the AtomArray is the cleaned,
    guidepost/virtual-atom-stripped protein the engine would otherwise serialize to ``.cif.gz``.
    """
    from biotite.structure import get_residue_count
    from biotite.structure.io.pdb import PDBFile

    path.parent.mkdir(parents=True, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(str(path))
    return int(get_residue_count(atom_array))


def _write_sidecar(path: Path, design: Design, cfg, metadata) -> None:
    """Minimal provenance sidecar ``.json`` next to each PDB (dev ``05``: ``.cif.gz`` + sidecar
    ``.json``). Best-effort: provenance is informational, never load-bearing."""
    import json

    rec = {
        "prompt_id": design.prompt_id,
        "condition": design.condition,
        "lambda_scale": design.lambda_scale,
        "idx": design.idx,
        "n_residues": design.n_residues,
        "seed": int(cfg.eval.get("seed", 0)),
        "variant": cfg.variant.get("name"),
        "spa_ckpt": cfg.eval.get("ckpt"),
        "length": cfg.eval.get("length"),
        "num_timesteps": cfg.eval.get("num_timesteps"),
        "rfd3_metadata": metadata or {},
    }
    try:
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump(rec, fh, indent=2, default=str)
    except Exception:
        pass


# --------------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    """Re-seed every RNG the RFD3 sampler reads, right before a run (see module docstring)."""
    from lightning.fabric import seed_everything

    seed_everything(int(seed), workers=True, verbose=False)


def _normalize_conditions(value) -> list[str]:
    if value is None:
        return ["baseline"]
    conds = [value] if isinstance(value, str) else list(value)
    # control ablations (dev 06): 'nullprompt' = SPA live on the learned null token e∅ (no real prompt);
    # 'shuffle' = SPA fed a row-permuted prompt (scrambled structure). Both config-gated add-ons.
    # ('nullprompt', NOT 'null' — a bare 'null' is a YAML/Hydra reserved literal that parses to None.)
    allowed = ("baseline", "spa", "nullprompt", "shuffle")
    for c in conds:
        if c not in allowed:
            raise ValueError(f"unknown condition {c!r} (expected one of {allowed})")
    return conds


def _normalize_lambdas(value) -> list[float]:
    if value is None:
        return [1.0]
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def _run_once(engine, spec=None) -> list:
    """Run the full RFD3 sampler once and return the K ``RFD3Output`` (one per diffusion-batch idx).

    ``spec`` is an optional native :class:`DesignInputSpecification` (the Run-B hard⊕soft motif, dev
    ``14`` §1); ``None`` ⇒ today's unconditional design (``inputs=None``). Either way the engine's
    ``diffusion_batch_size`` yields K designs for the single (motif or empty) spec, returned under one
    ``example_id`` — so the ``next(iter(...))`` below is correct in both modes.
    """
    outputs = engine.run(inputs=spec, out_dir=None)  # {example_id: [RFD3Output, ...]}
    if not outputs:
        raise RuntimeError("engine.run produced no outputs (empty design specification).")
    return next(iter(outputs.values()))


def generate(cfg, *, engine=None, adapter=None) -> list[Design]:
    """Generate RFD3 ± SPA designs from a composed config; write PDBs; return :class:`Design` records.

    Iterates ``eval.conditions`` × ``eval.lambda_scale``. Conditions: ``baseline`` (wrapped-no-prompt
    ≡ vanilla RFD3, runs once at λ=0); ``spa`` (the real structural prompt); and the control ablations
    ``nullprompt`` (SPA live on the learned null token e∅ — no real prompt) and ``shuffle`` (SPA fed a
    row-permuted prompt — scrambled structure). ``baseline`` runs once; ``spa``/``nullprompt``/``shuffle``
    sweep λ. Each run re-seeds to ``eval.seed`` then rolls out the full sampler for K =
    ``eval.num_designs`` designs, writing ``{prompt_id}_{condition}_lambda{λ}_{idx}.pdb`` (+ a small
    sidecar ``.json``) under ``eval.out_dir``.

    Args:
        cfg: composed config (``eval`` / ``model`` / ``variant`` / ``hardware`` / ``paths`` groups).
        engine: an already-built :class:`RFD3InferenceEngine` to reuse (built from ``cfg`` if None) —
            an injection point for tests/drivers that want one engine across calls.
        adapter: an already-attached :class:`~spa.model.wrapper.SPAAdapter` (attached + ckpt-loaded
            from ``cfg`` if None); must wrap ``engine``'s host net.
    """
    import torch

    from ..train.harness import frozen_rfd3_net
    from ..utils.device import resolve_device

    ev = cfg.eval
    device = resolve_device(cfg.hardware.device)
    conditions = _normalize_conditions(ev.get("conditions", "baseline"))
    lambdas = _normalize_lambdas(ev.get("lambda_scale", 1.0))
    seed = int(ev.get("seed", 0))
    K = int(ev.num_designs)
    out_dir = _resolve_out_dir(ev.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = _prompt_id(cfg)

    if engine is None:
        engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    if adapter is None:
        adapter = load_adapter(net, cfg, device)
    adapter.eval()
    adapter_dtype = next(adapter.parameters()).dtype

    # Native motif (hard) for the Run-B hard⊕soft eval — applied to BOTH conditions (baseline = motif-only
    # RFD3; spa = motif ⊕ SPA). None ⇒ unconditional (today's path, unchanged). dev 14 §1.
    motif_spec, motif_residues = build_motif(cfg)

    # Sub-region "scaffolding" eval (dev 17 §7 / 16 §9.5): SPA conditions on a sub-region S of the
    # prompt only (mask non-S rows). Mutually exclusive with the native motif — opposite mask polarity
    # (motif masks S so SPA does the scaffold; this KEEPS S so SPA does only the sub-region). No native
    # motif is placed, so the design is unconditional-length (== N == prompt length).
    subregion = subregion_keep(cfg)
    if subregion is not None and motif_residues is not None:
        raise ValueError("eval.subregion and eval.motif are mutually exclusive (subregion keeps S; "
                         "motif masks S). Set only one.")

    # Resolve + batch the prompt to [K, N, c_kv] once (constant across the diffusion batch). Needed by
    # 'spa' (real prompt) and 'shuffle' (row-permuted prompt); the 'null'/'baseline' controls need none.
    prompt_batched = None
    shuffle_batched = None
    prompt_mask = None
    if any(c in ("spa", "shuffle") for c in conditions):
        p = resolve_prompt(cfg, device)              # [N, c_kv]
        if motif_residues is not None:               # non-overlap: SPA attends to the scaffold rows only (§2)
            N = p.shape[0]
            contig = str(cfg.eval.motif["contig"])
            L = _contig_length(contig)
            if N != L:                               # review #3/#6: prompt must match the contig design length
                raise ValueError(
                    f"SPA prompt length N={N} != contig design length L={L} (contig {contig!r}). The prompt "
                    f"and the motif contig must be the same length — check for a BOS/EOS-unstripped prompt "
                    f"cache or a cross-length/cross-fold prompt (dev 14 §0/§2)."
                )
            bad = [i for i in motif_residues if i >= N]
            if bad:
                raise ValueError(
                    f"motif residue index ≥ prompt length {N}: {bad} — design/contig misalignment (dev 14 §0/§2)."
                )
            if len(motif_residues) >= N:             # review #4: all-motif contig ⇒ every row masked ⇒ NaN softmax
                raise ValueError(
                    f"all-motif contig: {len(motif_residues)} motif rows of N={N} leaves no scaffold row for SPA "
                    f"to attend → masked softmax would be NaN. Use a contig with diffused gaps."
                )
            prompt_mask = torch.zeros(K, N, dtype=torch.bool, device=device)
            prompt_mask[:, motif_residues] = True
            print(f"[generate] SPA prompt-mask: {len(motif_residues)} motif rows masked of N={N} (non-overlap).")
        elif subregion is not None:                  # sub-region scaffolding: SPA attends to S's rows only
            prompt_mask = subregion_key_padding_mask(subregion, p.shape[0], K, device)
        if "spa" in conditions:
            prompt_batched = p[None].expand(K, -1, -1).to(device=device, dtype=adapter_dtype).contiguous()
        if "shuffle" in conditions:                  # control: permute prompt rows ⇒ scrambled structure
            perm = torch.randperm(p.shape[0], generator=torch.Generator().manual_seed(seed)).to(p.device)
            shuffle_batched = p[perm][None].expand(K, -1, -1).to(device=device, dtype=adapter_dtype).contiguous()
            print(f"[generate] SPA prompt-shuffle control: permuted {p.shape[0]} prompt rows (seed {seed}).")

    designs: list[Design] = []
    for condition in conditions:
        run_lambdas = [0.0] if condition == "baseline" else lambdas  # spa/null/shuffle sweep λ; baseline once
        for lam in run_lambdas:
            if condition == "baseline":
                adapter.clear_prompt()               # wrappers return base only == vanilla RFD3 (± native motif)
            elif condition == "nullprompt":          # control: SPA live on the learned null token e∅ (no real prompt)
                adapter.set_null_prompt(K)
                adapter.set_scale(lam)
            elif condition == "shuffle":             # control: SPA fed the row-permuted (scrambled) prompt
                adapter.set_prompt(shuffle_batched, key_padding_mask=prompt_mask)
                adapter.set_scale(lam)
            else:                                    # spa: the real structural prompt
                adapter.set_prompt(prompt_batched, key_padding_mask=prompt_mask)
                adapter.set_scale(lam)

            _seed_all(seed)                          # paired noise + identity-gate determinism
            with torch.no_grad():
                output_list = _run_once(engine, motif_spec)

            lam_label = 0.0 if condition == "baseline" else float(lam)
            for idx, rfd3_out in enumerate(output_list):
                name = f"{pid}_{condition}_lambda{_fmt_lambda(lam_label)}_{idx}.pdb"
                path = out_dir / name
                aa = rfd3_out.atom_array
                n_res = write_pdb(aa, path)
                design = Design(prompt_id=pid, condition=condition, lambda_scale=lam_label,
                                idx=idx, path=path, n_residues=n_res, atom_array=aa)
                _write_sidecar(path, design, cfg, getattr(rfd3_out, "metadata", None))
                designs.append(design)
                # OPT-IN per-step trajectory dump (feature-flagged, default off; dev prototype).
                # When +eval.dump_trajectory=true the engine attaches per-step AtomArrayStacks to the
                # RFD3Output; persist each as a multi-MODEL PDB alongside the design. NOTE: the foundry
                # engine (engine.py:306-309) CROSSES the two field labels, so we dump both series under
                # their RAW field names — pick the "clean refining" one by CONTENT downstream, not name.
                if bool(ev.get("dump_trajectory", False)):
                    from biotite.structure.io.pdb import PDBFile as _PDBFile
                    for _field in ("denoised_trajectory_stack", "noisy_trajectory_stack"):
                        _stack = getattr(rfd3_out, _field, None)
                        if _stack is None:
                            continue
                        _tp = path.with_name(f"{path.stem}_traj_{_field.split('_')[0]}.pdb")
                        _pf = _PDBFile()
                        _pf.set_structure(_stack)
                        _pf.write(str(_tp))
                        print(f"[generate] trajectory[{_field}]: {len(_stack)} frames -> {_tp}")
            print(f"[generate] {condition} λ={_fmt_lambda(lam_label)} -> {len(output_list)} design(s)")

    print(f"[generate] wrote {len(designs)} design(s) to {out_dir}")
    return designs
