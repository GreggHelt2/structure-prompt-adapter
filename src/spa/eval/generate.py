"""Stage 1 of the SPA validation flywheel: generate RFD3 ± SPA designs from a trained adapter.

Spec: dev ``05_validation_pipeline.md`` §1–§2 ("Stage 0 — Generate (RFD3 ± SPA)") and the
identity-gate invariant of dev ``02``/``03`` (wrapped-no-prompt == vanilla RFD3). This is the
**inference** path — RFD3's real 200-step diffusion sampler (``RFD3InferenceEngine.run``), NOT the
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
    ``num_timesteps`` (if given) overrides the 200-step sampler — the run-time cost knob.
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
    for c in conds:
        if c not in ("baseline", "spa"):
            raise ValueError(f"unknown condition {c!r} (expected 'baseline' or 'spa')")
    return conds


def _normalize_lambdas(value) -> list[float]:
    if value is None:
        return [1.0]
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def _run_once(engine) -> list:
    """Run the full RFD3 sampler once and return the K ``RFD3Output`` (one per diffusion-batch idx)."""
    outputs = engine.run(inputs=None, out_dir=None)  # {example_id: [RFD3Output, ...]}
    if not outputs:
        raise RuntimeError("engine.run produced no outputs (empty design specification).")
    return next(iter(outputs.values()))


def generate(cfg, *, engine=None, adapter=None) -> list[Design]:
    """Generate RFD3 ± SPA designs from a composed config; write PDBs; return :class:`Design` records.

    Iterates ``eval.conditions`` × ``eval.lambda_scale`` (λ applies to ``spa`` only; ``baseline``
    runs once at λ=0). Each run re-seeds to ``eval.seed`` then rolls out the full sampler for K =
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

    # Resolve + batch the prompt to [K, N, c_kv] once (constant across the diffusion batch) — only if
    # any 'spa' run is requested.
    prompt_batched = None
    if "spa" in conditions:
        p = resolve_prompt(cfg, device)              # [N, c_kv]
        prompt_batched = p[None].expand(K, -1, -1).to(device=device, dtype=adapter_dtype).contiguous()

    designs: list[Design] = []
    for condition in conditions:
        run_lambdas = lambdas if condition == "spa" else [0.0]  # baseline ignores λ (clear_prompt)
        for lam in run_lambdas:
            if condition == "baseline":
                adapter.clear_prompt()               # wrappers return base only == vanilla RFD3
            else:
                adapter.set_prompt(prompt_batched)
                adapter.set_scale(lam)

            _seed_all(seed)                          # paired noise + identity-gate determinism
            with torch.no_grad():
                output_list = _run_once(engine)

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
            print(f"[generate] {condition} λ={_fmt_lambda(lam_label)} -> {len(output_list)} design(s)")

    print(f"[generate] wrote {len(designs)} design(s) to {out_dir}")
    return designs
