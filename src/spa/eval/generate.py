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


#: The three sampler knobs reachable as TOP-LEVEL ``eval`` keys. Kept because every driver in this
#: repo and every invocation recorded in the dev docs uses them; new knobs go under ``eval.sampler``.
_LEGACY_SAMPLER_KEYS = ("num_timesteps", "gamma_0", "step_scale")


def _sampler_fields() -> dict:
    """``SampleDiffusionConfig``'s fields, by INTROSPECTION so this cannot go stale.

    Copying the field list would reintroduce exactly the drift this function exists to prevent: the
    host gains a knob, our copy does not, and setting it becomes a silent no-op again.
    """
    from rfd3.model.inference_sampler import SampleDiffusionConfig

    return dict(SampleDiffusionConfig.__dataclass_fields__)


def _coerce_config_value(name: str, value, field):
    """Coerce to the field's declared default type. OmegaConf can hand back strings."""
    default = getattr(field, "default", None)
    if isinstance(default, bool):
        # bool("false") is True, so parse text explicitly rather than casting.
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
            raise RuntimeError(f"eval.{name}: {value!r} is not a boolean")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def resolve_sampler_overrides(ev) -> dict:
    """Collect RFD3 sampler overrides from the ``eval`` group, refusing SILENT NO-OPS.

    **The problem this solves.** Hydra's ``+`` adds undeclared keys without complaint, so
    ``+eval.gamma_min=2.0`` used to land in ``cfg.eval``, never be read, and never warn: the run
    reported success and generated at the checkpoint's value. That is the same failure class as a
    driver setting the step count while leaving ``gamma_0`` alone, and it is worse, because there is
    no log line at all to contradict.

    **The surface.** ``SampleDiffusionConfig`` has ~21 fields and only three were ever reachable.
    Anything else now goes through the namespaced block::

        +eval.sampler.gamma_min=2.0
        +eval.sampler.use_classifier_free_guidance=true

    Namespaced rather than top-level on purpose: passing through any ``eval.<k>`` that happened to
    match a sampler field would mean a future ``eval`` key named ``p`` or ``kind`` silently becoming a
    sampler override. Today those sets intersect in exactly the three legacy names.

    Raises rather than guessing on: an unknown ``eval.sampler`` field, a knob set both ways at once,
    and a top-level ``eval.<k>`` that is a sampler field but not one of the three legacy names.
    """
    fields = _sampler_fields()
    out: dict = {}

    for key in _LEGACY_SAMPLER_KEYS:
        value = ev.get(key)
        if value is not None:
            out[key] = _coerce_config_value(f"sampler.{key}", value, fields[key])

    block = ev.get("sampler") or {}
    for key in list(block.keys()):
        key = str(key)
        if key not in fields:
            raise RuntimeError(
                f"eval.sampler.{key} is not a field of RFD3's SampleDiffusionConfig. "
                f"Valid fields: {sorted(fields)}"
            )
        value = block.get(key)
        if value is None:
            continue
        if key in out and _coerce_config_value(f"sampler.{key}", value, fields[key]) != out[key]:
            raise RuntimeError(
                f"sampler knob '{key}' set two ways: eval.{key}={out[key]} and "
                f"eval.sampler.{key}={value}. Set it once."
            )
        out[key] = _coerce_config_value(f"sampler.{key}", value, fields[key])

    # The guard that closes the trap. A top-level eval key naming a sampler field that SPA does not
    # read at top level would otherwise be accepted by Hydra and silently dropped here.
    for key in fields:
        if key in _LEGACY_SAMPLER_KEYS:
            continue
        if ev.get(key) is not None:
            raise RuntimeError(
                f"eval.{key} is an RFD3 sampler field but SPA does not read it at the top level, so "
                f"it would be SILENTLY IGNORED. Use eval.sampler.{key}={ev.get(key)!r} instead."
            )
    return out


#: ``RFD3InferenceConfig`` fields SPA computes itself, mapped to the key that actually controls each.
#: Refused inside ``eval.engine`` rather than silently losing to the derived value.
_ENGINE_DERIVED = {
    "ckpt_path": "paths.rfd3_ckpt",
    "diffusion_batch_size": "eval.num_designs",
    "specification": "eval.specification",
    "inference_sampler": "eval.sampler (or the legacy eval.num_timesteps / gamma_0 / step_scale)",
    "seed": "eval.seed",
    "dump_trajectories": "eval.dump_trajectory",
}


def _engine_fields() -> dict:
    """``RFD3InferenceConfig``'s fields, by introspection. Same rationale as :func:`_sampler_fields`."""
    from rfd3.engine import RFD3InferenceConfig

    return dict(RFD3InferenceConfig.__dataclass_fields__)


def resolve_engine_overrides(ev) -> dict:
    """Collect RFD3 *engine* overrides from ``eval.engine``, refusing silent no-ops.

    The engine twin of :func:`resolve_sampler_overrides`, and it exists for the same reason: SPA
    named 6 of ``RFD3InferenceConfig``'s 20 fields and the other 14 were unreachable, so RFD3+SPA
    could not be driven into configurations plain RFD3 supports. That was never a property of the
    adapter, only of this constructor, which re-implements engine construction rather than wrapping
    RFD3's CLI::

        +eval.engine.low_memory_mode=true
        +eval.engine.prevalidate_inputs=false
        +eval.engine.global_prefix=myrun_

    **The six SPA computes are refused here, not silently overridden**, because each already has a
    key that controls it and accepting both would make the effective value depend on argument order.

    ⚠️ **Unlike the sampler, these need no readback check.** Sampler overrides are merged into the
    checkpoint's ``train_cfg`` and can silently fail to land, which is what
    :func:`_assert_sampler_effective` exists for. Engine fields are dataclass keyword arguments: they
    are set by construction, and an unknown one is a ``TypeError`` at the call. The validation below
    is for a better error message and to catch the top-level-key trap, not because the value might
    not apply.
    """
    fields = _engine_fields()
    settable = sorted(set(fields) - set(_ENGINE_DERIVED))
    out: dict = {}

    block = ev.get("engine") or {}
    for key in list(block.keys()):
        key = str(key)
        if key in _ENGINE_DERIVED:
            raise RuntimeError(
                f"eval.engine.{key} is derived by SPA from {_ENGINE_DERIVED[key]}. Set that instead, "
                "so one knob has one spelling."
            )
        if key not in fields:
            raise RuntimeError(
                f"eval.engine.{key} is not a field of RFD3's RFD3InferenceConfig. "
                f"Valid fields: {settable}"
            )
        value = block.get(key)
        if value is None:
            continue
        out[key] = _coerce_config_value(f"engine.{key}", value, fields[key])

    if out.get("low_memory_mode"):
        # Verified in foundry, and its own comment calls it a HACK: engine.py:203 does
        # `os.environ["RFD3_LOW_MEMORY_MODE"] = "1"` and NOTHING in the repo ever unsets it. RFD3.py:43
        # then reads that variable to decide `use_chunked_pll`, which changes whether P_LL is passed to
        # the diffusion module at all. So this is not a per-engine setting, it is a PROCESS-GLOBAL
        # ONE-WAY LATCH: every model built later in the same process inherits it, including one
        # constructed with low_memory_mode=False. Measured by hitting it, when a test that built an
        # engine this way broke an unrelated training test later in the same pytest session with
        # "RFD3DiffusionModule.forward() missing 1 required positional argument: 'P_LL'".
        import sys
        for stream in (sys.stderr, sys.stdout):
            print(
                "[engine] WARNING: low_memory_mode=True sets the process-global env var "
                "RFD3_LOW_MEMORY_MODE=1, which RFD3 never clears. It changes the forward call "
                "convention (chunked P_LL) for EVERY model built later in this process, including "
                "ones that did not ask for it. Safe in a one-engine-per-process driver; unsafe in a "
                "loop that builds several engines. Unset it manually to undo.",
                file=stream, flush=True,
            )

    for key in fields:
        if key in _ENGINE_DERIVED:
            continue
        if ev.get(key) is not None:
            raise RuntimeError(
                f"eval.{key} is an RFD3 engine field but SPA does not read it at the top level, so "
                f"it would be SILENTLY IGNORED. Use eval.engine.{key}={ev.get(key)!r} instead."
            )
    return out


def resolve_specification(ev) -> dict:
    """``eval.specification`` as a **fully plain** dict, nested mappings included.

    ⚠️ **``dict(cfg.eval.specification)`` is not enough and this is not a style point.** It is shallow:
    top-level keys become a plain dict while any NESTED mapping stays an ``omegaconf.DictConfig``, and
    RFD3's selection validator declares it accepts ``str | bool | dict | None``
    (``InputSelection.from_any``, ``rfd3/inference/parsing.py:44``), so it rejects the container with
    ``Cannot convert <class 'omegaconf.dictconfig.DictConfig'> to InputSelection``.

    ⛔ **Measured 2026-09-04**: every binder cell carrying ``select_hotspots`` and every diffused-ligand
    cell carrying ``select_fixed_atoms`` failed this way. So ``eval.specification`` could carry scalars
    but **not any selection dict**, which is exactly what a multi-chain binder or a diffused ligand
    needs. The dev estimate's claim that this surface was "passed through verbatim, so fully open" was
    true of the passing-through and wrong about the openness (dev ``90`` §3a).

    ⭐ **``build_motif`` already guarded its own ``fixed_atoms`` against this same validator.** The guard
    simply never reached the parallel path, and nothing exercised it. Keep both in step.
    """
    from omegaconf import OmegaConf

    raw = ev.get("specification")
    if OmegaConf.is_config(raw):
        return dict(OmegaConf.to_container(raw, resolve=True) or {})
    return dict(raw or {})


def build_eval_engine(cfg):
    """Build + initialize the RFD3 inference engine for generation (loads frozen host weights).

    Mirrors ``harness.build_engine`` but reads the ``eval`` group: ``num_designs`` becomes the
    diffusion batch (== K designs/run), ``length``/``specification`` set the design spec.

    **Sampler overrides (``num_timesteps``, ``gamma_0``, ``step_scale``).** Anything left unset here is
    inherited from the *released checkpoint's* ``train_cfg.model.net.inference_sampler``, because
    ``BaseInferenceEngine`` starts from ``checkpoint["train_cfg"]`` and then applies
    ``inference_sampler_overrides`` as ``model.net.inference_sampler.{key}``. An EMPTY dict therefore
    leaves the checkpoint's values standing, which is how this project silently ran at the checkpoint's
    **100 steps and gamma_0 = 0.8** rather than the **200 / 0.6** used throughout the RFdiffusion3 paper
    (and applied by the shipped ``rfd3`` CLI). Only ``num_timesteps`` used to be exposed, so the other
    two deviations were not merely undocumented, they were *unreachable* from our config surface.

    Exposing ``gamma_0`` and ``step_scale`` is what makes the pipeline-calibration experiment possible:
    generating at RFD3's own settings is a prerequisite for comparing our designable rates to theirs.
    Defaults are unchanged: omit them and behaviour is byte-identical to before.
    """
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    # Opt-in bitwise determinism (dev plan/91 §3.2). No-op unless eval.deterministic is true, and
    # nothing is patched at import, so the default path stays byte-identical. Applied here because
    # build_eval_engine is the one choke point every driver routes through, including the probes
    # that drive the engine directly via _run_once.
    from .determinism import maybe_enable as _maybe_deterministic
    _maybe_deterministic(cfg)

    ev = cfg.eval
    # ⚠️ `dict()` alone is NOT enough: it is shallow, so a NESTED mapping (select_hotspots,
    # select_fixed_atoms, select_buried, cif_parser_args) stays an OmegaConf DictConfig and RFD3's
    # before-validator rejects it with "Cannot convert <class 'omegaconf.dictconfig.DictConfig'> to
    # InputSelection" (input_parsing.py InputSelection.from_any). Measured 2026-09-04: every binder
    # cell carrying `select_hotspots` and every diffused-ligand cell carrying `select_fixed_atoms`
    # failed this way. `build_motif` already guards its own `fixed_atoms` for exactly this reason;
    # this path did not, which made `eval.specification` unable to carry any selection dict.
    spec = resolve_specification(ev)
    if ev.get("length") is not None:
        spec.setdefault("length", int(ev.length))
    sampler = resolve_sampler_overrides(ev)
    engine_overrides = resolve_engine_overrides(ev)

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
            # Everything else RFD3's engine accepts, from eval.engine.<field>. Empty by default, so
            # omitting the block is byte-identical to before this existed.
            **engine_overrides,
        )
    )
    # initialize() RETURNS the merged config (checkpoint train_cfg + our overrides), which is the
    # very object used to construct the model. Verifying against it tests _assign_override directly.
    merged_cfg = engine.initialize()
    _assert_sampler_effective(merged_cfg, sampler)
    return engine


def _assert_sampler_effective(merged_cfg, requested: dict):
    """Verify the requested sampler overrides ACTUALLY landed in the config the model was built from.

    A knob that silently fails to apply is the worst outcome for a calibration run: the job reports
    "200 steps, gamma_0 0.6" and generates at the checkpoint's 100 / 0.8 anyway, with nothing
    downstream able to tell. That is not hypothetical, it is the exact bug that went undetected for
    this whole project (an empty override dict leaving the checkpoint's values standing).

    ``BaseInferenceEngine.initialize()`` returns the merged config, built by applying
    ``inference_sampler_overrides`` as ``model.net.inference_sampler.{key}`` onto
    ``checkpoint["train_cfg"]``, and that config is what constructs the model. So we read the values
    back off it. An earlier version walked live module attributes and could not find the sampler at
    all, which is why this reads config instead.

    Prints unconditionally on stdout AND stderr: container log levels swallowed ``logging.info``,
    making an earlier version of this check silently invisible. Raises only on a real mismatch.
    """
    import sys

    def _say(msg):
        print(msg, file=sys.stderr, flush=True)
        print(msg, flush=True)

    # Read back EVERY requested key, not a hardcoded three: eval.sampler.<field> can now request any
    # SampleDiffusionConfig field, and a knob whose landing is unverified is exactly what this
    # function exists to prevent. The three legacy names are always shown, requested or not, so the
    # log line still states the sampler configuration in full.
    keys = tuple(dict.fromkeys(_LEGACY_SAMPLER_KEYS + tuple(requested)))
    effective = {}
    try:
        from omegaconf import OmegaConf
        for k in keys:
            effective[k] = OmegaConf.select(merged_cfg, f"model.net.inference_sampler.{k}")
    except Exception:
        node = merged_cfg
        for attr in ("model", "net", "inference_sampler"):
            node = getattr(node, attr, None) if node is not None else None
        for k in keys:
            effective[k] = getattr(node, k, None) if node is not None else None

    if all(v is None for v in effective.values()):
        _say("[sampler] WARNING: could not read model.net.inference_sampler from the merged config; "
             f"requested={requested or '{} (checkpoint defaults)'}. "
             "If this run is a calibration, VERIFY MANUALLY before trusting it.")
        return

    _say(f"[sampler] EFFECTIVE: {effective}  (requested overrides: "
         f"{requested or '{} -> checkpoint defaults'})")

    def _differs(effective_value, requested_value) -> bool:
        """Numeric comparison when both sides are numbers, equality otherwise.

        `kind`, `solver` and `center_option` are strings and `float()` on them raises, which the
        earlier hardcoded-three version never had to handle.
        """
        num = (int, float)
        if isinstance(effective_value, num) and isinstance(requested_value, num) \
                and not isinstance(effective_value, bool) and not isinstance(requested_value, bool):
            return float(effective_value) != float(requested_value)
        return effective_value != requested_value

    mismatched = {
        k: (v, effective.get(k))
        for k, v in requested.items()
        if effective.get(k) is not None and _differs(effective[k], v)
    }
    if mismatched:
        raise RuntimeError(
            "Sampler override(s) did not take effect: "
            + ", ".join(f"{k}: requested {req}, effective {eff}" for k, (req, eff) in mismatched.items())
            + ". Generation would have run at the CHECKPOINT's settings. Refusing to continue, since "
              "a calibration run at the wrong sampler settings is worse than none."
        )


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
    from omegaconf import OmegaConf
    from rfd3.inference.input_parsing import DesignInputSpecification

    # `fixed_atoms` selects which atoms of the motif residues RFD3 pins. A bool (True = §4 all-atoms of the
    # contig residues), OR a **per-residue atom selection** — a dict/str like {"A120":"OG,CB,CA"} (RFD3
    # keywords ALL/BKBN/TIP or explicit atoms) — pinning sidechain *tip* atoms for atomic enzyme motifs (dev
    # 26 §8.1; foundry's enzyme_design.json format). RFD3's before-validator coerces a dict via
    # InputSelection.from_any (input_parsing.py:362); convert an OmegaConf DictConfig to a plain dict first.
    fixed_atoms = m.get("fixed_atoms", True)
    if OmegaConf.is_config(fixed_atoms):
        fixed_atoms = OmegaConf.to_container(fixed_atoms, resolve=True)
    spec_kwargs = {"input": str(m["source_pdb"]), "select_fixed_atoms": fixed_atoms}

    # Two motif modes (dev 26 §8.6). INDEXED (shape B): a `contig` fixes the positions → design-frame
    # indices are known up front. UNINDEXED (shape A, the paper's enzyme mode; dev 27 §5): `unindex` +
    # `length`, RFD3 chooses each design's motif positions (its diffused_index_map) → residues=None (the SPA
    # prompt-mask is moot under a foreign-fold prompt; the design-side motif-RMSD is scored post-hoc from
    # diffused_index_map, and the pin is guaranteed by RFD3's revealed-coord freeze).
    contig = m.get("contig")
    if contig:
        contig = str(contig)
        residues = _parse_contig_motif_indices(contig)
        spec_kwargs["contig"] = contig
        if m.get("unindex"):
            spec_kwargs["unindex"] = str(m["unindex"])
        if cfg.eval.get("length") is not None:
            print("[generate] motif active -> design length is set by eval.motif.contig; eval.length ignored.")
        where = (f"{len(residues)} fixed residues at design indices "
                 f"[{min(residues)}..{max(residues)}] (contig {contig!r})")
    else:
        unindex = m.get("unindex")
        if not unindex:
            raise ValueError("eval.motif needs either `contig` (indexed) or `unindex` (unindexed, + `length`).")
        length = m.get("length") if m.get("length") is not None else cfg.eval.get("length")
        if length is None:
            raise ValueError("unindexed motif (eval.motif.unindex, no contig) needs eval.motif.length or eval.length.")
        spec_kwargs["unindex"] = str(unindex)
        spec_kwargs["length"] = str(length)
        residues = None
        n_un = len([t for t in str(unindex).replace(" ", "").split(",") if t])
        where = f"{n_un} unindexed (model-placed) residues, length {length} (unindex {str(unindex)!r})"

    spec = DesignInputSpecification(**spec_kwargs)
    atom_note = "" if not isinstance(fixed_atoms, dict) else f", atom-level ({len(fixed_atoms)} sel)"
    print(f"[generate] motif: {where} from {m['source_pdb']}{atom_note}")
    return spec, residues


def motif_atom_spec(cfg_motif):
    """Per-fixed-atom scoring spec for an atomic (tip-atom) motif, or ``None`` for the bool/keyword path.

    Returns a list of records ``{"design_idx": int, "chain": str, "resid": int, "atoms": [names]}`` — one
    per motif residue carrying an **explicit** atom list in ``fixed_atoms`` — pairing the design-frame index
    (from the contig walk, :func:`_parse_contig_motif`) with the source ``(chain, author-resid)`` and the
    exact atom names RFD3 pinned. Consumed by :func:`spa.eval.score.motif_atom_rmsd` so scoring measures the
    *same* atoms the model held (dev 26 §8.6, item 2). Returns ``None`` (⇒ the caller keeps the Cα
    ``motif_rmsd``) unless **every** contig motif residue has an explicit comma-atom list — a keyword
    selector (ALL/BKBN/TIP) or an unlisted/range residue names no fixed atom set to score against.
    """
    from omegaconf import OmegaConf

    fa = cfg_motif.get("fixed_atoms")
    if OmegaConf.is_config(fa):
        fa = OmegaConf.to_container(fa, resolve=True)
    if not isinstance(fa, dict):
        return None
    parsed = _parse_contig_motif(str(cfg_motif["contig"]))     # [(design_idx, chain, author_resid), ...]
    spec = []
    for design_idx, chain, resid in parsed:
        val = fa.get(f"{chain}{resid}")
        if not isinstance(val, str) or val.strip().upper() in ("ALL", "BKBN", "TIP", ""):
            return None                                        # not an explicit atom list -> Cα scoring path
        atoms = [a.strip() for a in val.split(",") if a.strip()]
        if not atoms:
            return None
        spec.append({"design_idx": int(design_idx), "chain": str(chain), "resid": int(resid), "atoms": atoms})
    return spec or None


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


def write_pdb(atom_array, path: Path) -> tuple[int, Path]:
    """Write an RFD3 biotite ``AtomArray`` to PDB (the dev ``05`` F1.5.2 CIF→PDB role, in-memory —
    RFD3's native dump is mmCIF; ProteinMPNN's ``parse_PDB`` wants PDB). Returns ``(residue count,
    path actually written)`` -- the path can differ from the one requested, see the collision note below.

    Uses biotite directly (present in ``spa-dev``; gemmi is not) — the AtomArray is the cleaned,
    guidepost/virtual-atom-stripped protein the engine would otherwise serialize to ``.cif.gz``.
    """
    from biotite.structure import get_residue_count
    from biotite.structure.io.pdb import PDBFile

    import hashlib
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(atom_array)

    # ⛔ NEVER SILENTLY OVERWRITE A DESIGN. Serialize to memory first so the bytes can be compared.
    #
    # WHY THIS EXISTS. dev plan/30 section 1.2 records unrecoverable data loss of exactly this shape:
    # ProteinMPNN FASTAs carried a run-independent name and shared one directory, so "any filename
    # reused across dates kept only the last write", proving the mechanism with 160 cross-tree
    # collisions of which 0 were byte-identical. Designs have never hit it (measured 2026-09-09: 953
    # archived design directories, ZERO holding two configs), so this is defence in depth, not a fix.
    #
    # ⭐ WHY DEFLECT RATHER THAN REFUSE, which is the opposite of what the caller-side check does. By
    # the time we are here the structure has already been GENERATED: refusing would discard a real
    # computed design, which is also data loss, just a different one. So the rule is asymmetric.
    # Refuse BEFORE spending compute; never discard AFTER. A weird filename is recoverable; a
    # destroyed file is not.
    #
    # ⭐ WHY A CONTENT HASH RATHER THAN A CONFIG HASH. A config hash cannot separate two runs of the
    # SAME config, which on the stock nondeterministic build produce different structures (0.056 to
    # 2.469 A at one seed, dev plan/91 section 1.2), so a config-hash suffix would collide with itself.
    # A content hash always differs when the content differs, and it makes the common case free:
    # under deterministic=true a legitimate rerun writes identical bytes and this no-ops.
    buf = io.StringIO()
    pdb.write(buf)
    data = buf.getvalue().encode()

    if path.exists():
        existing = path.read_bytes()
        if existing == data:
            return int(get_residue_count(atom_array)), path   # idempotent rerun, nothing to do
        h = hashlib.md5(data).hexdigest()[:8]
        deflected = path.with_name(f"{path.stem}__{h}{path.suffix}")
        print(f"[generate] ⛔ COLLISION: {path.name} exists with DIFFERENT content. Not overwriting.\n"
              f"[generate]    wrote {deflected.name} instead. Two runs are sharing one out_dir with\n"
              f"[generate]    the same design identity; fix the driver's out_dir. See dev plan/30 1.2.",
              flush=True)
        path = deflected

    path.write_bytes(data)
    return int(get_residue_count(atom_array)), path


def _write_sidecar(path: Path, design: Design, cfg, metadata, seed: int | None = None) -> None:
    """Provenance sidecar ``.json`` next to each PDB (dev ``05``: ``.cif.gz`` + sidecar ``.json``).

    ⭐ THE SIDECAR MUST CARRY THE FULL REPRODUCIBILITY IDENTITY, because it is what travels WITH a
    design. Three fields were added 2026-09-09 after each was measured to change the structure:

      num_designs  K is the diffusion batch dimension, so design *i* at K=4 differs from design *i*
                   at K=8 by up to 2.433 A (dev ``results/44`` section 4.1). It was absent entirely:
                   a scan of 20,529 sidecars for K=1 returned ZERO while the same scan over 710
                   RUN_PROVENANCE.json files found 28 runs, so auditing K from design artifacts alone
                   said the project had never run at K=1.
      deterministic  the stock and patched builds differ by up to 2.469 A at one seed
                   (dev ``plan/91`` section 1.2) and were indistinguishable after the fact except by
                   grepping the run log for the "[determinism] ENABLED" banner.
      gpu          MEASURED 2026-09-09 (dev ``results/44`` section 4b): the same design at the same
                   seed on an A5000 against an H100, identical torch and CUDA, differs by 2.387 A
                   aligned Ca-RMSD at L=208, against a 2.0 A designability threshold. Platform is as
                   load-bearing as K.

    ⚠️ Sidecars written before 2026-09-09 lack all three. K is recoverable from RUN_PROVENANCE.json,
    the build only from the run log, and the platform from neither on cloud runs.

    Best-effort: nothing depends on it at run time.
    """
    import json

    rec = {
        "prompt_id": design.prompt_id,
        "condition": design.condition,
        "lambda_scale": design.lambda_scale,
        "idx": design.idx,
        "n_residues": design.n_residues,
        # ⛔ the seed this design was ACTUALLY generated at, which under eval.seeds is not cfg.eval.seed.
        "seed": int(seed if seed is not None else cfg.eval.get("seed", 0)),
        "num_designs": cfg.eval.get("num_designs"),      # K: part of the identity, see docstring
        "deterministic": bool(cfg.eval.get("deterministic", False)),
        "variant": cfg.variant.get("name"),
        "spa_ckpt": cfg.eval.get("ckpt"),
        "length": cfg.eval.get("length"),
        "num_timesteps": cfg.eval.get("num_timesteps"),
        "rfd3_metadata": metadata or {},
    }
    try:                                                  # platform, see docstring
        import torch
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["torch"] = torch.__version__
            rec["cuda"] = torch.version.cuda
    except Exception:
        pass
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


def _normalize_seeds(ev) -> list[int]:
    """``eval.seeds`` as a list, defaulting to ``[eval.seed]``.

    ⭐ Accepts a scalar or a list so ``eval.seeds=7`` and ``eval.seeds=[0,1,2]`` both work. ⛔ Duplicates
    are dropped and order is preserved: two identical seeds in one invocation would generate the same
    design twice and, worse, write it to the same path, which is the collision this whole feature has to
    avoid rather than create.
    """
    raw = ev.get("seeds", None)
    if raw is None:
        return [int(ev.get("seed", 0))]
    vals = [raw] if isinstance(raw, (int, str)) else list(raw)
    out: list[int] = []
    for v in vals:
        iv = int(v)
        if iv not in out:
            out.append(iv)
    if not out:
        raise ValueError("eval.seeds resolved to an empty list; omit it to use eval.seed instead")
    return out


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



def _build_lambda_profile(cfg):
    """Build the per-residue λ profile ``[I]`` from ``eval.profile``, or ``None`` for uniform λ.

    ⭐ **Why this exists.** Until 2026-09-05 every eval-driven run applied a **uniform λ over every
    design-frame token**. For a monomer that is what you want. For a **multimer** it also steers the fixed
    partner chain and the chain-break token, and for a **protein-ligand** design it steers the ligand's own
    tokens, so neither run isolates the channel it is about (dev ``results/42`` §9). The wrapper has
    supported per-residue profiles since the two-steer work (``SPAWrapper.set_profile``); only this CLI
    surface was missing.

    ⛔ **Explicit, never inferred from the contig.** ``tokens`` is the total design-frame token count and
    ``steered`` a list of ``[start, end)`` spans receiving λ; every other token gets 0. Deriving these from
    a contig would be a silent-failure surface on the generation path, and a wrong span would steer the
    wrong residues while producing plausible output. **Every bound is validated and a violation raises.**

    Config::

        eval.profile: {tokens: 215, steered: [[0, 100]]}
    """
    spec = getattr(getattr(cfg, "eval", cfg), "profile", None)
    if spec is None:
        return None
    import torch          # lazy, matching this module's import discipline
    tokens = int(spec["tokens"])
    spans = [list(map(int, sp)) for sp in spec["steered"]]
    if tokens <= 0:
        raise ValueError(f"eval.profile.tokens must be positive, got {tokens}")
    w = torch.zeros(tokens, dtype=torch.float32)
    for lo, hi in spans:
        if not (0 <= lo < hi <= tokens):
            raise ValueError(
                f"eval.profile.steered span [{lo}, {hi}) is out of range for tokens={tokens}. "
                "Spans are half-open design-frame indices; they are NOT inferred from the contig, so "
                "check the contig's own token count (designed residues + fixed chains) by hand."
            )
        w[lo:hi] = 1.0
    if float(w.sum()) == 0.0:
        raise ValueError("eval.profile.steered selects no tokens; omit `profile` for uniform λ instead.")
    return w

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

    # Record the as-run config BEFORE spending GPU time, so a crashed or killed run still leaves a
    # usable record (dev audit 2026-07-31: 26 of 104 local run dirs had no recoverable config, and
    # the artifacts cannot be regenerated because the sampler is not bitwise reproducible). Writing
    # it here covers every driver that routes through generate(); the probe drivers, which drive the
    # engine directly via _run_once, call spa.eval.provenance.write themselves.
    from . import provenance as _prov
    # ⛔ BEFORE the provenance write, not after. The record is written early on purpose (a killed run
    # must still leave one), so anything that must appear in it has to be decided by now. Enabling in
    # build_eval_engine alone recorded `enabled: false` on a run that was in fact deterministic,
    # which is precisely the misleading artifact dev plan/91 §3.2 requires this field to prevent.
    # enable_deterministic() is idempotent, so the build_eval_engine hook (which covers drivers that
    # never reach here) stays and simply no-ops the second time.
    from .determinism import maybe_enable as _maybe_deterministic
    _maybe_deterministic(cfg)
    _prov.write(out_dir, cfg, started=_prov._now_pacific())

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
        # `self_prompt` (default True) = the §4 hard⊕soft case: the SPA prompt IS the motif's own structure
        # (N == L), so the motif rows are masked out of the prompt (non-overlap, §7.2). `self_prompt=False`
        # = a FOREIGN fold prompt G that does not contain the motif (dev 26 §8.6 change #3) — see the elif.
        motif_self_prompt = (bool(cfg.eval.motif.get("self_prompt", True))
                             if cfg.eval.get("motif") else True)
        if motif_residues is not None and motif_self_prompt:  # non-overlap: SPA attends to scaffold rows only (§2)
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
        elif motif_residues is not None:             # foreign-fold prompt (self_prompt=False; dev 26 §8.6 #3):
            # SPA steers the scaffold toward a WHOLE foreign fold G that does NOT contain the motif, so there
            # are no motif rows to mask (the §7.2 non-overlap is moot) and N=|G| need not equal L. RFD3 still
            # pins the motif on the design side regardless (revealed coords); SPA attends to all of G. Leaving
            # prompt_mask = None ⇒ the full prompt is used; cross-attention handles N ≠ L (as localization does).
            print(f"[generate] SPA foreign-fold prompt (self_prompt=False): attending to all N={p.shape[0]} "
                  f"prompt rows; motif pinned design-side, no non-overlap mask (dev 26 §8.6).")
        elif subregion is not None:                  # sub-region scaffolding: SPA attends to S's rows only
            prompt_mask = subregion_key_padding_mask(subregion, p.shape[0], K, device)
        if "spa" in conditions:
            prompt_batched = p[None].expand(K, -1, -1).to(device=device, dtype=adapter_dtype).contiguous()
        if "shuffle" in conditions:                  # control: permute prompt rows ⇒ scrambled structure
            perm = torch.randperm(p.shape[0], generator=torch.Generator().manual_seed(seed)).to(p.device)
            shuffle_batched = p[perm][None].expand(K, -1, -1).to(device=device, dtype=adapter_dtype).contiguous()
            print(f"[generate] SPA prompt-shuffle control: permuted {p.shape[0]} prompt rows (seed {seed}).")

    # ⭐ Per-residue λ profile (dev results/42 §9-§10). None keeps today's uniform-λ behaviour exactly.
    profile_vec = _build_lambda_profile(cfg)
    if profile_vec is not None:
        n_on = int((profile_vec > 0).sum())
        print(f"[generate] λ PROFILE ACTIVE: {n_on} of {profile_vec.numel()} design-frame tokens steered "
              f"(the rest are held at λ=0).")

    # ⭐ eval.seeds: a LIST of seeds looped INSIDE the (condition, λ) cells (dev plan/100 §5).
    # Default [eval.seed], so an unset config runs exactly one seed and behaves identically to before.
    #
    # WHY IT EXISTS. Under the K=1 convention (plan/100) a design is identified by its SEED rather than
    # by its row in a diffusion batch, because K is part of the reproducibility identity: design i at
    # K=4 differs from design i at K=8 by up to 2.433 A (results/44 §4.1). Getting N designs therefore
    # means N seeds, and without this loop that is N separate invocations paying N model loads at ~9.1 s
    # each, which is not the cost model plan/100 §6a measured.
    #
    # ⛔ THE SEED MUST REACH THE FILENAME. It is now a WITHIN-invocation varying axis, alongside
    # condition and λ, so the filename has to discriminate it or every seed writes the same path and
    # silently keeps the last. That is the mechanism behind plan/30 §1.2's unrecoverable FASTA loss,
    # where a run-independent name met a shared directory. Axes that vary BETWEEN invocations are
    # discriminated by out_dir instead; measured 2026-09-09, 953 archived design directories hold
    # exactly zero cases of two configs sharing one, so that half of the invariant is holding.
    seeds = _normalize_seeds(ev)
    multi_seed = len(seeds) > 1

    designs: list[Design] = []
    for condition in conditions:
        run_lambdas = [0.0] if condition == "baseline" else lambdas  # spa/null/shuffle sweep λ; baseline once
        for lam in run_lambdas:
            if condition == "baseline":
                adapter.clear_prompt()               # wrappers return base only == vanilla RFD3 (± native motif)
                adapter.set_profile(None)            # never inherit a profile from a previous λ iteration
            elif condition == "nullprompt":          # control: SPA live on the learned null token e∅ (no real prompt)
                adapter.set_null_prompt(K)
                adapter.set_scale(lam)
                adapter.set_profile(profile_vec)
            elif condition == "shuffle":             # control: SPA fed the row-permuted (scrambled) prompt
                adapter.set_prompt(shuffle_batched, key_padding_mask=prompt_mask)
                adapter.set_scale(lam)
                adapter.set_profile(profile_vec)
            else:                                    # spa: the real structural prompt
                adapter.set_prompt(prompt_batched, key_padding_mask=prompt_mask)
                adapter.set_scale(lam)
                adapter.set_profile(profile_vec)

            for run_seed in seeds:
                _seed_all(run_seed)                  # paired noise + identity-gate determinism
                with torch.no_grad():
                    output_list = _run_once(engine, motif_spec)

                lam_label = 0.0 if condition == "baseline" else float(lam)
                for idx, rfd3_out in enumerate(output_list):
                    # ⛔ The seed appears ONLY when more than one is running. A single-seed run keeps
                    # today's exact name, so every archived path, every driver and the ~18 analysis
                    # regexes matching `_lambda([0-9_.]+)_(\d+)\.pdb` are untouched. A multi-seed run
                    # deliberately does NOT match those, so an old script errors instead of silently
                    # mis-joining designs from different seeds.
                    sd = f"_s{run_seed}" if multi_seed else ""
                    name = f"{pid}_{condition}_lambda{_fmt_lambda(lam_label)}{sd}_{idx}.pdb"
                    path = out_dir / name
                    aa = rfd3_out.atom_array
                    # ⚠️ write_pdb may DEFLECT to a hash-suffixed name on a content collision rather
                    # than overwrite, so it returns the path it actually used. Using `path` blindly
                    # would attach the sidecar to a file this run did not write.
                    n_res, path = write_pdb(aa, path)
                    design = Design(prompt_id=pid, condition=condition, lambda_scale=lam_label,
                                    idx=idx, path=path, n_residues=n_res, atom_array=aa)
                    _write_sidecar(path, design, cfg, getattr(rfd3_out, "metadata", None),
                                   seed=run_seed)
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
                print(f"[generate] {condition} λ={_fmt_lambda(lam_label)} seed={run_seed} "
                      f"-> {len(output_list)} design(s)")

    print(f"[generate] wrote {len(designs)} design(s) to {out_dir}")
    return designs
