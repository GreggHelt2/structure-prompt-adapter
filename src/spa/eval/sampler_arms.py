"""The RFdiffusion3 sampler settings, defined once for every driver in this repo.

**Why this module exists.** RFD3 has two published "defaults" and both are real. The released
checkpoint ships ``num_timesteps=100, gamma_0=0.8, step_scale=1.5`` inside its ``train_cfg``; the
shipped ``rfd3`` CLI applies ``200 / 0.6 / 1.5``, which is what the RFdiffusion3 paper used
throughout. ``BaseInferenceEngine`` starts from ``checkpoint["train_cfg"]`` and merges
``inference_sampler_overrides`` on top, so **an unset key leaves the checkpoint's value standing**.
That merge order is how this project ran its entire evaluation program at 100 / 0.8 while its own
config comment claimed 200 (dev ``docs/plan/07_open_questions.md`` I.10, ``docs/results/22`` §1a).

**Why an ARM rather than three independent knobs.** The knobs are not independent in practice: they
name two *published configurations*, and a run that sets the step count but not ``gamma_0`` is
silently a third thing that matches neither paper nor checkpoint. ``scripts/cloud/run_eval.sh``
carried exactly that hazard for months, exposing ``NUM_TIMESTEPS`` and nothing else. Selecting an arm
makes a partial setting unrepresentable.

**Both arms are always fully specified**, including ``step_scale``, which is 1.5 in both. Passing it
explicitly costs nothing numerically and buys provenance: ``spa.eval.generate`` logs
``[sampler] EFFECTIVE ... (requested overrides: ...)`` and an inherited setting logs the empty dict,
so a populated override is what makes a completed run self-documenting about what it actually ran
(``docs/results/22`` §1a).

⚠️ **These values pin what the arms mean; they do not read the checkpoint.** ``ours`` is the released
``rfd3_foundry_2025_12_01`` checkpoint's own sampler, so today the two agree. A future checkpoint
shipping different defaults would diverge from ``ours`` silently. That is deliberate (a pinned
configuration is reproducible where an inherited one is not), but it is the one assumption here worth
re-checking when the host checkpoint changes.
"""

from __future__ import annotations

__all__ = ["ARMS", "ALIASES", "DEFAULT_ARM", "resolve", "resolve_with_legacy",
           "hydra_overrides", "add_arm_argument"]

#: The two published configurations. Keys are the canonical arm names.
ARMS: dict[str, dict[str, float]] = {
    # RFdiffusion3 paper, Fig. S4f: "step scale eta = 1.5, noise level gamma_0 = 0.6, and 200
    # denoising steps (used throughout this work unless otherwise specified)". Also what the shipped
    # `rfd3` CLI applies via configs/inference_engine/rfdiffusion3.yaml.
    "rfd3": {"num_timesteps": 200, "gamma_0": 0.6, "step_scale": 1.5},
    # The released checkpoint's own train_cfg.model.net.inference_sampler, verified by loading
    # rfd3_latest.ckpt. Every SPA result before 2026-07-27 inherited these without stating them.
    "ours": {"num_timesteps": 100, "gamma_0": 0.8, "step_scale": 1.5},
}

#: Historical spellings kept working so existing artifact paths stay valid. `bench_m10_overhead.py`
#: names its output directories `s100_L150` / `s200_L150`, and those paths are cited in
#: dev `docs/results/34`; renaming the keys would orphan them.
ALIASES: dict[str, str] = {"100": "ours", "200": "rfd3", "checkpoint": "ours", "paper": "rfd3"}

#: What a driver uses when the caller says nothing. Numerically identical to inheriting from the
#: checkpoint, but it is *stated* rather than inherited, which is the whole point.
DEFAULT_ARM = "ours"


def resolve(arm: str | None = None) -> dict[str, float]:
    """Return the full ``{num_timesteps, gamma_0, step_scale}`` triple for ``arm``.

    ``None`` selects :data:`DEFAULT_ARM`. Raises ``ValueError`` on an unknown name rather than
    falling back, because a sampler typo that silently runs the wrong configuration is exactly the
    failure this module exists to prevent.
    """
    name = DEFAULT_ARM if arm is None else str(arm)
    name = ALIASES.get(name, name)
    if name not in ARMS:
        raise ValueError(
            f"unknown sampler arm {arm!r}; expected one of "
            f"{sorted(ARMS)} (aliases: {sorted(ALIASES)})"
        )
    return dict(ARMS[name])


def hydra_overrides(arm: str | None = None, prefix: str = "eval.") -> list[str]:
    """Return the arm as Hydra override strings, e.g. ``['eval.num_timesteps=200', ...]``.

    Suitable for splicing into a ``run_flywheel.py`` / ``probe_threeway.py`` command line, both of
    which are ``@hydra.main`` entry points reading the same ``eval`` group that
    :func:`spa.eval.generate.build_eval_engine` consumes.
    """
    return [f"{prefix}{k}={v}" for k, v in resolve(arm).items()]


def add_arm_argument(parser, flag: str = "--sampler-arm"):
    """Attach a ``--sampler-arm`` option to an ``argparse`` parser, with the arms in the help text."""
    parser.add_argument(
        flag, default=DEFAULT_ARM, choices=sorted(ARMS),
        help=(f"RFD3 sampler configuration (default: {DEFAULT_ARM}). "
              f"rfd3 = {ARMS['rfd3']}; ours = {ARMS['ours']}"),
    )
    return parser


def resolve_with_legacy(arm: str | None = None, num_timesteps: int | None = None) -> dict[str, float]:
    """Resolve an arm while honouring the older standalone ``--num-timesteps`` flag.

    Several probe drivers shipped a ``--num-timesteps`` option before ``gamma_0`` was reachable at
    all. Those invocations are recorded in dev ``docs/`` and must keep working, but the flag on its
    own is the hazard this module exists to remove: a run at 200 steps with the checkpoint's
    ``gamma_0=0.8`` matches neither RFdiffusion3's published configuration nor the checkpoint's.

    So: ``None`` (the historical default) defers to the arm, a value that AGREES with the arm is
    accepted, and a value that DISAGREES raises rather than silently producing a third configuration.
    """
    settings = resolve(arm)
    if num_timesteps is None or int(num_timesteps) == int(settings["num_timesteps"]):
        return settings
    raise SystemExit(
        f"--num-timesteps={num_timesteps} disagrees with --sampler-arm={arm or DEFAULT_ARM} "
        f"({settings}). Setting the step count alone leaves gamma_0 at the arm's value, which is a "
        f"configuration matching neither the RFdiffusion3 paper nor the released checkpoint. Select "
        f"an arm instead: {sorted(ARMS)}."
    )
