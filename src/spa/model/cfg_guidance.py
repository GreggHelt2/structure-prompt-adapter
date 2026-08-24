"""Classifier-free guidance on SPA's prompt channel: the ω knob, as a sampler-time hook.

Spec: dev ``docs/plan/56_cfg_guidance_for_spa.md`` §2.3 (the wrapper), §2.3a (the per-step ω path),
§4 (what ω actually guides). Phase-0's measurement of the quantity ω multiplies is
``docs/results/27_cfg_phase0_probe.md``.

WHAT THIS IS
RFdiffusion3 already implements classifier-free guidance and ships it **off**
(``inference_sampler.py:276-307``). Per sampler step it runs a second, "reference" forward pass and
extrapolates on the **coordinate delta**::

    delta_L = delta_L + (cfg_scale - 1) * (delta_L - delta_L_ref)

The host builds its reference pass by stripping features named in ``cfg_features`` out of ``f``. It
does **not** touch :class:`~spa.model.wrapper.SPAContext`, so turning the host's flag on with nothing
else guides RFD3's own H-bond/RASA channels while SPA rides along at unit weight (``plan/56`` §4.5).
This module supplies the missing piece: during the reference pass it swaps SPA onto its learned null
token, so the two passes differ **only** by the prompt and ω amplifies SPA's prompt channel.

⭐ THE COEFFICIENTS SUM TO 1, AND THAT IS THE WHOLE SEMANTICS
``delta_new = ω·delta_cond - (ω-1)·delta_ref``. Anything identical in both passes enters at net weight
1: not amplified, and **not removed either**. Only what *differs* is multiplied by ``(ω-1)``. So with
``cfg_features=[]`` and this wrapper installed, every native conditioning channel (a pinned motif,
hotspots, ``is_non_loopy``, partial sequence) stays **fully active at its designed strength** and only
SPA's prompt is inflated. "Not amplified" never means "not active" (``plan/56`` §4.2).

⚠️ BUT THE EXTRAPOLATION HAPPENS AFTER BOTH PASSES, ON COORDINATES
A constraint perfectly respected *inside* each forward can still be disturbed *by* the extrapolation,
because ``delta_L`` is applied to every atom including pinned motif atoms and the sampler path we take
never overwrites them with ground truth (``allow_realignment: False``). Any hard ⊕ soft run with ω > 1
must therefore report ``motif_rmsd`` alongside its other numbers (``plan/56`` §6.2, §8). Phase A is
soft-only and places no motif, so the question does not arise there; it is step 2 of §6.3, not step 1.

TWO SOURCE-LEVEL TRAPS, BOTH VERIFIED 2026-08-23 AND BOTH CORRECTING THE SPEC
1. **``cfg_scale = 1.0`` via the CONFIG route does not silently no-op, it CRASHES.** ``RFD3.__init__``
   sets its own flag to ``use_classifier_free_guidance and cfg_scale != 1.0`` (``RFD3.py:61-63``) and so
   skips building ``f_ref``, but it hands the **raw** dict to ``ConditionalDiffusionSampler``, whose
   flag stays ``True``. The sampler then enters its CFG block with ``f_ref=None`` and ``strip_X``
   raises ``TypeError``. ``plan/56`` §2.4(1) reads this as a free identity test; it is not.
   :func:`arm_cfg` sidesteps it entirely by setting **both** flags explicitly on the live objects
   after construction, which is why ω = 1.0 is a legitimate, exactly-identity setting here.
2. **``cfg_scale`` lives one attribute deeper than it looks.** ``net.inference_sampler`` is a
   ``ConditionalDiffusionSampler`` that *delegates* to ``self.sampler``, and line 307 reads
   ``self.cfg_scale`` on that inner object. Writing ``net.inference_sampler.cfg_scale`` silently does
   nothing. :func:`resolve_sampler` is the single place that knows this.

ISOLATION
Nothing here runs unless a caller explicitly installs it, and nothing in the existing inference path
imports this module. With CFG unrequested, ``build_eval_engine`` never adds the CFG keys to its
override dict, the checkpoint's ``use_classifier_free_guidance: False`` stands, and the host's CFG
branch is unreachable. This module is additive: it modifies no existing file.
"""

from __future__ import annotations

import torch.nn as nn

from .lambda_schedule import ScheduledLambda

__all__ = ["CFGPromptSwap", "arm_cfg", "disarm_cfg", "resolve_sampler", "read_cfg_state"]


# ------------------------------------------------------------------------------------------------
# Arming the host's CFG on the LIVE objects (no config plumbing, no edits to generate.py)
# ------------------------------------------------------------------------------------------------


def resolve_sampler(net):
    """The object whose ``cfg_scale`` line 307 actually reads.

    ``net.inference_sampler`` is a ``ConditionalDiffusionSampler`` facade that forwards
    ``sample_diffusion_like_af3`` to ``self.sampler`` (``inference_sampler.py:566-595``); the sampling
    loop's ``self`` is that inner object. Falls back to the facade itself if a future host drops the
    indirection, so a version change degrades to "writes the only sampler there is" rather than to a
    silent no-op.
    """
    facade = net.inference_sampler
    return getattr(facade, "sampler", facade)


def arm_cfg(net, *, cfg_scale: float = 2.0, cfg_features=(), cfg_t_max=None) -> dict:
    """Turn the host's CFG on **after** ``engine.initialize()``, and return what was set.

    Every flag the host consults is a plain attribute read at forward time, so CFG can be armed on a
    built engine without touching config plumbing:

    ==================================================  ====================================
    attribute                                           read at
    ==================================================  ====================================
    ``net.use_classifier_free_guidance``                ``RFD3.py:88`` (gates building ``f_ref``)
    ``net.cfg_features``                                ``RFD3.py:89`` (passed to ``strip_f``)
    ``sampler.use_classifier_free_guidance``            ``inference_sampler.py:276``
    ``sampler.cfg_scale``                               ``inference_sampler.py:307``
    ``sampler.cfg_t_max``                               ``inference_sampler.py:277``
    ==================================================  ====================================

    Both flags are set **explicitly**, which is what makes ``cfg_scale=1.0`` safe here (see this
    module's docstring, trap 1) rather than the ``TypeError`` the config route produces.

    Args:
        net: the live RFD3 module (``spa.train.harness.frozen_rfd3_net(engine)``).
        cfg_scale: ω. ``1.0`` is an exact no-op at line 307 while still running the full second
            forward pass, which is precisely the in-run identity control ``plan/56`` §8 requires.
        cfg_features: native per-atom features to ALSO strip in the reference pass. Default ``()``
            is configuration (ii) of ``plan/56`` §4.3: ω guides SPA's prompt **only**. Passing the
            host's default list instead gives configuration (iii), where one scale inflates a sum of
            two effects that cannot then be attributed to either. Do not do that first.
        cfg_t_max: guidance applies only while ``c_t > cfg_t_max``; ``None`` = every step. Note
            ``results/27`` measured the prompt's real influence as early-and-mid, and that the late
            ratio rise is a collapsing-denominator artifact, so a late-weighted gate is the wrong
            lesson to draw from that column.

    Returns:
        The prior values, suitable for :func:`disarm_cfg`.
    """
    sampler = resolve_sampler(net)
    prior = read_cfg_state(net)
    net.use_classifier_free_guidance = True
    net.cfg_features = list(cfg_features)
    sampler.use_classifier_free_guidance = True
    sampler.cfg_scale = float(cfg_scale)
    sampler.cfg_t_max = cfg_t_max
    return prior


def disarm_cfg(net, prior: dict | None = None) -> None:
    """Restore CFG to ``prior`` (from :func:`arm_cfg`), or hard-off if ``prior`` is None."""
    sampler = resolve_sampler(net)
    if prior is None:
        prior = {"net_flag": False, "cfg_features": [], "sampler_flag": False,
                 "cfg_scale": 2.0, "cfg_t_max": None}
    net.use_classifier_free_guidance = prior["net_flag"]
    net.cfg_features = prior["cfg_features"]
    sampler.use_classifier_free_guidance = prior["sampler_flag"]
    sampler.cfg_scale = prior["cfg_scale"]
    sampler.cfg_t_max = prior["cfg_t_max"]


def read_cfg_state(net) -> dict:
    """Read the five live CFG attributes back off the objects that actually consume them.

    This is the substitute for ``generate._assert_sampler_effective``, which verifies against the
    merged *config* and therefore cannot see a post-construction arming at all. A driver should call
    this after :func:`arm_cfg` and record it in the run's provenance. It is necessary but **not
    sufficient**: the only proof CFG really ran is :attr:`CFGPromptSwap.n_reference`, which counts
    reference passes the wrapper actually observed.
    """
    sampler = resolve_sampler(net)
    return {
        "net_flag": getattr(net, "use_classifier_free_guidance", None),
        "cfg_features": list(getattr(net, "cfg_features", []) or []),
        "sampler_flag": getattr(sampler, "use_classifier_free_guidance", None),
        "cfg_scale": getattr(sampler, "cfg_scale", None),
        "cfg_t_max": getattr(sampler, "cfg_t_max", None),
    }


# ------------------------------------------------------------------------------------------------
# The wrapper: swap SPA onto its learned null token for the reference pass
# ------------------------------------------------------------------------------------------------


class CFGPromptSwap(nn.Module):
    """Wraps ``net.diffusion_module`` and runs the CFG reference pass on SPA's learned null prompt.

    ⭐ TELLING THE TWO CALLS APART IS EXACT, NOT HEURISTIC. Per sampler step the host makes exactly two
    ``diffusion_module`` calls: the conditional one receives the original ``f`` dict, hoisted outside
    the loop, and the reference one receives ``f_ref``, built **once** at ``RFD3.py:90``. Both are
    stable objects for the whole rollout, and ``strip_f`` returns a fresh dict (``cfg_utils.py:36``)
    even when numerically identical. So "the first ``f`` object seen is the conditional one, any other
    object is the reference" is sufficient, needs no tensor comparison, and cannot drift. The
    conditional call always precedes the reference call within a step (``:244``/``:260`` before
    ``:282``), and ``cfg_t_max`` can skip a reference call entirely, which object identity handles and
    a parity counter would not.

    ⭐ ALL FOUR CONTEXT FIELDS ARE SAVED AND RESTORED. ``SPAContext`` carries ``k``, ``v``,
    ``key_padding_mask`` **and** ``prompts``, the multi-prompt slot list. A swap that forgot ``prompts``
    would silently break exactly the regional and hard ⊕ soft configurations most worth guiding
    (``plan/56`` §2.3).

    ⛔ HOOKS DO NOT STACK, AND BOTH ORDERS ARE CAUGHT. This class and
    :class:`~spa.model.lambda_schedule.ScheduledLambda` compete for the single ``net.diffusion_module``
    slot, and each advances its own per-call state. Since CFG makes **two** calls per sampler step, a
    stacked λ schedule would advance twice per step and the reference pass would run at the *next*
    step's λ, so the two passes would differ by prompt **and** by λ and ω would amplify the mixture.
    Silent, and wrong. Guards:

    * installed **second**: :meth:`install` refuses when the slot already holds a known hook;
    * installed **first**, with something wrapped around it afterwards: :meth:`forward` refuses when
      ``net.diffusion_module is not self``, which also catches any future third wrapper.

    It must subclass ``nn.Module`` because ``diffusion_module`` is a registered submodule and torch's
    ``__setattr__`` refuses a plain object there. The wrapped module, the adapter and the net are
    stored via ``object.__setattr__`` so they are not re-registered as children (assigning ``net``
    normally would make the net its own descendant).

    Args:
        module: the module currently in the ``net.diffusion_module`` slot.
        adapter: the :class:`~spa.model.wrapper.SPAAdapter` whose context is swapped.
        omega_schedule: optional ``(step_index, t) -> float`` applied on each **conditional** call,
            giving the arbitrary ω(t) curve of ``plan/56`` §2.3a. ``None`` = constant ω, whatever
            :func:`arm_cfg` set. Ordering works because line 307 runs after both forwards, so a value
            written during a step's conditional call takes effect for that same step.
        batch: diffusion batch ``D`` for the null token's K/V. ``None`` = infer from ``X_noisy_L``,
            whose dim 0 is ``D`` in both passes.
    """

    def __init__(self, module, adapter, *, omega_schedule=None, batch: int | None = None) -> None:
        super().__init__()
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_net", None)
        self._omega_schedule = omega_schedule
        self._batch = batch
        self.reset()

    def reset(self) -> None:
        """Clear per-rollout state. Call between arms if a wrapper instance is reused."""
        object.__setattr__(self, "_cond_f", None)
        self.n_conditional = 0
        self.n_reference = 0
        self.applied: list[dict] = []

    # ------------------------------------------------------------------------------------------
    def _save_context(self) -> dict:
        ctx = self._adapter.context
        return {"k": ctx.k, "v": ctx.v, "key_padding_mask": ctx.key_padding_mask,
                "prompts": ctx.prompts}

    def _restore_context(self, saved: dict) -> None:
        ctx = self._adapter.context
        ctx.k = saved["k"]
        ctx.v = saved["v"]
        ctx.key_padding_mask = saved["key_padding_mask"]
        ctx.prompts = saved["prompts"]

    def _set_omega(self, value: float) -> None:
        resolve_sampler(self._net).cfg_scale = float(value)

    # ------------------------------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        net = self._net
        if net is not None and net.diffusion_module is not self:
            raise RuntimeError(
                "CFGPromptSwap is no longer the module in net.diffusion_module: another hook was "
                "installed on top of it. λ-schedule and ω hooks cannot be stacked, because CFG makes "
                "two diffusion_module calls per sampler step and a per-call counter would advance "
                "twice, running the reference pass at the wrong λ (dev plan/56 §2.3a). Install one "
                "hook at a time."
            )
        if "f" not in kwargs:
            raise RuntimeError(
                "CFGPromptSwap could not find the `f` kwarg on a diffusion_module call, so it cannot "
                "tell the conditional pass from the CFG reference pass. The host passes f= as a "
                "keyword at inference_sampler.py:244/:260 (conditional) and :282 (reference); a host "
                "version that changed this needs the detection rule in this class revisited."
            )
        f = kwargs["f"]
        if self._cond_f is None:
            # The first call of a rollout is always the conditional one (:244/:260 precede :282).
            object.__setattr__(self, "_cond_f", f)

        if f is self._cond_f:
            t = kwargs.get("t")
            t_val = float(t.flatten()[0].item()) if t is not None else None
            if self._omega_schedule is not None:
                self._set_omega(self._omega_schedule(self.n_conditional, t_val))
            self.applied.append({
                "call": self.n_conditional,
                "t": t_val,
                "omega": float(resolve_sampler(net).cfg_scale) if net is not None else None,
            })
            self.n_conditional += 1
            return self._module(*args, **kwargs)

        # --- the CFG reference pass: SPA live on its learned null token, real prompt restored after ---
        saved = self._save_context()
        batch = self._batch
        if batch is None:
            X = kwargs.get("X_noisy_L")
            if X is None and args:
                X = args[0]
            if X is None:
                raise RuntimeError(
                    "CFGPromptSwap needs the diffusion batch D to build the null token's K/V and "
                    "could not infer it from X_noisy_L. Pass batch=K explicitly."
                )
            batch = int(X.shape[0])
        try:
            self._adapter.set_null_prompt(batch)
            self.n_reference += 1
            return self._module(*args, **kwargs)
        finally:
            self._restore_context(saved)

    # ------------------------------------------------------------------------------------------
    def install(self, net):
        """Put this wrapper in ``net.diffusion_module``. Returns the original for restoration."""
        current = net.diffusion_module
        if isinstance(current, (CFGPromptSwap, ScheduledLambda)):
            raise RuntimeError(
                f"{type(current).__name__} is already installed on net.diffusion_module. Hooks on "
                "this slot cannot be stacked (dev plan/56 §2.3a): CFG makes two diffusion_module "
                "calls per sampler step, so a stacked per-call counter advances twice and the "
                "reference pass would run at the wrong λ. Uninstall the other hook first."
            )
        object.__setattr__(self, "_net", net)
        net.diffusion_module = self
        return current

    @staticmethod
    def uninstall(net, original) -> None:
        net.diffusion_module = original

    # ------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        """Per-rollout counts, for the run record.

        ``n_reference == 0`` after a rollout means **CFG never ran**, whatever the config said, and
        any ω result from that arm is really an ω = 1 result. Drivers should assert on it rather than
        trusting a config readback (see :func:`read_cfg_state`).
        """
        omegas = [r["omega"] for r in self.applied if r["omega"] is not None]
        return {
            "n_conditional": self.n_conditional,
            "n_reference": self.n_reference,
            "omega_min": min(omegas) if omegas else None,
            "omega_max": max(omegas) if omegas else None,
            "omega_mean": (sum(omegas) / len(omegas)) if omegas else None,
        }
