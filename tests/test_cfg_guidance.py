"""Tests for the SPA classifier-free-guidance hook (dev ``docs/plan/56_cfg_guidance_for_spa.md`` §2.5).

The four tests §2.5 requires are all here, but two of them cannot be written the way §2.5 phrases them,
and the reason is worth stating because it shapes the whole file.

⚠️ "BIT-IDENTICAL" IS NOT AVAILABLE ON THIS HOST. `07_open_questions` I.12 and ``results/27`` established
that RFD3's rollout is nondeterministic on GPU: the nondeterminism lives in the **bf16 autocast** path
and none of the four determinism levers (``use_deterministic_algorithms``, cuDNN determinism, fp32 matmul
``highest``, ``CUBLAS_WORKSPACE_CONFIG``) moves it. Two runs of *unmodified* code with the same seed
already differ. So a full-rollout equality assertion would be testing the GPU, not this module.

The identity claims are therefore split by where each can actually be settled:

* **exactly, on CPU with fakes** (this file): the algebra of the extrapolation at ω = 1, the routing of
  the two passes, restoration of all four context fields, and the stacking guards.
* **structurally, against the real engine** (this file, skipped without ckpt + CUDA): the rollout
  completes through the never-before-executed host CFG branch, the reference pass fires on every step,
  and the context survives the rollout intact.
* **numerically, in the driver** (``scripts/eval/probe_cfg_omega.py``): the armed ω = 1.0 arm against the
  CFG-off arm, read against the run-to-run floor measured on the same fold. That is the honest form of
  the §8 in-run control, and it is a per-fold check across all 15 folds rather than one assertion.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from spa.model.cfg_guidance import (
    CFGPromptSwap,
    arm_cfg,
    disarm_cfg,
    read_cfg_state,
    resolve_sampler,
)
from spa.model.lambda_schedule import ScheduledLambda
from spa.model.wrapper import SPAContext, SPAPromptSlot


# ----------------------------------------------------------------------------------------------------
# fakes: enough of the host to exercise routing without a 2 GB checkpoint
# ----------------------------------------------------------------------------------------------------

class _FakeAdapter:
    """Stands in for SPAAdapter: owns a real SPAContext and a set_null_prompt that marks it."""

    NULL_K = "NULL_K"
    NULL_V = "NULL_V"

    def __init__(self, ctx: SPAContext | None = None):
        self._context = ctx if ctx is not None else SPAContext()
        self.null_calls = []

    @property
    def context(self) -> SPAContext:
        return self._context

    def set_null_prompt(self, batch: int) -> None:
        self.null_calls.append(batch)
        self._context.k = self.NULL_K
        self._context.v = self.NULL_V
        self._context.key_padding_mask = None
        self._context.prompts = None


class _RecordingModule(nn.Module):
    """Records what the SPA context held at the moment it was called."""

    def __init__(self, ctx: SPAContext):
        super().__init__()
        object.__setattr__(self, "_ctx", ctx)
        self.seen = []

    def forward(self, *a, **kw):
        self.seen.append({"k": self._ctx.k, "v": self._ctx.v,
                          "key_padding_mask": self._ctx.key_padding_mask,
                          "prompts": self._ctx.prompts})
        return {"X_L": torch.zeros(1)}


class _FakeInnerSampler:
    def __init__(self):
        self.use_classifier_free_guidance = False
        self.cfg_scale = 2.0
        self.cfg_t_max = None


class _FakeFacade:
    """Mirrors ConditionalDiffusionSampler: a facade that delegates to `.sampler`."""

    def __init__(self):
        self.sampler = _FakeInnerSampler()
        # deliberately shadowed attributes: writing these must NOT be what the code does
        self.cfg_scale = "DECOY"
        self.use_classifier_free_guidance = "DECOY"


class _FakeNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.diffusion_module = module
        self.inference_sampler = _FakeFacade()
        self.use_classifier_free_guidance = False
        self.cfg_features = ["active_donor", "active_acceptor", "ref_atomwise_rasa"]


def _wire(batch: int = 2, omega_schedule=None):
    ctx = SPAContext(k="REAL_K", v="REAL_V", key_padding_mask="REAL_MASK", prompts=None)
    adapter = _FakeAdapter(ctx)
    module = _RecordingModule(ctx)
    net = _FakeNet(module)
    hook = CFGPromptSwap(module, adapter, omega_schedule=omega_schedule, batch=batch)
    original = hook.install(net)
    return ctx, adapter, module, net, hook, original


def _cond(**kw):
    """A conditional-pass call: carries the run's original `f` object."""
    return kw


F_COND = {"is_motif_atom_unindexed": "f_cond"}
F_REF = {"is_motif_atom_unindexed": "f_ref"}   # strip_f returns a FRESH dict (cfg_utils.py:36)


# ----------------------------------------------------------------------------------------------------
# §2.5 test 2 — ROUTING: real prompt on the conditional pass, null on the reference pass
# ----------------------------------------------------------------------------------------------------

def test_routing_conditional_sees_real_prompt_reference_sees_null():
    """The whole mechanism. If both passes saw the same prompt there would be nothing for ω to amplify,
    and if the conditional pass saw the null the design would be unsteered."""
    ctx, adapter, module, net, hook, _ = _wire()

    hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))
    hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))

    assert module.seen[0]["k"] == "REAL_K", "conditional pass must see the real prompt"
    assert module.seen[1]["k"] == _FakeAdapter.NULL_K, "reference pass must see the learned null token"
    assert hook.n_conditional == 1 and hook.n_reference == 1


def test_all_four_context_fields_are_restored_after_the_reference_pass():
    """SPAContext carries `prompts` (the multi-prompt slot list) as a SEPARATE field from k/v. A swap
    that forgot it would silently break exactly the regional and hard-soft configurations most worth
    guiding (plan/56 §2.3)."""
    slots = [SPAPromptSlot(k="K1", v="V1", profile=None, key_padding_mask=None)]
    ctx = SPAContext(k="REAL_K", v="REAL_V", key_padding_mask="REAL_MASK", prompts=slots)
    adapter = _FakeAdapter(ctx)
    module = _RecordingModule(ctx)
    net = _FakeNet(module)
    hook = CFGPromptSwap(module, adapter, batch=2)
    hook.install(net)

    hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))
    hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))

    assert ctx.k == "REAL_K"
    assert ctx.v == "REAL_V"
    assert ctx.key_padding_mask == "REAL_MASK"
    assert ctx.prompts is slots, "the multi-prompt slot list must be restored, not dropped"
    # and the reference pass really did run on the null, not on the slots
    assert module.seen[1]["prompts"] is None


def test_context_is_restored_even_if_the_reference_pass_raises():
    """A rollout that dies mid-reference must not leave SPA stuck on the null token, because drivers
    reuse ONE engine across arms and every later arm would then be silently unsteered."""
    ctx, adapter, module, net, hook, _ = _wire()

    class _Boom(nn.Module):
        def forward(self, *a, **kw):
            raise RuntimeError("boom")

    object.__setattr__(hook, "_module", _Boom())
    with pytest.raises(RuntimeError, match="boom"):
        hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))
    # first call was treated as conditional, so force a real reference call
    hook.reset()
    object.__setattr__(hook, "_cond_f", F_COND)
    with pytest.raises(RuntimeError, match="boom"):
        hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))
    assert ctx.k == "REAL_K" and ctx.v == "REAL_V" and ctx.key_padding_mask == "REAL_MASK"


def test_first_call_of_a_rollout_is_treated_as_conditional():
    """The host calls conditional (:244/:260) before reference (:282) within every step, so the first
    `f` object seen defines the conditional identity for the rest of the rollout."""
    ctx, adapter, module, net, hook, _ = _wire()
    for _ in range(3):
        hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))
        hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))
    assert hook.n_conditional == 3 and hook.n_reference == 3
    assert [s["k"] for s in module.seen] == ["REAL_K", _FakeAdapter.NULL_K] * 3


def test_cfg_t_max_skipping_a_reference_call_does_not_desynchronize():
    """`cfg_t_max` gates guidance off below a noise level (inference_sampler.py:276-278), so some steps
    make only the conditional call. Object identity handles that; a parity counter would not."""
    ctx, adapter, module, net, hook, _ = _wire()
    hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))
    hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))
    hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))     # guidance gated off this step
    hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))
    hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))
    assert hook.n_conditional == 3 and hook.n_reference == 2
    assert [s["k"] for s in module.seen] == [
        "REAL_K", _FakeAdapter.NULL_K, "REAL_K", "REAL_K", _FakeAdapter.NULL_K]


def test_batch_is_inferred_from_x_noisy_when_not_given():
    """D is dim 0 of X_noisy_L in both passes, so the null token's K/V matches the query's batch."""
    ctx = SPAContext(k="REAL_K", v="REAL_V")
    adapter = _FakeAdapter(ctx)
    module = _RecordingModule(ctx)
    net = _FakeNet(module)
    hook = CFGPromptSwap(module, adapter)          # no explicit batch
    hook.install(net)
    hook(f=F_COND, X_noisy_L=torch.zeros(7, 4, 3))
    hook(f=F_REF, X_noisy_L=torch.zeros(7, 4, 3))
    assert adapter.null_calls == [7]


def test_missing_f_kwarg_raises_rather_than_guessing():
    """If a host version stopped passing f= as a keyword, the conditional/reference rule would silently
    treat every call as conditional and ω would guide nothing. Fail loudly instead."""
    ctx, adapter, module, net, hook, _ = _wire()
    with pytest.raises(RuntimeError, match="could not find the `f` kwarg"):
        hook(X_noisy_L=torch.zeros(2, 4, 3))


# ----------------------------------------------------------------------------------------------------
# §2.5 test 1 — IDENTITY at ω = 1.0, settled as algebra rather than as a GPU comparison
# ----------------------------------------------------------------------------------------------------

def _host_extrapolation(delta, delta_ref, cfg_scale):
    """Verbatim inference_sampler.py:307, so the test tracks the host rather than a paraphrase."""
    return delta + (cfg_scale - 1) * (delta - delta_ref)


def test_omega_one_is_an_exact_identity_on_the_coordinate_delta():
    """(1.0 - 1.0) is exactly 0.0 in floating point and 0.0 * finite is exactly 0.0, so ω = 1 leaves
    delta_L untouched bit-for-bit while the full second forward pass still runs. That is what makes it
    a usable in-run control (plan/56 §8) rather than merely a cheap one."""
    delta = torch.randn(4, 32, 3, dtype=torch.float64)
    delta_ref = torch.randn(4, 32, 3, dtype=torch.float64)
    out = _host_extrapolation(delta, delta_ref, 1.0)
    assert torch.equal(out, delta), "ω = 1 must be bit-exact, not merely close"


def test_identical_deltas_make_omega_a_no_op_at_any_scale():
    """The λ = 0 case in algebra: if SPA contributes nothing, the two passes agree and there is nothing
    for ω to amplify at any scale."""
    delta = torch.randn(4, 32, 3, dtype=torch.float64)
    for omega in (1.0, 1.5, 2.0, 3.0, 10.0):
        assert torch.equal(_host_extrapolation(delta, delta.clone(), omega), delta)


def test_extrapolation_coefficients_sum_to_one():
    """The one piece of algebra behind plan/56 §4: delta_new = ω·cond - (ω-1)·ref. A channel identical
    in both passes therefore enters at net weight 1, neither amplified nor removed."""
    cond = torch.randn(4, 16, 3, dtype=torch.float64)
    ref = torch.randn(4, 16, 3, dtype=torch.float64)
    for omega in (1.5, 2.0, 3.0):
        expected = omega * cond - (omega - 1) * ref
        assert torch.allclose(_host_extrapolation(cond, ref, omega), expected, atol=1e-12)


# ----------------------------------------------------------------------------------------------------
# arming: the live-object route, and the two source-level traps it sidesteps
# ----------------------------------------------------------------------------------------------------

def test_arm_cfg_writes_the_inner_sampler_not_the_facade():
    """ConditionalDiffusionSampler delegates to `.sampler`, and line 307 reads `self.cfg_scale` on THAT
    object. Writing the facade's attribute is a silent no-op; the decoy values assert we do not."""
    net = _FakeNet(_RecordingModule(SPAContext()))
    arm_cfg(net, cfg_scale=2.5, cfg_features=[], cfg_t_max=None)
    assert net.inference_sampler.sampler.cfg_scale == 2.5
    assert net.inference_sampler.cfg_scale == "DECOY", "must not write the facade"
    assert resolve_sampler(net) is net.inference_sampler.sampler


def test_arm_cfg_sets_both_flags_so_omega_one_does_not_crash():
    """Via the CONFIG route, cfg_scale=1.0 leaves the net's flag False (RFD3.py:61-63) while the
    sampler's stays True, so the sampler enters its CFG block with f_ref=None and strip_X raises. Arming
    post-construction sets both explicitly, which is what makes ω = 1.0 legitimate here."""
    net = _FakeNet(_RecordingModule(SPAContext()))
    arm_cfg(net, cfg_scale=1.0)
    assert net.use_classifier_free_guidance is True
    assert net.inference_sampler.sampler.use_classifier_free_guidance is True
    assert net.inference_sampler.sampler.cfg_scale == 1.0


def test_arm_cfg_empties_cfg_features_by_default():
    """Configuration (ii) of plan/56 §4.3: ω guides SPA's prompt ONLY. Leaving the host's three
    H-bond/RASA features in place would give configuration (iii), where one scale inflates a sum of two
    effects that cannot be attributed to either."""
    net = _FakeNet(_RecordingModule(SPAContext()))
    assert net.cfg_features                       # the host ships three
    arm_cfg(net)
    assert net.cfg_features == []


def test_disarm_restores_the_prior_state():
    net = _FakeNet(_RecordingModule(SPAContext()))
    before = read_cfg_state(net)
    prior = arm_cfg(net, cfg_scale=3.0)
    disarm_cfg(net, prior)
    assert read_cfg_state(net) == before


def test_read_cfg_state_reports_the_consuming_objects():
    net = _FakeNet(_RecordingModule(SPAContext()))
    arm_cfg(net, cfg_scale=1.5, cfg_features=["ref_atomwise_rasa"], cfg_t_max=80.0)
    st = read_cfg_state(net)
    assert st == {"net_flag": True, "cfg_features": ["ref_atomwise_rasa"], "sampler_flag": True,
                  "cfg_scale": 1.5, "cfg_t_max": 80.0}


# ----------------------------------------------------------------------------------------------------
# the ω schedule (plan/56 §2.3a), and the runtime proof that CFG actually ran
# ----------------------------------------------------------------------------------------------------

def test_omega_schedule_is_applied_on_conditional_calls_only():
    """Line 307 runs after BOTH forwards, so a value written during a step's conditional call takes
    effect for that same step. Writing on the reference call would apply it a step late."""
    seen = []

    def sched(step, t):
        seen.append((step, t))
        return 1.0 + step

    ctx, adapter, module, net, hook, _ = _wire(omega_schedule=sched)
    for _ in range(3):
        hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3), t=torch.tensor([5.0]))
        hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3), t=torch.tensor([5.0]))
    assert [s[0] for s in seen] == [0, 1, 2], "schedule must advance once per SAMPLER STEP, not per call"
    assert net.inference_sampler.sampler.cfg_scale == 3.0
    assert [r["omega"] for r in hook.applied] == [1.0, 2.0, 3.0]


def test_summary_counts_reference_passes_as_the_proof_cfg_ran():
    """n_reference == 0 after a rollout means CFG never ran whatever the config said, and the arm is
    really an ω = 1 arm. This is the check a config readback cannot make."""
    ctx, adapter, module, net, hook, _ = _wire()
    for _ in range(5):
        hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))
    assert hook.summary()["n_reference"] == 0
    hook(f=F_REF, X_noisy_L=torch.zeros(2, 4, 3))
    assert hook.summary()["n_reference"] == 1
    assert hook.summary()["n_conditional"] == 5


# ----------------------------------------------------------------------------------------------------
# the stacking guard: both install orders, since each fails differently and both fail silently
# ----------------------------------------------------------------------------------------------------

def test_refuses_to_install_on_top_of_a_lambda_schedule():
    module = _RecordingModule(SPAContext())
    net = _FakeNet(module)
    ScheduledLambda(module, _FakeAdapter(), [1.0]).install(net)
    with pytest.raises(RuntimeError, match="already installed"):
        CFGPromptSwap(net.diffusion_module, _FakeAdapter()).install(net)


def test_refuses_to_install_on_top_of_another_cfg_hook():
    module = _RecordingModule(SPAContext())
    net = _FakeNet(module)
    CFGPromptSwap(module, _FakeAdapter()).install(net)
    with pytest.raises(RuntimeError, match="already installed"):
        CFGPromptSwap(net.diffusion_module, _FakeAdapter()).install(net)


def test_detects_a_hook_installed_ON_TOP_of_it_at_forward_time():
    """The reverse order, which an install-time check inside this module cannot see. Caught by the slot
    identity check instead, which also catches any future third wrapper."""
    ctx, adapter, module, net, hook, _ = _wire()
    ScheduledLambda(net.diffusion_module, _FakeAdapter(), [1.0]).install(net)   # wraps AROUND the hook
    with pytest.raises(RuntimeError, match="no longer the module in net.diffusion_module"):
        hook(f=F_COND, X_noisy_L=torch.zeros(2, 4, 3))


def test_uninstall_restores_the_original_module():
    ctx, adapter, module, net, hook, original = _wire()
    assert net.diffusion_module is hook
    CFGPromptSwap.uninstall(net, original)
    assert net.diffusion_module is original is module


def test_wrapped_module_and_net_are_not_registered_as_children():
    """Storing _module/_adapter/_net via object.__setattr__ keeps them out of the child list. For _net
    this is not an optimization: normal assignment would make the net its own descendant."""
    inner = nn.Linear(4, 4)
    net = _FakeNet(inner)
    hook = CFGPromptSwap(inner, _FakeAdapter())
    hook.install(net)
    assert all(child is not inner for child in hook.children())
    assert all(child is not net for child in hook.children())
    assert len(list(hook.parameters())) == 0


def test_hook_is_an_nn_module_so_it_can_occupy_the_submodule_slot():
    """`diffusion_module` is a registered submodule; torch's __setattr__ refuses a plain object there."""
    assert isinstance(CFGPromptSwap(nn.Identity(), _FakeAdapter()), nn.Module)


# ----------------------------------------------------------------------------------------------------
# §2.5 test 4 — SMOKE against the real engine, and §2.5 test 3 (λ=0) structurally
#
# Skipped without the real ckpt + CUDA, exactly like tests/test_identity_at_init.py, so the suite stays
# green elsewhere. This is the first time this project executes the host's CFG branch at all
# (plan/56 §2.7), so "it completes and the reference pass fired on every step" is the load-bearing claim
# here; the numerical comparisons live in the driver, against a measured run-to-run floor.
# ----------------------------------------------------------------------------------------------------

CKPT = os.environ.get("SPA_RFD3_CKPT",
                      os.path.expanduser("~/projects/spa/models/rfdiffusion3/rfd3_latest.ckpt"))
_real = pytest.mark.skipif(not (os.path.exists(CKPT) and torch.cuda.is_available()),
                           reason="real RFD3 ckpt and/or CUDA device not available")

SMOKE_LENGTH = int(os.environ.get("SPA_CFG_TEST_LENGTH", "32"))
SMOKE_K = int(os.environ.get("SPA_CFG_TEST_K", "2"))
SMOKE_STEPS = int(os.environ.get("SPA_CFG_TEST_STEPS", "8"))


@pytest.fixture(scope="module")
def real_engine():
    from omegaconf import OmegaConf

    from spa.eval.generate import build_eval_engine, load_adapter
    from spa.train.harness import frozen_rfd3_net

    cfg = OmegaConf.create({
        "paths": {"rfd3_ckpt": CKPT},
        "hardware": {"device": "cuda"},
        "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                  "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
        "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                    "strip_bos_eos": True, "use_clss": False},
        "eval": {"num_designs": SMOKE_K, "length": SMOKE_LENGTH, "specification": None,
                 "num_timesteps": SMOKE_STEPS, "seed": 42, "ckpt": None,
                 "prompt_pdb": None, "prompt_cache": None, "use_sequence": False},
    })
    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, torch.device("cuda"))
    adapter.eval()
    return engine, net, adapter


@_real
def test_smoke_real_sampler_runs_the_host_cfg_branch_end_to_end(real_engine):
    """§2.5's smoke test. The zero-init adapter makes this a λ-neutral rollout, so what is under test is
    the HOST path: f_ref gets built, strip_f/strip_X run, the second forward executes, and line 307
    extrapolates, none of which this project had ever exercised."""
    from spa.eval.generate import _run_once

    engine, net, adapter = real_engine
    prompt = torch.randn(SMOKE_K, SMOKE_LENGTH, 1536,
                         device="cuda", dtype=next(adapter.parameters()).dtype)
    adapter.set_prompt(prompt)
    adapter.set_scale(1.0)

    prior = arm_cfg(net, cfg_scale=2.0, cfg_features=[])
    hook = CFGPromptSwap(net.diffusion_module, adapter, batch=SMOKE_K)
    original = hook.install(net)
    try:
        with torch.no_grad():
            outs = _run_once(engine)
    finally:
        CFGPromptSwap.uninstall(net, original)
        disarm_cfg(net, prior)

    assert len(outs) == SMOKE_K
    s = hook.summary()
    assert s["n_conditional"] > 0
    assert s["n_reference"] == s["n_conditional"], (
        f"CFG reference pass fired {s['n_reference']} times for {s['n_conditional']} sampler steps; "
        "with cfg_t_max=None it must fire on every step or ω is silently not applied everywhere")
    # the prompt must survive the rollout: drivers reuse one engine across arms
    assert adapter.context.k is not None and adapter.context.k.shape[0] == SMOKE_K


@_real
def test_real_sampler_restores_cfg_state_after_disarm(real_engine):
    """A leaked armed flag would turn every later arm in the run into a CFG arm without saying so."""
    engine, net, adapter = real_engine
    before = read_cfg_state(net)
    prior = arm_cfg(net, cfg_scale=2.0, cfg_features=[])
    assert read_cfg_state(net)["net_flag"] is True
    disarm_cfg(net, prior)
    assert read_cfg_state(net) == before
