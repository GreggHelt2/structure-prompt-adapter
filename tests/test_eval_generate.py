"""Stage-1 design-generation smoke test (dev 05_validation_pipeline.md Stage 0), against real RFD3.

Exercises ``spa.eval.generate`` end-to-end on the real RFD3 sampler and verifies the
wrapped-no-prompt identity gate. Skipped unless the real RFD3 ckpt and a CUDA device are present,
so the suite stays green elsewhere (gated exactly like ``test_harness.py``).

**Why the identity gate is checked at the forward level, not by comparing two rollouts.** The
wrapped-no-prompt == vanilla-RFD3 invariant is exact and lives in the *network forward*: when no
prompt is set, ``SPAWrappedAttention.forward`` returns ``self.orig(...)`` verbatim (zero added), so
every generation step is bit-for-bit vanilla. RFD3's full *rollout*, however, is **not**
reproducible run-to-run on GPU: its sampler noise IS reproducible (verified — ``torch.normal`` draws
match under a fixed seed), but nondeterministic CUDA kernels in the network forward amplify over the
diffusion trajectory (measured ~0.5–0.8 Å Cα divergence between two identically-seeded rollouts, with
different sequence-head outputs) — a property of RFD3, independent of SPA, and unfixable via
``torch.use_deterministic_algorithms`` (some ops have no deterministic impl). So comparing two
independent rollouts would test RFD3's (lack of) reproducibility, not SPA's identity. We therefore
assert the invariant where it is exact: on a REAL generation forward input captured from
``engine.run``, the SPA-attached net (prompt cleared, and — via zero-init ``Wo`` — even WITH a prompt)
reproduces the base RFD3 forward bit-for-bit (the same technique as ``test_identity_at_init.py``,
here on the generation host the engine actually samples with). The end-to-end ``generate`` run then
confirms the SPA path produces valid PDBs of the right size.

FAST by construction: short design (length 16), K=2, few sampler steps, one shared engine, untrained
zero-init adapter, and a precomputed (random) prompt tensor so ESM3 need not load.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

CKPT = os.environ.get("SPA_RFD3_CKPT", os.path.expanduser("~/projects/spa/models/rfdiffusion3/rfd3_latest.ckpt"))
LENGTH = int(os.environ.get("SPA_EVAL_TEST_LENGTH", "16"))
TIMESTEPS = int(os.environ.get("SPA_EVAL_TEST_TIMESTEPS", "10"))
K = 2
SEED = 0

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CKPT) and torch.cuda.is_available()),
    reason="real RFD3 ckpt and/or CUDA device not available",
)


def _cfg(out_dir, prompt_cache):
    return OmegaConf.create(
        {
            "paths": {"rfd3_ckpt": CKPT},
            "hardware": {"device": "cuda:0"},
            "model": {"c_query": 768, "c_kv": 1536, "c_model": 768, "n_head": 8, "shared_kv": True,
                      "zero_init_output": True, "lambda_init": 1.0, "input_rmsnorm": True},
            "variant": {"name": "C", "projector": "identity", "resampler_tokens": None,
                        "strip_bos_eos": True, "use_clss": False},
            "eval": {
                "ckpt": None,                       # untrained zero-init adapter (identity)
                "conditions": ["baseline", "spa"],
                "lambda_scale": 1.0,
                "num_designs": K,
                "length": LENGTH,
                "specification": None,
                "num_timesteps": TIMESTEPS,
                "seed": SEED,
                "prompt_pdb": None,
                "prompt_cache": str(prompt_cache),  # random [N,1536] -> no ESM3 load (fast)
                "use_sequence": False,
                "prompt_id": "smoke",
                "out_dir": str(out_dir),
            },
        }
    )


def _clone_any(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _clone_any(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_any(v) for v in x)
    return x


class _Stop(Exception):
    """Abort the rollout once the first real token-transformer input is captured."""


def _reset(adapter):
    for ca in adapter.cross_attn:
        torch.nn.init.zeros_(ca.to_out.weight)
        torch.nn.init.zeros_(ca.to_out.bias)
        ca.set_scale(1.0)
    adapter.clear_prompt()


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """Build ONE engine + attach the adapter; capture a real generation forward input; then run the
    full generate() end-to-end on the SAME engine."""
    from spa.eval.generate import build_eval_engine, generate, load_adapter
    from spa.train.harness import frozen_rfd3_net

    tmp = tmp_path_factory.mktemp("spa_eval")
    out_dir = tmp / "designs"
    prompt_cache = tmp / "prompt.pt"
    torch.save(torch.randn(12, 1536), prompt_cache)   # stand-in for an ESM3 [N,1536] cache entry
    cfg = _cfg(out_dir, prompt_cache)
    device = torch.device("cuda:0")

    engine = build_eval_engine(cfg)
    net = frozen_rfd3_net(engine)
    adapter = load_adapter(net, cfg, device)

    # Capture a REAL token-transformer input the engine feeds during a generation rollout.
    dt = net.diffusion_module.diffusion_transformer
    box = {}

    def hook(_m, args, kwargs):
        box["a"], box["kw"] = _clone_any(args), _clone_any(kwargs)
        raise _Stop

    handle = dt.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        engine.run(inputs=None, out_dir=None)
    except _Stop:
        pass
    finally:
        handle.remove()
    a, kw = box["a"], box["kw"]

    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(0)
            return dt(*a, **kw)

    _reset(adapter)
    out_clear = fwd()
    deterministic = torch.equal(out_clear, fwd())

    D, L = a[0].shape[0], a[0].shape[1]
    adapter.set_prompt(torch.randn(D, L + 2, cfg.model.c_kv, device=device))  # zero-init Wo => identity
    out_prompt = fwd()

    for ca in adapter.cross_attn:                       # grow Wo -> SPA path becomes live (sanity)
        torch.nn.init.xavier_uniform_(ca.to_out.weight)
    out_grown_changes = not torch.equal(out_clear, fwd())
    _reset(adapter)                                     # restore identity-at-init for generate()

    # End-to-end: reuse the SAME engine + adapter to roll out + write PDBs.
    designs = generate(cfg, engine=engine, adapter=adapter)

    return SimpleNamespace(
        out_clear=out_clear, deterministic=deterministic, out_prompt=out_prompt,
        out_grown_changes=out_grown_changes, designs=designs, out_dir=out_dir,
    )


def _by(designs, condition):
    return sorted([d for d in designs if d.condition == condition], key=lambda d: d.idx)


# ---- Identity gate (forward-level, deterministic) -------------------------------------------------

def test_generation_forward_is_deterministic(shared):
    assert shared.deterministic                         # precondition for the bit-for-bit gate


def test_identity_gate_baseline_equals_vanilla(shared):
    # Prompt cleared (baseline) AND, via zero-init Wo, even WITH a prompt set: the SPA-attached
    # generation forward is bit-for-bit the base RFD3 forward -> baseline generation == vanilla RFD3.
    assert torch.equal(shared.out_clear, shared.out_prompt)


def test_grown_adapter_changes_generation_forward(shared):
    assert shared.out_grown_changes                     # once Wo is non-zero, SPA actually perturbs


# ---- End-to-end generation (the spa.eval.generate API) -------------------------------------------

def test_counts_and_files(shared):
    base, spa = _by(shared.designs, "baseline"), _by(shared.designs, "spa")
    assert len(base) == K and len(spa) == K             # K designs per condition
    for d in base + spa:
        assert d.path.exists() and d.path.suffix == ".pdb"


def test_spa_runs_and_residue_counts(shared):
    # baseline + spa (λ=1, with prompt) both produced valid PDBs of the expected length.
    from biotite.structure import get_residue_count
    from biotite.structure.io.pdb import PDBFile

    for d in _by(shared.designs, "baseline") + _by(shared.designs, "spa"):
        assert d.n_residues == LENGTH
        parsed = PDBFile.read(str(d.path)).get_structure(model=1)
        assert get_residue_count(parsed) == LENGTH
