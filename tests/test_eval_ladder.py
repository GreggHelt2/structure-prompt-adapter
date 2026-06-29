"""Checkpoint-step adherence ladder tests (dev 13_longer_training_decision.md).

The real ladder needs a GPU + the RFD3/ESM3 stack, so these tests pin the parts SPA owns without
running any of it: the snapshot-filename → step parsing, the distribution helper, and — with the
heavy pieces (engine/adapter/prompt/generate/adherence) mocked — that `run_ladder` does baseline
once, sweeps each checkpoint at each λ, computes dTM vs the shared baseline, and writes the table.

NB: monkeypatch targets the MODULE OBJECTS, not dotted strings — `spa.eval.generate` the submodule is
shadowed in the package namespace by the `generate` function (`from .generate import generate`), so a
string path "spa.eval.generate.X" would resolve through the function and fail.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from spa.eval.ladder import _dist, _parse_step, run_ladder

# Module OBJECTS (via sys.modules) — `import spa.eval.generate as x` would bind x to the shadowing
# `generate` function in the package namespace, not the submodule. import_module returns the module.
gen_mod = importlib.import_module("spa.eval.generate")
score_mod = importlib.import_module("spa.eval.score")
harness_mod = importlib.import_module("spa.train.harness")
device_mod = importlib.import_module("spa.utils.device")


def test_parse_step():
    assert _parse_step("/tmp/spa_C_step1000.pt") == (1000, "step1000")
    assert _parse_step("/x/spa_C_runA_step20000.pt") == (20000, "step20000")
    assert _parse_step("/x/spa_C_runA_final.pt") == (30000, "final")  # adapter export == last step
    assert _parse_step("/x/weird_name.pt")[0] == 0


def test_dist():
    m, sd, mx = _dist([1.0, 2.0, 3.0])
    assert (round(m, 3), round(mx, 3)) == (2.0, 3.0) and sd > 0
    assert _dist([None, float("nan")]) == (None, None, None)
    assert _dist([5.0]) == (5.0, 0.0, 5.0)  # single value -> sd 0


def _install(monkeypatch, holder, tm_for):
    """Mock run_ladder's heavy deps on their module objects. `tm_for(cond, lam, ckpt)` -> adherence TM."""
    monkeypatch.setattr(device_mod, "resolve_device", lambda *_a, **_k: "cpu")
    monkeypatch.setattr(gen_mod, "build_eval_engine", lambda cfg: object())
    monkeypatch.setattr(harness_mod, "frozen_rfd3_net", lambda engine: object())
    monkeypatch.setattr(gen_mod, "load_adapter", lambda net, cfg, device: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(harness_mod, "load_spa", lambda adapter, path: holder.update(ckpt=path))
    monkeypatch.setattr(gen_mod, "resolve_prompt", lambda cfg, device: torch.zeros(4, 8))

    def fake_generate(cfg, *, engine=None, adapter=None):
        out = []
        for cond in list(cfg.eval.conditions):
            lams = [0.0] if cond == "baseline" else [float(x) for x in cfg.eval.lambda_scale]
            for lam in lams:
                for idx in range(int(cfg.eval.num_designs)):
                    out.append(SimpleNamespace(condition=cond, lambda_scale=lam, idx=idx,
                                               _tm=tm_for(cond, lam, holder.get("ckpt"))))
        return out
    monkeypatch.setattr(gen_mod, "generate", fake_generate)
    monkeypatch.setattr(score_mod, "adherence",
                        lambda d, p, *, tm_norm="prompt": SimpleNamespace(tm_score=d._tm, prompt_rmsd=10.0 - d._tm))


def test_run_ladder_baseline_once_sweep_and_dtm(tmp_path, monkeypatch):
    holder: dict = {}

    def tm_for(cond, lam, ckpt):
        if cond == "baseline":
            return 0.20
        step, _ = _parse_step(ckpt)        # upslope: adherence grows with training step
        return 0.00003 * step + 0.05 * lam
    _install(monkeypatch, holder, tm_for)

    cfg = OmegaConf.create({
        "eval": {"prompt_pdb": str(tmp_path / "p.pdb"), "num_designs": 4, "seed": 0,
                 "lambda_scale": [1.0, 2.0],
                 "ladder": [str(tmp_path / "spa_C_step1000.pt"), str(tmp_path / "spa_C_step10000.pt")],
                 "out_dir": str(tmp_path / "out")},
        "hardware": {"device": "cpu"},
    })
    OmegaConf.set_struct(cfg, False)
    pts = run_ladder(cfg)["points"]

    baseline = [p for p in pts if p.condition == "baseline"]
    assert len(baseline) == 1 and baseline[0].n == 4 and baseline[0].d_tm is None  # baseline ONCE
    spa = [p for p in pts if p.condition == "spa"]
    assert {(p.step, p.lambda_scale) for p in spa} == {(1000, 1.0), (1000, 2.0), (10000, 1.0), (10000, 2.0)}
    tm_1k = next(p for p in spa if p.step == 1000 and p.lambda_scale == 1.0)
    tm_10k = next(p for p in spa if p.step == 10000 and p.lambda_scale == 1.0)
    assert tm_10k.tm_mean > tm_1k.tm_mean                                  # upslope detected
    assert tm_1k.d_tm == pytest.approx(tm_1k.tm_mean - 0.20, abs=1e-6)     # dTM vs the SHARED baseline
    assert json.loads((tmp_path / "out" / "ladder_results.json").read_text())["points"]


def test_run_ladder_requires_ladder(tmp_path):
    cfg = OmegaConf.create({"eval": {"prompt_pdb": "p.pdb", "ladder": []}, "hardware": {"device": "cpu"}})
    OmegaConf.set_struct(cfg, False)
    with pytest.raises(ValueError, match="eval.ladder"):
        run_ladder(cfg)
