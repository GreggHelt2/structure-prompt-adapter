"""Stage-3 OF3 refolder tests (dev 05 Stage 3 / 07 F1.5.4).

OpenFold3 needs a GPU + ~30 s/fold, so these tests do NOT run it: they monkeypatch the subprocess to
validate the parts SPA owns — the multi-query JSON schema, the `run_openfold` command (flags + env +
conda wrapping), and the output-path reconstruction — exactly the surfaces that break silently if the
OF3 CLI/layout drifts. The real OF3 invocation is a manual smoke run (dev 05 verified 2.2 GB @76 res).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spa.eval.openfold3 import OF3Refolder
from spa.eval.score import Refolder


def _refolder(out_dir, **kw) -> OF3Refolder:
    params = dict(
        ckpt_path="/models/of3.pt",
        runner_yaml="/cfg/of3_nokernel.yml",
        out_dir=out_dir,
        conda_env="spa-verify-of3",
        num_diffusion_samples=1,
        seed=42,
    )
    params.update(kw)  # let callers override any default (e.g. conda_env=None)
    return OF3Refolder(**params)


def _seqset(name="design_x", sequences=("AAAA", "CCCC", "DDDD")):
    return SimpleNamespace(name=name, sequences=list(sequences))


def test_satisfies_refolder_protocol(tmp_path):
    """OF3Refolder must duck-type the score.Refolder injection point (runtime_checkable)."""
    assert isinstance(_refolder(tmp_path), Refolder)


def test_query_json_schema(tmp_path):
    """One single-chain protein query per sequence, keyed q{i}, chain sep stripped (dev 05 schema)."""
    qj = _refolder(tmp_path)._build_query_json(["AAAA", "BB/BB"])
    assert set(qj["queries"]) == {"q0", "q1"}
    chain = qj["queries"]["q0"]["chains"][0]
    assert chain == {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "AAAA"}
    assert qj["queries"]["q1"]["chains"][0]["sequence"] == "BBBB"  # '/' multi-chain sep removed


def test_command_has_required_flags(tmp_path):
    """The run_openfold argv carries MSA-free + ckpt + runner-yaml + 1 sample + conda wrapping."""
    cmd = _refolder(tmp_path)._build_command(Path("/q/queries.json"), Path("/out"))
    assert cmd[:4] == ["conda", "run", "-n", "spa-verify-of3"]
    assert cmd[4:6] == ["run_openfold", "predict"]
    assert "--use-msa-server=False" in cmd
    assert cmd[cmd.index("--inference-ckpt-path") + 1] == "/models/of3.pt"
    assert cmd[cmd.index("--runner-yaml") + 1] == "/cfg/of3_nokernel.yml"
    assert cmd[cmd.index("--num-diffusion-samples") + 1] == "1"
    assert cmd[cmd.index("--query-json") + 1] == "/q/queries.json"


def test_command_no_conda_when_env_none(tmp_path):
    cmd = _refolder(tmp_path, conda_env=None)._build_command(Path("/q.json"), Path("/out"))
    assert cmd[0] == "run_openfold"


def _fake_run_factory(monkeypatch, *, emit=True):
    """Patch subprocess.run to mimic run_openfold: read the query JSON, write one cif per query at the
    exact OF3 output path (writer.py layout), return success. `emit=False` skips writing (fold failed)."""
    def fake_run(cmd, capture_output=True, text=True, env=None):
        qjson = Path(cmd[cmd.index("--query-json") + 1])
        out = Path(cmd[cmd.index("--output-dir") + 1])
        seed = 42
        if emit:
            for qid in json.loads(qjson.read_text())["queries"]:
                d = out / qid / f"seed_{seed}"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{qid}_seed_{seed}_sample_1_model.cif").write_text("data_fake\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr("spa.eval.openfold3.subprocess.run", fake_run)


def test_refold_returns_one_cif_per_sequence(tmp_path, monkeypatch):
    _fake_run_factory(monkeypatch, emit=True)
    refolds = _refolder(tmp_path).refold(_seqset(sequences=["AAAA", "CCCC", "DDDD"]))
    assert len(refolds) == 3
    for p in refolds:
        assert Path(p).exists() and Path(p).name.endswith("_seed_42_sample_1_model.cif")
    # query JSON was written for the run
    assert (tmp_path / "of3" / "design_x" / "queries.json").exists()


def test_refold_drops_missing_cifs(tmp_path, monkeypatch):
    """A run that exits 0 but emits no structure -> empty list (best-of-K just has no candidates)."""
    _fake_run_factory(monkeypatch, emit=False)
    assert _refolder(tmp_path).refold(_seqset()) == []


def test_refold_empty_sequence_set_skips(tmp_path, monkeypatch):
    """No sequences -> no subprocess call, empty list."""
    called = {"n": 0}
    monkeypatch.setattr("spa.eval.openfold3.subprocess.run",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    assert _refolder(tmp_path).refold(_seqset(sequences=[])) == []
    assert called["n"] == 0


def test_refold_raises_on_subprocess_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("spa.eval.openfold3.subprocess.run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    with pytest.raises(RuntimeError, match="OpenFold3 refold failed"):
        _refolder(tmp_path).refold(_seqset())
