"""GPU memory must be released between flywheel stages, because Stage 3 is a SUBPROCESS.

These live outside `test_eval_flywheel.py` on purpose: that module is skipped wholesale unless a real
RFD3 checkpoint, a ProteinMPNN clone and a CUDA device are all present, and the checks here need none
of that. The ordering check in particular is pure source inspection, and it is the one that would have
caught the original bug.
"""

from __future__ import annotations

import pytest

def test_release_gpu_memory_is_a_noop_without_cuda(monkeypatch):
    from spa.eval import flywheel as fw

    class _FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch)
    assert fw.release_gpu_memory("test") == {}


def test_release_gpu_memory_reports_before_and_after():
    import torch

    from spa.eval import flywheel as fw

    got = fw.release_gpu_memory("unit test")
    if not torch.cuda.is_available():
        assert got == {}
        return
    assert set(got) == {"before_allocated_gib", "after_allocated_gib",
                        "before_reserved_gib", "after_reserved_gib"}
    # empty_cache() can only shrink the reserved arena, never grow it.
    assert got["after_reserved_gib"] <= got["before_reserved_gib"] + 1e-6


def test_flywheel_releases_at_both_stage_boundaries():
    """Static check: the release must sit AFTER generate and AFTER inverse_fold, before the refold.

    A unit test cannot exercise the ordering without a GPU and a real engine, and the ordering is the
    whole point: releasing after Stage 3 would be useless, since the subprocess has already failed.
    """
    import inspect

    from spa.eval import flywheel as fw

    src = inspect.getsource(fw.run_flywheel)
    i_gen = src.index("designs = generate(cfg)")
    i_rel1 = src.index('release_gpu_memory("after Stage 1')
    i_seq = src.index("seqsets = inverse_fold(")
    i_rel2 = src.index('release_gpu_memory("after Stage 2')
    i_refold = src.index("refold_all(")
    assert i_gen < i_rel1 < i_seq < i_rel2 < i_refold, "release calls are out of order"
