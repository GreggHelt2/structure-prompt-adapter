"""Scaffold sanity (kickoff step 1): the package imports and the portability boundary works.

These are real, passing tests — they verify the skeleton itself, not SPA behavior (which lands
in later steps and is gated by tests/test_identity_at_init.py).
"""


def test_package_imports():
    import spa

    assert spa.__version__


def test_resolve_device_cpu():
    from spa.utils.device import resolve_device

    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_never_raises():
    from spa.utils.device import resolve_device

    # `auto` must resolve regardless of whether CUDA is visible in this env.
    assert resolve_device("auto").type in {"cpu", "cuda"}
