"""Minimal smoke test to verify package imports."""


def test_import():
    import core_mem

    assert core_mem.__version__ == "0.1.0"
