"""Test biolmai compatibility shim."""
import warnings

import pytest


def test_biolmai_import_warns():
    import sys
    # Remove cached modules so warning fires again
    for mod in list(sys.modules):
        if mod == "biolmai" or mod.startswith("biolmai."):
            del sys.modules[mod]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import biolmai  # noqa: F401
    assert any("renamed to biolm" in str(x.message).lower() for x in w)


def test_biolmai_reexports_biolm():
    import biolmai
    import biolm
    assert biolmai.__version__ == biolm.__version__
    assert hasattr(biolmai, "Model")
