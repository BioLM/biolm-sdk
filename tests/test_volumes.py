"""Compatibility and deprecation behavior for :mod:`biolm.volumes`."""
import warnings

import pytest

from biolm.volumes import Volume


def test_volume_construction_warns_at_call_site():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        volume = Volume("results", api_key="test-token")

    assert volume.name == "results"
    assert volume._api_key == "test-token"
    assert len(caught) == 1
    warning = caught[0]
    assert warning.category is DeprecationWarning
    assert warning.filename == __file__
    message = str(warning.message)
    assert "Modal-backed volumes are runtime-managed" in message
    assert "not a local SDK storage API" in message
    assert "BioLM console" in message
    assert "Jupyter" in message
    assert "protocol export" in message


def test_volume_default_constructor_compatibility():
    with pytest.warns(DeprecationWarning):
        volume = Volume()

    assert volume.name is None
    assert volume._api_key is None


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("list", ()),
        ("create", ("results",)),
        ("get", ()),
        ("delete", ()),
    ],
)
def test_volume_methods_raise_consistent_unsupported_error(method_name, args):
    with pytest.warns(DeprecationWarning):
        volume = Volume("results")

    with pytest.raises(NotImplementedError) as excinfo:
        getattr(volume, method_name)(*args)

    assert str(excinfo.value) == (
        "Direct local volume management is unsupported. Modal-backed volumes "
        "are managed by BioLM runtimes; use the BioLM console, Jupyter, or "
        "protocol export workflows."
    )


def test_volume_remains_exported_from_top_level_package():
    from biolm import Volume as TopLevelVolume

    assert TopLevelVolume is Volume
