"""Tests for canonical ~/.biolm user paths with ~/.biolmai fallback."""
import warnings
from pathlib import Path

import pytest

from biolm.core.paths import (
    legacy_user_config_dir,
    resolve_user_path,
    user_config_dir,
    warn_deprecated_path,
)


def test_user_config_dir_is_biolm():
    assert user_config_dir() == Path.home() / ".biolm"


def test_legacy_user_config_dir_is_biolmai():
    assert legacy_user_config_dir() == Path.home() / ".biolmai"


def test_resolve_user_path_prefers_canonical(tmp_path, monkeypatch):
    monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
    canonical = tmp_path / ".biolm" / "credentials"
    legacy = tmp_path / ".biolmai" / "credentials"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"access": "legacy"}')
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"access": "canonical"}')

    assert resolve_user_path("credentials") == canonical


def test_resolve_user_path_falls_back_to_legacy_with_warning(tmp_path, monkeypatch):
    monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
    legacy = tmp_path / ".biolmai" / "credentials"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"access": "legacy"}')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_user_path("credentials")

    assert resolved == legacy
    assert any("deprecated" in str(w.message).lower() for w in caught)


def test_resolve_user_path_returns_canonical_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
    expected = tmp_path / ".biolm" / "pipelines"
    assert resolve_user_path("pipelines") == expected


def test_warn_deprecated_path_only_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated_path("/old", "/new")
        warn_deprecated_path("/old", "/new")
    assert len(caught) == 1
