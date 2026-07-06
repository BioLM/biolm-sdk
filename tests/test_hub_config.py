"""Tests for biolm.hub.config."""
from pathlib import Path

import pytest

from biolm.hub.config import (
    clear_hub_config,
    normalize_hub_url,
    read_hub_api_url,
    write_hub_api_url,
)


def test_normalize_hub_url_adds_api_v1():
    assert normalize_hub_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/api/v1"
    assert normalize_hub_url("http://127.0.0.1:8000/api/v1") == "http://127.0.0.1:8000/api/v1"


def test_normalize_hub_url_strips_other_api_versions():
    assert normalize_hub_url("http://localhost:8000/api/v3") == "http://localhost:8000/api/v1"


def test_write_and_read_hub_config(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    monkeypatch.setattr("biolm.hub.config.config_path", lambda: config_file)

    write_hub_api_url("http://127.0.0.1:8000")
    assert read_hub_api_url() == "http://127.0.0.1:8000/api/v1"


def test_clear_hub_config(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    monkeypatch.setattr("biolm.hub.config.config_path", lambda: config_file)

    write_hub_api_url("http://127.0.0.1:8000")
    assert clear_hub_config() is True
    assert read_hub_api_url() is None
    assert not config_file.exists()
