"""Tests for biolm.lab.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from biolm.lab.config import (
    load_config,
    resolve_auth,
    write_example_config,
)


def test_load_config_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.path is None
    assert cfg.connectors == {}
    assert cfg.experiments == {}


def test_load_config_and_experiments(tmp_path):
    path = tmp_path / "lltp.yaml"
    path.write_text(
        """
version: 1
default_connector: adaptyv
connectors:
  adaptyv:
    auth:
      token_env: ADAPTYV_API_TOKEN
    defaults:
      service_id: adaptyv-lltp.expression-v1
experiments:
  express:
    connector: adaptyv
    service_id: adaptyv-lltp.expression-v1
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.default_connector == "adaptyv"
    assert "adaptyv" in cfg.connectors
    exp = cfg.get_experiment("express")
    assert exp.connector == "adaptyv"
    assert exp.params["service_id"] == "adaptyv-lltp.expression-v1"


def test_resolve_auth_env_wins():
    auth = {"token": "from-yaml", "token_env": "MY_TOKEN"}
    resolved = resolve_auth(auth, environ={"MY_TOKEN": "from-env"})
    assert resolved["token"] == "from-env"

    resolved2 = resolve_auth(auth, environ={})
    assert resolved2["token"] == "from-yaml"


def test_write_example_config(tmp_path):
    path = tmp_path / "lltp.yaml"
    write_example_config(path)
    assert path.is_file()
    with pytest.raises(FileExistsError):
        write_example_config(path)
    write_example_config(path, force=True)
