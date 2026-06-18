"""Tests for config-based model registry."""
import asyncio
import textwrap

import pytest

from biolm.server.registry.config import ConfigRegistry


@pytest.mark.asyncio
async def test_config_registry_from_yaml(tmp_path):
    cfg = tmp_path / "server.yaml"
    cfg.write_text(textwrap.dedent("""
        models:
          - slug: esm2-8m
            url: http://modal.test/api/v3
            actions: [encode]
    """))
    reg = ConfigRegistry(config_path=str(cfg))
    await reg.refresh()
    entry = reg.get("esm2-8m")
    assert entry is not None
    assert entry.base_url == "http://modal.test/api/v3"
    assert entry.actions == ["encode"]


@pytest.mark.asyncio
async def test_config_registry_from_env_slugs():
    reg = ConfigRegistry(slugs=["esmfold"])
    await reg.refresh()
    assert reg.get("esmfold") is not None
