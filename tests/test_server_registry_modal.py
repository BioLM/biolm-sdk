"""Tests for Modal registry discovery."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from biolm.server.registry.modal import (
    GATEWAY_APP_NAME,
    ModalRegistry,
)


def test_slug_from_app_name_production_and_oss():
    reg = ModalRegistry()
    assert reg.slug_from_app_name("esm2-8m") == "esm2-8m"
    assert reg.slug_from_app_name("biolm-esm2-8m") == "esm2-8m"
    assert reg.slug_from_app_name("biolm_gateway") == "gateway"
    assert reg.slug_from_app_name(GATEWAY_APP_NAME) is None
    assert reg.slug_from_app_name("jupyter-hub-sandbox") is None


def test_fetch_gateway_actions_from_openapi():
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            if url.endswith("/"):
                return SimpleNamespace(
                    status_code=200,
                    json=lambda: {"supported_models_actions": [["esm2-8m", "encode"]]},
                )
            if url.endswith("/openapi.json"):
                return SimpleNamespace(
                    status_code=200,
                    json=lambda: {
                        "paths": {
                            "/api/v3/esm2-8m/encode": {},
                            "/api/v3/esmfold/predict": {},
                        }
                    },
                )
            raise AssertionError(url)

    with patch("biolm.server.registry.modal.httpx.Client", FakeClient):
        actions = ModalRegistry._fetch_gateway_actions("https://gateway.test")

    assert actions["esm2-8m"] == ["encode"]
    assert actions["esmfold"] == ["predict"]


@pytest.mark.asyncio
async def test_scan_modal_uses_gateway_for_deployed_apps(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-secret")

    fake_apps = [
        SimpleNamespace(name="esm2-8m", app_id="ap-1"),
        SimpleNamespace(name="biolm-gateway", app_id="ap-2"),
        SimpleNamespace(name="jupyter-hub-sandbox", app_id="ap-3"),
    ]

    with patch(
        "modal.experimental.list_deployed_apps",
        return_value=fake_apps,
    ), patch.object(
        ModalRegistry,
        "_resolve_gateway",
        return_value=(
            "https://biolm--biolm-gateway-gateway.modal.run/api/v3",
            {"esm2-8m": ["encode", "predict"]},
        ),
    ), patch.object(
        ModalRegistry,
        "_resolve_app_base_url",
        return_value=None,
    ), patch(
        "biolm.server.catalog.official_catalog_slugs",
        return_value={"esm2-8m", "esmfold"},
    ):
        reg = ModalRegistry(environment_name="main")
        await reg.refresh()

    entry = reg.get("esm2-8m")
    assert entry is not None
    assert entry.source == "modal"
    assert entry.base_url == "https://biolm--biolm-gateway-gateway.modal.run/api/v3"
    assert entry.actions == ["encode", "predict"]
    assert reg.get("jupyter-hub-sandbox") is None


@pytest.mark.asyncio
async def test_scan_modal_skips_apps_not_in_official_catalog(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-secret")

    fake_apps = [
        SimpleNamespace(name="esm2-8m", app_id="ap-1"),
        SimpleNamespace(name="not-a-biolm-model", app_id="ap-2"),
    ]

    with patch(
        "modal.experimental.list_deployed_apps",
        return_value=fake_apps,
    ), patch.object(
        ModalRegistry,
        "_resolve_gateway",
        return_value=("https://gateway.test/api/v3", {"esm2-8m": ["encode"]}),
    ), patch.object(
        ModalRegistry,
        "_resolve_app_base_url",
        return_value=None,
    ), patch(
        "biolm.server.catalog.official_catalog_slugs",
        return_value={"esm2-8m"},
    ):
        reg = ModalRegistry(environment_name="main")
        await reg.refresh()

    assert reg.get("esm2-8m") is not None
    assert reg.get("not-a-biolm-model") is None
    assert len(reg.list()) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modal_discovery_live_main_environment():
    """Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET and network access."""
    reg = ModalRegistry(environment_name="main")
    if not ModalRegistry.modal_credentials_present():
        pytest.skip("Modal credentials not configured")

    await reg.refresh()
    entries = reg.list()
    assert len(entries) > 10

    esm = reg.get("esm2-8m")
    assert esm is not None
    assert esm.base_url.endswith("/api/v3")
    assert "biolm-gateway-gateway" in esm.base_url or esm.base_url.startswith("http")
