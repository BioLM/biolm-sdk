"""Tests for biolm server HTTP routes."""
import pytest
from unittest.mock import AsyncMock, patch

from biolm.server.settings import ServerSettings


@pytest.fixture
def app():
    pytest.importorskip("fastapi")
    from biolm.server.app import create_app

    settings = ServerSettings.from_env(
        host="127.0.0.1",
        port=8787,
        auth_mode="none",
        models_env="esm2-8m",
        config_path="/nonexistent/server.yaml",
        refresh_seconds=3600,
    )
    return create_app(settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["auth_mode"] == "none"


def test_platform_route_returns_501(client):
    resp = client.get("/api/v3/protocols/foo/")
    assert resp.status_code == 501
    assert "Not supported" in resp.json()["detail"]


def test_community_models_lists_configured(client):
    resp = client.get("/api/ui/community-api-models/")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    slugs = {m.get("model_slug") or m.get("slug") for m in models}
    assert "esm2-8m" in slugs


def test_undeployed_model_returns_404(client):
    resp = client.get("/api/ui/community-api-models/not-a-model/")
    assert resp.status_code == 404


def test_catalog_endpoint(client):
    resp = client.get("/api/v3/catalog/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_proxy_forwards_post(client):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b'{"results": []}'
    mock_response.headers = {"content-type": "application/json"}

    with patch("biolm.server.proxy.httpx.AsyncClient.post", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response
        resp = client.post(
            "/api/v3/esm2-8m/encode/",
            json={"items": [{"sequence": "ACDE"}]},
        )
    assert resp.status_code == 200
    assert mock_req.called
