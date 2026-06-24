"""Tests for official platform catalog loading."""
from unittest.mock import patch

from biolm.server.catalog import (
    clear_catalog_cache,
    fetch_official_catalog,
    official_catalog_slugs,
    platform_schema_url,
)


def test_platform_schema_url():
    assert platform_schema_url("esm2-8m", "encode").endswith(
        "/api/v3/schema/esm2-8m/encode/"
    )
    assert "biolm.ai" in platform_schema_url("esm2-8m", "encode")


def test_official_catalog_slugs_from_platform():
    clear_catalog_cache()
    fake_catalog = [
        {"model_slug": "esm2-8m", "model_name": "ESM2 8M"},
        {"model_slug": "esmfold", "model_name": "ESMFold"},
    ]

    class FakeResponse:
        status_code = 200

        def json(self):
            return fake_catalog

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    with patch("biolm.server.catalog.httpx.Client", FakeClient):
        slugs = official_catalog_slugs()

    assert slugs == {"esm2-8m", "esmfold"}
    clear_catalog_cache()


def test_fetch_official_catalog_uses_cache():
    clear_catalog_cache()
    calls = {"n": 0}
    fake_catalog = [{"model_slug": "esm2-8m"}]

    class FakeResponse:
        status_code = 200

        def json(self):
            return fake_catalog

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            calls["n"] += 1
            return FakeResponse()

    with patch("biolm.server.catalog.httpx.Client", FakeClient):
        first = fetch_official_catalog()
        second = fetch_official_catalog()

    assert first == fake_catalog
    assert second == fake_catalog
    assert calls["n"] == 1
    clear_catalog_cache()
