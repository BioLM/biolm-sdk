"""Tests for BIOLM_* / BIOLMAI_* env resolution."""
import importlib
import os
import warnings


def _reload_const():
    import biolm.core.const as const
    return importlib.reload(const)


def test_biolm_local_sets_domain_and_api_url(monkeypatch):
    for key in (
        "BIOLM_BASE_DOMAIN",
        "BIOLMAI_BASE_DOMAIN",
        "BIOLM_BASE_API_URL",
        "BIOLMAI_BASE_API_URL",
        "BIOLM_LOCAL",
        "BIOLMAI_LOCAL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BIOLM_LOCAL", "true")
    const = _reload_const()
    assert const.BIOLM_BASE_DOMAIN == "http://localhost:8000"
    assert const.BIOLM_BASE_API_URL == "http://localhost:8000/api/v3"


def test_api_url_hybrid_keeps_platform_domain(monkeypatch):
    """Model API can point at local proxy while platform stays on biolm.ai."""
    for key in (
        "BIOLM_BASE_DOMAIN",
        "BIOLMAI_BASE_DOMAIN",
        "BIOLM_BASE_API_URL",
        "BIOLMAI_BASE_API_URL",
        "BIOLM_LOCAL",
        "BIOLMAI_LOCAL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BIOLM_BASE_API_URL", "http://127.0.0.1:8787/api/v3")
    const = _reload_const()
    assert const.BIOLM_BASE_API_URL == "http://127.0.0.1:8787/api/v3"
    assert const.BIOLM_BASE_DOMAIN == "https://biolm.ai"
    assert const.get_model_catalog_base() == "http://127.0.0.1:8787"
    assert const.OAUTH_AUTHORIZE_URL == "https://biolm.ai/o/authorize/"


def test_legacy_biolmai_domain_warns(monkeypatch):
    for key in (
        "BIOLM_BASE_DOMAIN",
        "BIOLMAI_BASE_DOMAIN",
        "BIOLM_BASE_API_URL",
        "BIOLMAI_BASE_API_URL",
        "BIOLM_LOCAL",
        "BIOLMAI_LOCAL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BIOLMAI_BASE_DOMAIN", "http://example.test")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        const = _reload_const()
    assert const.BIOLM_BASE_DOMAIN == "http://example.test"
    assert any("BIOLMAI_BASE_DOMAIN" in str(x.message) for x in w)
