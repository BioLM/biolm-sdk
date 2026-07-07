"""Tests for biolm.hub.discovery."""
from biolm.hub.discovery import parse_openapi_paths


SAMPLE_PATHS = {
    "/api/v1/esm2-8m/encode": {"post": {}},
    "/api/v1/esm2-8m/predict": {"post": {}},
    "/api/v1/esmfold/predict": {"post": {}},
    "/health": {"get": {}},
}


def test_parse_openapi_paths():
    models = parse_openapi_paths(SAMPLE_PATHS)
    assert len(models) == 2
    by_slug = {m["model_slug"]: m["actions"] for m in models}
    assert by_slug["esm2-8m"] == ["encode", "predict"]
    assert by_slug["esmfold"] == ["predict"]
