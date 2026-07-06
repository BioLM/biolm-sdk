"""Discover models and health from a biolm-hub gateway (bh serve or deployed)."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from biolm.hub.config import hub_origin, normalize_hub_url

log = logging.getLogger(__name__)

_ROUTE_RE = re.compile(r"^/api/v1/([^/]+)/([^/]+)/?$")


def _openapi_url(origin: str) -> str:
    return f"{origin.rstrip('/')}/openapi.json"


def parse_openapi_paths(paths: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse OpenAPI paths into model list entries for the CLI."""
    by_slug: Dict[str, set[str]] = {}
    for path, methods in paths.items():
        if "post" not in methods:
            continue
        match = _ROUTE_RE.match(path)
        if not match:
            continue
        slug, action = match.group(1), match.group(2)
        by_slug.setdefault(slug, set()).add(action)

    models: List[Dict[str, Any]] = []
    for slug in sorted(by_slug):
        actions = sorted(by_slug[slug])
        models.append(
            {
                "model_slug": slug,
                "slug": slug,
                "model_name": slug.replace("-", " ").title(),
                "name": slug.replace("-", " ").title(),
                "actions": actions,
            }
        )
    return models


def list_models_from_openapi(
    base_url: str,
    *,
    client: Optional[httpx.Client] = None,
) -> List[Dict[str, Any]]:
    """Fetch model slugs and actions from hub OpenAPI."""
    origin = hub_origin(normalize_hub_url(base_url))
    url = _openapi_url(origin)
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))
    try:
        resp = client.get(url)
        if resp.status_code != 200:
            log.warning("Hub OpenAPI fetch failed: %s %s", resp.status_code, url)
            return []
        data = resp.json()
        paths = data.get("paths", {})
        if not isinstance(paths, dict):
            return []
        return parse_openapi_paths(paths)
    except Exception as exc:
        log.warning("Hub OpenAPI fetch error: %s", exc)
        return []
    finally:
        if owns_client and client is not None:
            client.close()


def fetch_hub_status(
    base_url: str,
    *,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Probe hub health and return summary metadata."""
    api_url = normalize_hub_url(base_url)
    origin = hub_origin(api_url)
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))
    result: Dict[str, Any] = {
        "api_url": api_url,
        "origin": origin,
        "healthy": False,
        "message": "",
        "route_count": 0,
        "slug_count": 0,
    }
    try:
        health = client.get(f"{origin}/")
        if health.status_code == 200:
            payload = health.json()
            if isinstance(payload, dict):
                result["healthy"] = payload.get("status") == "ok"
                result["message"] = payload.get("message", "")
        models = list_models_from_openapi(api_url, client=client)
        result["route_count"] = sum(len(m.get("actions", [])) for m in models)
        result["slug_count"] = len(models)
        if models and not result["healthy"]:
            result["healthy"] = True
    except Exception as exc:
        result["message"] = str(exc)
    finally:
        if owns_client and client is not None:
            client.close()
    return result
