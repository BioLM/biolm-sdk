"""Platform and bundled OSS model catalog (not hub route discovery)."""
from __future__ import annotations

import json
import logging
import time
from importlib import resources
from typing import Any, Dict, List, Optional, Set

import httpx

from biolm.core.const import get_base_domain

log = logging.getLogger(__name__)

CATALOG_TTL_SECONDS = 300
PLATFORM_CATALOG_PATHS = (
    "/api/ui/community-api-models/",
    "/ui/community-api-models/",
)

_catalog_cache: Optional[tuple[float, List[Dict[str, Any]]]] = None


def _load_bundled_catalog() -> List[Dict[str, Any]]:
    """Offline fallback when platform catalog is unreachable."""
    try:
        data_path = resources.files("biolm.hub.data").joinpath("catalog.json")
        with data_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        log.warning("Could not load bundled catalog: %s", exc)
    return []


def fetch_official_catalog(*, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Fetch the official model catalog from the BioLM platform (biolm.ai)."""
    global _catalog_cache
    now = time.monotonic()
    if (
        not force_refresh
        and _catalog_cache is not None
        and (now - _catalog_cache[0]) < CATALOG_TTL_SECONDS
    ):
        return _catalog_cache[1]

    platform_base = get_base_domain().rstrip("/")
    for path in PLATFORM_CATALOG_PATHS:
        url = f"{platform_base}{path}"
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    _catalog_cache = (now, data)
                    log.info("Loaded %d model(s) from official catalog at %s", len(data), url)
                    return data
        except Exception as exc:
            log.debug("Official catalog fetch failed for %s: %s", url, exc)

    bundled = _load_bundled_catalog()
    if bundled:
        log.warning(
            "Using bundled catalog fallback (%d models); platform catalog unavailable",
            len(bundled),
        )
    _catalog_cache = (now, bundled)
    return bundled


def official_catalog_slugs() -> Set[str]:
    """Slugs in the official BioLM catalog."""
    slugs: Set[str] = set()
    for model in fetch_official_catalog():
        slug = model.get("model_slug") or model.get("slug")
        if slug:
            slugs.add(slug)
    return slugs


def catalog_by_slug() -> Dict[str, Dict[str, Any]]:
    return {
        (m.get("model_slug") or m.get("slug")): m
        for m in fetch_official_catalog()
        if m.get("model_slug") or m.get("slug")
    }


def get_catalog_model(slug: str) -> Optional[Dict[str, Any]]:
    return catalog_by_slug().get(slug)


def list_catalog_models() -> List[Dict[str, Any]]:
    return fetch_official_catalog()


def clear_catalog_cache() -> None:
    """Clear cached platform catalog (for tests)."""
    global _catalog_cache
    _catalog_cache = None
