"""OSS model catalog loading and merge with registry."""
from __future__ import annotations

import json
import logging
from importlib import resources
from typing import Any, Dict, List, Optional

from biolm.server.registry.base import ModelEntry, ModelStatus

log = logging.getLogger(__name__)


def load_catalog() -> List[Dict[str, Any]]:
    """Load bundled OSS catalog JSON."""
    try:
        data_path = resources.files("biolm.server.data").joinpath("catalog.json")
        with data_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        log.warning("Could not load bundled catalog: %s", exc)
    return []


def catalog_by_slug() -> Dict[str, Dict[str, Any]]:
    return {
        (m.get("model_slug") or m.get("slug")): m
        for m in load_catalog()
        if m.get("model_slug") or m.get("slug")
    }


def resolve_exposed_models(registry_entries: List[ModelEntry]) -> List[Dict[str, Any]]:
    """Merge registry deployments with catalog metadata for community-api-models."""
    catalog = catalog_by_slug()
    exposed: List[Dict[str, Any]] = []
    for entry in registry_entries:
        meta = dict(catalog.get(entry.slug, {}))
        meta.setdefault("model_slug", entry.slug)
        meta.setdefault("slug", entry.slug)
        meta.setdefault("model_name", entry.slug)
        meta.setdefault("name", entry.slug)
        if entry.actions:
            meta["actions"] = entry.actions
        elif "actions" not in meta:
            actions = []
            if meta.get("encoder"):
                actions.append("encode")
            if meta.get("predictor"):
                actions.append("predict")
            if meta.get("generator"):
                actions.append("generate")
            meta["actions"] = actions
        meta["deployment_status"] = entry.status.value
        meta["deployment_source"] = entry.source
        meta["deployment_url"] = entry.base_url
        exposed.append(meta)
    return exposed


def get_catalog_model(slug: str) -> Optional[Dict[str, Any]]:
    return catalog_by_slug().get(slug)


def list_catalog_models() -> List[Dict[str, Any]]:
    return load_catalog()
