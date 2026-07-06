"""Persisted configuration for pointing py-biolm at a biolm-hub gateway."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_HUB_ORIGIN = "http://127.0.0.1:8000"
HUB_API_SUFFIX = "/api/v1"


def config_path() -> Path:
    return Path.home() / ".biolm" / "config.yaml"


def normalize_hub_url(url: str) -> str:
    """Normalize a bh serve / gateway URL to the v1 API base."""
    url = (url or DEFAULT_HUB_ORIGIN).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if url.endswith(HUB_API_SUFFIX):
        return url
    for suffix in ("/api/v3", "/api/v2"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return f"{url.rstrip('/')}{HUB_API_SUFFIX}"


def hub_origin(api_url: str) -> str:
    """Strip /api/v* suffix to get the site root (catalog UI, health)."""
    url = api_url.rstrip("/")
    for suffix in (HUB_API_SUFFIX, "/api/v3", "/api/v2"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def is_hub_api_url(url: str) -> bool:
    return url.rstrip("/").endswith(HUB_API_SUFFIX)


def read_config() -> Dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_hub_api_url() -> Optional[str]:
    value = read_config().get("hub_api_url")
    if isinstance(value, str) and value.strip():
        return normalize_hub_url(value)
    return None


def write_hub_api_url(url: str) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_config()
    data["hub_api_url"] = normalize_hub_url(url)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return path


def clear_hub_config() -> bool:
    path = config_path()
    if not path.exists():
        return False
    data = read_config()
    if "hub_api_url" not in data:
        return False
    data.pop("hub_api_url", None)
    if data:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        path.unlink(missing_ok=True)
    return True
