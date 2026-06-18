"""Config-based model registry."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from biolm.server.registry.base import ModelEntry, ModelStatus

log = logging.getLogger(__name__)


class ConfigRegistry:
    """Load deployments from env var and optional YAML config."""

    def __init__(
        self,
        slugs: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        default_base_url_template: str = "http://127.0.0.1:8000/api/v3",
    ):
        self._entries: Dict[str, ModelEntry] = {}
        self._slugs = slugs or []
        self._config_path = config_path
        self._default_template = default_base_url_template

    async def refresh(self) -> None:
        entries: Dict[str, ModelEntry] = {}

        if self._config_path and Path(self._config_path).exists():
            try:
                with open(self._config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for item in data.get("models", []):
                    slug = item.get("slug") or item.get("model_slug")
                    url = item.get("url") or item.get("base_url")
                    if slug and url:
                        entries[slug] = ModelEntry(
                            slug=slug,
                            base_url=url.rstrip("/"),
                            status=ModelStatus.UNKNOWN,
                            source="config",
                            actions=item.get("actions") or [],
                        )
            except Exception as exc:
                log.warning("Failed to load server config %s: %s", self._config_path, exc)

        for slug in self._slugs:
            if slug not in entries:
                entries[slug] = ModelEntry(
                    slug=slug,
                    base_url=f"{self._default_template.rstrip('/')}",
                    status=ModelStatus.UNKNOWN,
                    source="env",
                )

        self._entries = entries
        log.info("Config registry loaded %d model(s)", len(self._entries))

    def get(self, slug: str) -> Optional[ModelEntry]:
        return self._entries.get(slug)

    def list(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def snapshot(self) -> Dict[str, ModelEntry]:
        return dict(self._entries)
