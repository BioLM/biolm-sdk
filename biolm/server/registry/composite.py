"""Composite registry merging config and Modal sources."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from biolm.server.registry.base import ModelEntry, ModelRegistry, ModelStatus
from biolm.server.registry.config import ConfigRegistry
from biolm.server.registry.modal import ModalRegistry

log = logging.getLogger(__name__)


class CompositeRegistry:
    """Merge registries; Modal entries override config for same slug."""

    def __init__(
        self,
        config_registry: ConfigRegistry,
        modal_registry: ModalRegistry,
        health_check: bool = True,
    ):
        self._config = config_registry
        self._modal = modal_registry
        self._health_check = health_check
        self._entries: Dict[str, ModelEntry] = {}

    async def refresh(self) -> None:
        await self._config.refresh()
        await self._modal.refresh()

        merged: Dict[str, ModelEntry] = {}
        for entry in self._config.list():
            merged[entry.slug] = entry
        for entry in self._modal.list():
            merged[entry.slug] = entry

        if self._health_check:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                for slug, entry in list(merged.items()):
                    merged[slug] = await self._check_health(client, entry)

        self._entries = merged

    async def _check_health(self, client: httpx.AsyncClient, entry: ModelEntry) -> ModelEntry:
        base = entry.base_url.rstrip("/")
        actions = entry.actions or ["encode", "predict", "generate", "inference"]
        status = ModelStatus.UNREACHABLE

        for action in actions:
            probe_url = f"{base}/{entry.slug}/{action}"
            try:
                resp = await client.head(probe_url)
                if resp.status_code in (200, 405):
                    status = ModelStatus.READY
                    break
            except Exception:
                continue

        if status != ModelStatus.READY:
            for action in ("encode", "predict"):
                probe_url = f"{base}/schema/{entry.slug}/{action}/"
                try:
                    resp = await client.get(probe_url)
                    if resp.status_code == 200:
                        status = ModelStatus.READY
                        break
                except Exception:
                    continue

        return ModelEntry(
            slug=entry.slug,
            base_url=entry.base_url,
            status=status,
            source=entry.source,
            actions=entry.actions,
        )

    def get(self, slug: str) -> Optional[ModelEntry]:
        return self._entries.get(slug)

    def list(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def snapshot(self) -> Dict[str, ModelEntry]:
        return dict(self._entries)
