"""Modal workspace introspection for deployed BioLM models."""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

from biolm.server.registry.base import ModelEntry, ModelStatus

log = logging.getLogger(__name__)

# Conventions for OSS Modal deployments (see docs/server/oss-integration.md)
APP_NAME_PREFIX = "biolm-"
APP_LABEL_KEY = "biolm.model"


class ModalRegistry:
    """Discover deployed models via Modal SDK conventions."""

    def __init__(self):
        self._entries: Dict[str, ModelEntry] = {}

    @staticmethod
    def modal_credentials_present() -> bool:
        return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))

    async def refresh(self) -> None:
        self._entries = {}
        if not self.modal_credentials_present():
            log.info("Modal credentials not configured; skipping Modal registry scan")
            return
        try:
            entries = await self._scan_modal()
            self._entries = entries
            log.info("Modal registry found %d model(s)", len(entries))
        except ImportError:
            log.warning("Modal SDK not installed; install biolm[server] for Modal discovery")
        except Exception as exc:
            log.warning("Modal registry scan failed: %s", exc)

    async def _scan_modal(self) -> Dict[str, ModelEntry]:
        """Scan Modal for biolm-* apps. Stub: convention documented; full wiring in follow-up."""
        # Modal introspection API varies by version; OSS repo will finalize conventions.
        # For now, attempt a best-effort import and log guidance if unavailable.
        try:
            import modal  # noqa: F401
        except ImportError as exc:
            raise ImportError("modal package required") from exc

        entries: Dict[str, ModelEntry] = {}
        log.info(
            "Modal scan stub active — deploy apps named '%s{slug}' with label '%s=<slug>' "
            "or set BIOLM_SERVER_MODELS / ~/.biolm/server.yaml until OSS conventions land.",
            APP_NAME_PREFIX,
            APP_LABEL_KEY,
        )
        return entries

    @staticmethod
    def slug_from_app_name(name: str) -> Optional[str]:
        if name.startswith(APP_NAME_PREFIX):
            return name[len(APP_NAME_PREFIX):]
        match = re.match(r"^biolm[-_](.+)$", name)
        return match.group(1) if match else None

    def get(self, slug: str) -> Optional[ModelEntry]:
        return self._entries.get(slug)

    def list(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def snapshot(self) -> Dict[str, ModelEntry]:
        return dict(self._entries)
