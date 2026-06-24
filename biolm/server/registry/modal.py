"""Modal workspace introspection for deployed BioLM models."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

from biolm.server.registry.base import ModelEntry, ModelStatus

log = logging.getLogger(__name__)

# Conventions for OSS Modal deployments (see docs/server/oss-integration.md)
APP_NAME_PREFIX = "biolm-"
APP_LABEL_KEY = "biolm.model"
GATEWAY_APP_NAME = "biolm-gateway"
GATEWAY_FUNCTION_NAMES = ("gateway", "app", "api", "web")
EXCLUDED_APP_NAMES = frozenset(
    {
        GATEWAY_APP_NAME,
        "mlflow-tracking",
        "jupyter-hub-sandbox",
    }
)
DEFAULT_MODAL_ENVIRONMENT = "main"


class ModalRegistry:
    """Discover deployed models via Modal SDK conventions."""

    def __init__(self, environment_name: Optional[str] = None):
        self._entries: Dict[str, ModelEntry] = {}
        self._environment_name = environment_name or os.environ.get(
            "BIOLM_SERVER_MODAL_ENV", DEFAULT_MODAL_ENVIRONMENT
        )

    @staticmethod
    def modal_credentials_present() -> bool:
        if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
            return True
        try:
            import modal.config as modal_config

            cfg = modal_config.config.to_dict()
            return bool(cfg.get("token_id") and cfg.get("token_secret"))
        except Exception:
            return False

    @property
    def environment_name(self) -> str:
        return self._environment_name

    async def refresh(self) -> None:
        self._entries = {}
        if not self.modal_credentials_present():
            log.info("Modal credentials not configured; skipping Modal registry scan")
            return
        try:
            entries = await asyncio.to_thread(self._scan_modal_sync)
            self._entries = entries
            log.info(
                "Modal registry found %d model(s) in environment '%s'",
                len(entries),
                self._environment_name,
            )
        except ImportError:
            log.warning("Modal SDK not installed; install biolm[server] for Modal discovery")
        except Exception as exc:
            log.warning("Modal registry scan failed: %s", exc)

    def _scan_modal_sync(self) -> Dict[str, ModelEntry]:
        from modal.experimental import get_app_objects, list_deployed_apps
        from biolm.server.catalog import official_catalog_slugs

        entries: Dict[str, ModelEntry] = {}
        apps = list_deployed_apps(environment_name=self._environment_name)
        gateway_base_url, gateway_actions = self._resolve_gateway(self._environment_name)
        catalog_slugs = official_catalog_slugs()

        for app_info in apps:
            slug = self.slug_from_app_name(app_info.name)
            if not slug:
                continue
            if catalog_slugs and slug not in catalog_slugs:
                log.debug(
                    "Skipping Modal app '%s': slug '%s' not in official BioLM catalog",
                    app_info.name,
                    slug,
                )
                continue

            base_url = gateway_base_url or self._resolve_app_base_url(
                app_info.name, self._environment_name
            )
            if not base_url:
                log.debug(
                    "Skipping Modal app '%s': no web endpoint URL resolved",
                    app_info.name,
                )
                continue

            actions = gateway_actions.get(slug, []) if gateway_actions else []
            entries[slug] = ModelEntry(
                slug=slug,
                base_url=base_url,
                status=ModelStatus.UNKNOWN,
                source="modal",
                actions=actions,
            )

        return entries

    def _resolve_gateway(
        self, environment_name: str
    ) -> Tuple[Optional[str], Dict[str, List[str]]]:
        """Return shared gateway API base URL and per-slug actions, if deployed."""
        import modal
        from modal.experimental import get_app_objects

        try:
            objs = get_app_objects(GATEWAY_APP_NAME, environment_name=environment_name)
        except Exception as exc:
            log.debug("No Modal gateway app in '%s': %s", environment_name, exc)
            return None, {}

        gateway_root: Optional[str] = None
        for name in GATEWAY_FUNCTION_NAMES:
            obj = objs.get(name)
            if isinstance(obj, modal.Function):
                url = obj.get_web_url()
                if url:
                    gateway_root = url.rstrip("/")
                    break

        if not gateway_root:
            for obj in objs.values():
                if isinstance(obj, modal.Function):
                    url = obj.get_web_url()
                    if url:
                        gateway_root = url.rstrip("/")
                        break

        if not gateway_root:
            return None, {}

        actions = self._fetch_gateway_actions(gateway_root)
        return f"{gateway_root}/api/v3", actions

    @staticmethod
    def _fetch_gateway_actions(gateway_root: str) -> Dict[str, List[str]]:
        """Parse gateway OpenAPI or status payload for model actions."""
        actions: Dict[str, Set[str]] = {}
        try:
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                status_resp = client.get(f"{gateway_root}/")
                if status_resp.status_code == 200:
                    payload = status_resp.json()
                    for slug, action in payload.get("supported_models_actions", []):
                        if slug and action:
                            actions.setdefault(slug, set()).add(action)

                openapi_resp = client.get(f"{gateway_root}/openapi.json")
                if openapi_resp.status_code == 200:
                    for path in openapi_resp.json().get("paths", {}):
                        match = re.match(r"^/api/v3/([^/]+)/([^/]+)/?$", path)
                        if match:
                            slug, action = match.group(1), match.group(2)
                            actions.setdefault(slug, set()).add(action)
        except Exception as exc:
            log.debug("Failed to fetch gateway metadata from %s: %s", gateway_root, exc)

        return {slug: sorted(vals) for slug, vals in actions.items()}

    @staticmethod
    def _resolve_app_base_url(app_name: str, environment_name: str) -> Optional[str]:
        """Resolve a per-app Modal web endpoint when no shared gateway is deployed."""
        import modal
        from modal.experimental import get_app_objects

        try:
            objs = get_app_objects(app_name, environment_name=environment_name)
        except Exception:
            return None

        for obj in objs.values():
            if isinstance(obj, modal.Function):
                url = obj.get_web_url()
                if url:
                    return f"{url.rstrip('/')}/api/v3"
        return None

    @staticmethod
    def slug_from_app_name(name: str) -> Optional[str]:
        if name in EXCLUDED_APP_NAMES:
            return None
        if name.startswith(APP_NAME_PREFIX):
            return name[len(APP_NAME_PREFIX) :]
        match = re.match(r"^biolm[-_](.+)$", name)
        if match:
            return match.group(1)
        return name

    @staticmethod
    def gateway_host(base_url: str) -> str:
        parsed = urlparse(base_url)
        return parsed.netloc or base_url

    def get(self, slug: str) -> Optional[ModelEntry]:
        return self._entries.get(slug)

    def list(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def snapshot(self) -> Dict[str, ModelEntry]:
        return dict(self._entries)
