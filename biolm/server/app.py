"""FastAPI application for biolm server."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from biolm.server.auth import AuthBackend, build_auth_backend
from biolm.server.catalog import (
    get_catalog_model,
    list_catalog_models,
    platform_schema_url,
    resolve_exposed_models,
)
from biolm.server.proxy import ModelProxy
from biolm.server.registry import CompositeRegistry, ConfigRegistry, ModalRegistry
from biolm.server.routes.platform import not_supported
from biolm.server.settings import ServerSettings

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - optional extra
    FastAPI = None  # type: ignore
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore


def create_app(settings: Optional[ServerSettings] = None):
    if FastAPI is None:
        raise ImportError(
            "biolm server requires the server extra: pip install biolm[server]"
        )

    settings = settings or ServerSettings.from_env()
    settings.validate()

    config_registry = ConfigRegistry(
        slugs=settings.configured_slugs(),
        config_path=settings.config_path,
    )
    modal_registry = ModalRegistry(environment_name=settings.modal_environment)
    registry = CompositeRegistry(config_registry, modal_registry, health_check=settings.health_check)
    proxy = ModelProxy()
    auth_backend: AuthBackend = build_auth_backend(settings.auth_mode, settings.server_token)
    refresh_task: Optional[asyncio.Task] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal refresh_task
        await registry.refresh()

        async def _refresh_loop():
            while True:
                await asyncio.sleep(settings.refresh_seconds)
                try:
                    await registry.refresh()
                except Exception as exc:
                    log.warning("Registry refresh failed: %s", exc)

        refresh_task = asyncio.create_task(_refresh_loop())
        yield
        if refresh_task:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
        await proxy.close()

    app = FastAPI(title="biolm server", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        failure = auth_backend.validate(request)
        if failure is not None:
            return failure
        return await call_next(request)

    @app.get("/health")
    async def health():
        entries = registry.list()
        return {
            "status": "ok",
            "host": settings.host,
            "port": settings.port,
            "modal_environment": settings.modal_environment,
            "modal_configured": ModalRegistry.modal_credentials_present(),
            "auth_mode": settings.auth_mode,
            "model_count": len(entries),
            "deployments": [e.to_dict() for e in entries],
        }

    @app.get("/api/ui/community-api-models/")
    @app.get("/ui/community-api-models/")
    async def list_models():
        return resolve_exposed_models(registry.list())

    @app.get("/api/ui/community-api-models/{slug}/")
    @app.get("/ui/community-api-models/{slug}/")
    async def model_detail(slug: str):
        entry = registry.get(slug)
        if not entry:
            return JSONResponse(status_code=404, content={"detail": f"Model '{slug}' not deployed on this server."})
        meta = get_catalog_model(slug) or {}
        result = dict(meta)
        result.setdefault("model_slug", slug)
        result.setdefault("slug", slug)
        result["deployment_status"] = entry.status.value
        result["deployment_url"] = entry.base_url
        return result

    @app.get("/api/v3/schema/{model}/{action}/")
    async def schema_route(model: str, action: str):
        entry = registry.get(model)
        if not entry:
            return JSONResponse(status_code=404, content={"detail": f"Model '{model}' not deployed on this server."})
        return await proxy.forward_get(platform_schema_url(model, action), "")

    @app.post("/api/v3/{model}/{action}/")
    async def model_action(model: str, action: str, body: Dict[str, Any]):
        entry = registry.get(model)
        if not entry:
            return JSONResponse(status_code=404, content={"detail": f"Model '{model}' not deployed on this server."})
        return await proxy.forward_json(entry.base_url, f"{model}/{action}/", body)

    @app.get("/api/v3/protocols/{protocol_id}/")
    async def protocol_fetch(protocol_id: str):
        return not_supported()

    @app.api_route("/api/users/me/", methods=["GET"])
    @app.api_route("/api/auth/{path:path}", methods=["GET", "POST"])
    @app.api_route("/o/{path:path}", methods=["GET", "POST"])
    async def platform_auth(path: str = ""):
        return not_supported()

    @app.get("/api/v3/catalog/")
    async def full_catalog():
        """Full official catalog from the BioLM platform."""
        return list_catalog_models()

    return app
