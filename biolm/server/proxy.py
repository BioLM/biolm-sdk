"""HTTP proxy to Modal deployment endpoints."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import httpx
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(1200.0, connect=10.0)


class ModelProxy:
    def __init__(self, timeout: httpx.Timeout = DEFAULT_TIMEOUT):
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def forward(
        self,
        entry_base_url: str,
        path: str,
        request: Request,
    ) -> Response:
        url = f"{entry_base_url.rstrip('/')}/{path.lstrip('/')}"
        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        try:
            resp = await self._client.request(
                request.method,
                url,
                content=body,
                headers=headers,
            )
        except httpx.RequestError as exc:
            log.warning("Proxy error for %s: %s", url, exc)
            import json
            body = json.dumps({"error": str(exc), "detail": str(exc), "status_code": 502})
            return Response(content=body, status_code=502, media_type="application/json")

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
            },
            media_type=resp.headers.get("content-type"),
        )

    async def get_json(self, entry_base_url: str, path: str) -> Tuple[int, Optional[dict]]:
        url = f"{entry_base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = await self._client.get(url)
            if resp.status_code == 200:
                return resp.status_code, resp.json()
            return resp.status_code, None
        except Exception:
            return 502, None

    async def forward_json(self, entry_base_url: str, path: str, body: dict) -> Response:
        url = f"{entry_base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = await self._client.post(url, json=body)
        except httpx.RequestError as exc:
            log.warning("Proxy error for %s: %s", url, exc)
            import json
            payload = json.dumps({"error": str(exc), "detail": str(exc), "status_code": 502})
            return Response(content=payload, status_code=502, media_type="application/json")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
            },
            media_type=resp.headers.get("content-type"),
        )

    async def forward_get(self, entry_base_url: str, path: str) -> Response:
        url = f"{entry_base_url.rstrip('/')}/{path.lstrip('/')}" if path else entry_base_url
        try:
            resp = await self._client.get(url)
        except httpx.RequestError as exc:
            log.warning("Proxy error for %s: %s", url, exc)
            import json
            payload = json.dumps({"error": str(exc), "detail": str(exc), "status_code": 502})
            return Response(content=payload, status_code=502, media_type="application/json")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
            },
            media_type=resp.headers.get("content-type"),
        )
