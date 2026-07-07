"""Protocol Submission API client — programmatic run submission, progress tracking, and results retrieval.

This file is a renamed copy of ``biolmai/protocol_runs.py`` from `py-biolm`,
kept for backwards compatibility when migrating to the ``biolm`` namespace.
"""

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from biolm.core.const import BIOLMAI_BASE_DOMAIN

_DEFAULT_TIMEOUT = 30
_UPLOAD_TIMEOUT = 120
_DOWNLOAD_TIMEOUT = 120


def _auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    token = api_key or os.environ.get("BIOLMAI_TOKEN") or os.environ.get("BIOLM_TOKEN")
    if not token:
        raise ValueError(
            "No API key found. Set the BIOLMAI_TOKEN environment variable or pass api_key= to ProtocolClient().\n"
            "Get a token at https://biolm.ai/console/user/api-keys/"
        )
    return {"Authorization": f"Token {token}"}


class ProtocolRunError(Exception):
    """A protocol run failed, was cancelled, or the API returned an error."""


class ProtocolNotFoundError(ProtocolRunError):
    """The requested protocol slug/version does not exist or is not accessible."""


class ProtocolRun:
    """A submitted protocol run returned by :meth:`ProtocolClient.submit`."""

    def __init__(self, data: Dict[str, Any], client: "ProtocolClient") -> None:
        self.run_id: str = data["run_id"]
        self.protocol_slug: str = data.get("protocol_slug", "")
        self.protocol_version: int = data.get("protocol_version", 1)
        self.status: str = data.get("status", "scheduled")
        self._client = client
        self._failure_error: Optional[str] = None

    def refresh(self) -> "ProtocolRun":
        data = self._client._get(f"runs/{self.run_id}/")
        self.status = data.get("status", self.status)
        return self

    def progress(self) -> Dict[str, Any]:
        return self._client._get(f"runs/{self.run_id}/progress/")

    def wait(self, timeout: float = 3600.0, show_progress: bool = True) -> "ProtocolRun":
        snap = self.progress()
        self.status = snap.get("status", self.status)
        channel_id = snap.get("channel_id", f"telemetry_{self.run_id}")

        if self.status in ("succeeded", "failed", "cancelled"):
            self._check_terminal()
            return self

        _ws_scheme = "wss" if self._client._base.startswith("https://") else "ws"
        domain = (
            self._client._base.replace("https://", "")
            .replace("http://", "")
            .split("/")[0]
        )
        ws_url = f"{_ws_scheme}://{domain}/ws/telemetry/{channel_id}/"
        self._listen_ws(ws_url, show_progress=show_progress, timeout=timeout)
        self._check_terminal()
        return self

    def _listen_ws(self, ws_url: str, show_progress: bool = True, timeout: float = 3600.0) -> None:
        try:
            import asyncio
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required for run.wait(). Install it: pip install websockets"
            )

        async def _listen():
            extra_headers = list(self._client._headers().items())
            async with websockets.connect(
                ws_url,
                additional_headers=extra_headers,
                open_timeout=_DEFAULT_TIMEOUT,
                close_timeout=10,
                ping_interval=8,
                ping_timeout=5,
            ) as ws:
                async for raw in ws:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    env = data.get("data", data) if isinstance(data, dict) else {}
                    payload = env.get("payload", env)
                    status = payload.get("status") or env.get("status")
                    error = payload.get("failure_error") or env.get("failure_error")
                    if status:
                        self.status = status
                    if error:
                        self._failure_error = error
                    if show_progress and status:
                        pct = payload.get("progress_pct") or env.get("progress_pct")
                        pct_str = f" {pct}%" if pct is not None else ""
                        print(f"  [{self.run_id}] {self.status}{pct_str}")
                    if self.status in ("succeeded", "failed", "cancelled"):
                        return

        try:
            asyncio.get_running_loop()
            import nest_asyncio

            nest_asyncio.apply()
        except RuntimeError:
            pass

        try:
            asyncio.run(asyncio.wait_for(_listen(), timeout=timeout))
        except asyncio.TimeoutError:
            try:
                self.refresh()
            except Exception:
                pass
            raise TimeoutError(f"Protocol run {self.run_id} did not complete within {timeout:.0f}s.")

    def _check_terminal(self) -> None:
        if self.status == "failed":
            detail = self._failure_error
            if not detail:
                try:
                    detail = (
                        self._client._get(f"runs/{self.run_id}/").get("failure_error")
                        or "unknown error"
                    )
                except Exception:
                    detail = "unknown error"
            raise ProtocolRunError(f"Protocol run {self.run_id} failed: {detail}")
        if self.status == "cancelled":
            raise ProtocolRunError(f"Protocol run {self.run_id} was cancelled.")

    def results(self) -> Dict[str, Any]:
        return self._client._get(f"runs/{self.run_id}/results/")

    def download(
        self,
        output_dir: Union[str, Path] = ".",
        file_type: str = "csv",
    ) -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        url = self._client._url(f"runs/{self.run_id}/download/?file_type={file_type}")
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT) as http:
            resp = http.get(url, headers=self._client._headers())
        if not resp.is_success:
            raise ProtocolRunError(f"GET {url} returned {resp.status_code}: {resp.text[:500]}")

        fname = output_dir / f"{self.run_id}_results.{file_type}.zip"
        fname.write_bytes(resp.content)
        return str(fname)


class ProtocolClient:
    """Programmatic client for BioLM protocol submission endpoints."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self._api_key = api_key
        self._base = base_url or f"{BIOLMAI_BASE_DOMAIN.rstrip('/')}/api/protocols/"

    def _headers(self) -> Dict[str, str]:
        return _auth_headers(self._api_key)

    def _url(self, path: str) -> str:
        path = path.lstrip("/")
        return self._base.rstrip("/") + "/" + path

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._url(path)
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as http:
            resp = http.get(url, headers=self._headers(), params=params)
        if resp.status_code == 404:
            raise ProtocolNotFoundError(f"Protocol resource not found: {url}")
        if not resp.is_success:
            raise ProtocolRunError(f"GET {url} returned {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def _post(
        self,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self._url(path)
        with httpx.Client(timeout=_UPLOAD_TIMEOUT) as http:
            if files:
                resp = http.post(url, headers=self._headers(), files=files, data=data)
            else:
                resp = http.post(url, headers=self._headers(), json=json_body)
        if not resp.is_success:
            raise ProtocolRunError(f"POST {url} returned {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def list(self, search: Optional[str] = None, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if search:
            params["search"] = search
        return self._get("", params=params)

    def get(self, slug: str, version: Optional[int] = None) -> Dict[str, Any]:
        params = {"version": version} if version is not None else None
        return self._get(f"{slug}/", params=params)

    def submit(
        self,
        slug: str,
        inputs: Dict[str, Any],
        version: Optional[int] = None,
        run_name: Optional[str] = None,
        environment_id: Optional[int] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> ProtocolRun:
        if files:
            form_data: Dict[str, str] = {"inputs": json.dumps(inputs)}
            if version is not None:
                form_data["version"] = str(version)
            if run_name:
                form_data["run_name"] = run_name
            if environment_id is not None:
                form_data["environment_id"] = str(environment_id)
            multipart = {k: (None, v) for k, v in files.items()}
            data = self._post(f"{slug}/runs/", files=multipart, data=form_data)
        else:
            body: Dict[str, Any] = {"inputs": inputs}
            if version is not None:
                body["version"] = version
            if run_name:
                body["run_name"] = run_name
            if environment_id is not None:
                body["environment_id"] = environment_id
            data = self._post(f"{slug}/runs/", json_body=body)
        return ProtocolRun(data, client=self)

    def run_and_wait(
        self,
        slug: str,
        inputs: Dict[str, Any],
        run_name: Optional[str] = None,
        timeout: float = 3600.0,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        run = self.submit(slug, inputs, run_name=run_name)
        run.wait(timeout=timeout, show_progress=show_progress)
        return run.results().get("results", {})

