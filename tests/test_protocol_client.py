from __future__ import annotations

"""Tests for ProtocolClient, ProtocolRun, and the top-level run_protocol() function.

HTTP calls are mocked entirely via unittest.mock.patch on httpx.Client so no real
network requests are made. Each _get / _post / _delete helper opens httpx.Client as
a context manager, so the mock hierarchy is:

    mock_client_cls.return_value.__enter__.return_value  →  mock_http (the "session")
    mock_http.get.return_value   / .post.return_value / .delete.return_value
        .status_code, .is_success, .text, .json(), .content

Integration tests (bottom of file):
    TestProtocolClientIntegration — hits a real BioLM server with a real token.
    Skipped unless BIOLMAI_TOKEN and BIOLMAI_BASE_DOMAIN are set.
    Run with:
        BIOLMAI_TOKEN=sk-... BIOLMAI_BASE_DOMAIN=http://localhost:7777 \\
            pytest tests/test_protocol_client.py::TestProtocolClientIntegration -v

Tests are grouped into three classes:
    TestProtocolClient  — auth + list / get / submit / get_run / run_and_wait
    TestProtocolRun     — wait / progress / results / download_files / cancel / refresh
    TestTopLevel        — biolmai.run_protocol()
"""

import io
import os
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from biolmai.protocol_runs import (
    ProtocolClient,
    ProtocolNotFoundError,
    ProtocolRun,
    ProtocolRunError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = "ALY_abc123"
_SLUG = "my-proto"
_TOKEN = "test-token-xyz"

# Minimal ProtocolRun data dict
_RUN_DATA = {
    "run_id": _RUN_ID,
    "protocol_slug": _SLUG,
    "protocol_version": 1,
    "status": "scheduled",
}


def _make_client(token: str = _TOKEN, base_url: str = "https://biolm.ai") -> ProtocolClient:
    """Return a ProtocolClient with a known token, no env dependency."""
    return ProtocolClient(api_key=token, base_url=base_url)


def _mock_response(
    status_code: int = 200,
    json_data: object = None,
    text: str = "",
    content: bytes = b"",
) -> MagicMock:
    """Build a fake httpx.Response-like mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.text = text or (str(json_data) if json_data is not None else "")
    resp.json.return_value = json_data if json_data is not None else {}
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def _patch_httpx_client(method: str, response: MagicMock):
    """Context-manager factory: patches httpx.Client so that ``method`` (get/post/delete)
    returns *response*.  Returns the patch object (use as context manager)."""
    mock_cls = MagicMock()
    mock_http = MagicMock()
    mock_cls.return_value.__enter__.return_value = mock_http
    mock_cls.return_value.__exit__.return_value = False
    getattr(mock_http, method).return_value = response
    return mock_cls, mock_http


# ---------------------------------------------------------------------------
# TestProtocolClient
# ---------------------------------------------------------------------------


class TestProtocolClient:
    """Tests for ProtocolClient auth, list, get, submit, get_run, run_and_wait."""

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def test_no_token_raises(self, monkeypatch):
        """ProtocolClient with no token source raises ValueError with helpful message."""
        monkeypatch.delenv("BIOLMAI_TOKEN", raising=False)
        monkeypatch.delenv("BIOLM_TOKEN", raising=False)
        client = ProtocolClient()  # no api_key= kwarg

        mock_cls, mock_http = _patch_httpx_client("get", _mock_response(200, {}))
        with patch("httpx.Client", mock_cls):
            with pytest.raises(ValueError, match="BIOLM_TOKEN"):
                client._get("")

    def test_token_from_env(self, monkeypatch):
        """BIOLM_TOKEN env var is used in the Authorization header."""
        monkeypatch.delenv("BIOLMAI_TOKEN", raising=False)
        monkeypatch.setenv("BIOLM_TOKEN", "env-token-123")
        client = ProtocolClient()

        resp = _mock_response(200, {"count": 0, "results": []})
        mock_cls, mock_http = _patch_httpx_client("get", resp)
        with patch("httpx.Client", mock_cls):
            client._get("")

        call_kwargs = mock_http.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Token env-token-123"

    def test_token_from_legacy_env(self, monkeypatch):
        """Deprecated BIOLMAI_TOKEN env var is used when BIOLM_TOKEN is unset."""
        monkeypatch.delenv("BIOLM_TOKEN", raising=False)
        monkeypatch.setenv("BIOLMAI_TOKEN", "legacy-env-token-123")
        client = ProtocolClient()

        resp = _mock_response(200, {"count": 0, "results": []})
        mock_cls, mock_http = _patch_httpx_client("get", resp)
        with patch("httpx.Client", mock_cls):
            client._get("")

        call_kwargs = mock_http.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Token legacy-env-token-123"

    def test_token_from_arg(self, monkeypatch):
        """api_key= kwarg takes precedence over BIOLM_TOKEN env var."""
        monkeypatch.delenv("BIOLMAI_TOKEN", raising=False)
        monkeypatch.setenv("BIOLM_TOKEN", "env-token-999")
        client = ProtocolClient(api_key="kwarg-token-456")

        resp = _mock_response(200, {"count": 0, "results": []})
        mock_cls, mock_http = _patch_httpx_client("get", resp)
        with patch("httpx.Client", mock_cls):
            client._get("")

        call_kwargs = mock_http.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Token kwarg-token-456"

    # ------------------------------------------------------------------
    # list()
    # ------------------------------------------------------------------

    def test_list_success(self):
        """list() returns paginated dict from the GET response."""
        payload = {"count": 2, "next": None, "previous": None, "results": [{"slug": "a"}, {"slug": "b"}]}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = client.list()

        assert result["count"] == 2
        assert len(result["results"]) == 2

    def test_list_search_param(self):
        """list(search='antibody') includes ?search=antibody in the request params."""
        client = _make_client()
        resp = _mock_response(200, {"count": 0, "results": []})
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            client.list(search="antibody")

        call_kwargs = mock_http.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params.get("search") == "antibody"

    def test_list_http_error(self):
        """list() raises ProtocolRunError when the server returns 500."""
        client = _make_client()
        resp = _mock_response(500, text="internal server error")
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="500"):
                client.list()

    # ------------------------------------------------------------------
    # get()
    # ------------------------------------------------------------------

    def test_get_success(self):
        """get() returns protocol detail with inputs_schema present."""
        payload = {
            "slug": _SLUG,
            "version": 1,
            "inputs_schema": {"sequence": {"type": "text", "required": True}},
        }
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = client.get(_SLUG)

        assert "inputs_schema" in result
        assert result["slug"] == _SLUG

    def test_get_with_version(self):
        """get(slug, version=2) passes ?version=2 in the request params."""
        client = _make_client()
        resp = _mock_response(200, {"slug": _SLUG, "version": 2, "inputs_schema": {}})
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            client.get(_SLUG, version=2)

        call_kwargs = mock_http.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params.get("version") == 2

    def test_get_404(self):
        """get() raises ProtocolNotFoundError on 404."""
        client = _make_client()
        resp = _mock_response(404, text="not found")
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolNotFoundError):
                client.get("nonexistent-slug")

    # ------------------------------------------------------------------
    # submit()
    # ------------------------------------------------------------------

    def test_submit_returns_protocol_run(self):
        """submit() returns a ProtocolRun with run_id, protocol_slug, and status."""
        payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("post", resp)

        with patch("httpx.Client", mock_cls):
            run = client.submit(_SLUG, inputs={"sequence": "MKTAY"})

        assert isinstance(run, ProtocolRun)
        assert run.run_id == _RUN_ID
        assert run.protocol_slug == _SLUG
        assert run.status == "scheduled"

    def test_submit_json_body(self):
        """submit() sends JSON body containing 'inputs' (and optional run_name)."""
        import json as _json

        payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, mock_http = _patch_httpx_client("post", resp)

        with patch("httpx.Client", mock_cls):
            client.submit(_SLUG, inputs={"sequence": "MKTAY"}, run_name="my-run")

        call_kwargs = mock_http.post.call_args
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json", {})
        assert "inputs" in sent_json
        assert sent_json["inputs"] == {"sequence": "MKTAY"}
        assert sent_json.get("run_name") == "my-run"

    def test_submit_with_files(self):
        """submit(files=...) sends a multipart request (files= kwarg, not json=)."""
        payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, mock_http = _patch_httpx_client("post", resp)

        fake_file = io.BytesIO(b"ATOM 1 ...")
        with patch("httpx.Client", mock_cls):
            client.submit(_SLUG, inputs={"sequence": "MKTAY"}, files={"pdb_file": fake_file})

        call_kwargs = mock_http.post.call_args
        # Multipart uses files= not json=
        assert call_kwargs.kwargs.get("json") is None
        assert call_kwargs.kwargs.get("files") is not None or call_kwargs[1].get("files") is not None

    def test_submit_error_response(self):
        """submit() raises ProtocolRunError when the server returns 400."""
        client = _make_client()
        resp = _mock_response(400, text="invalid inputs")
        mock_cls, _ = _patch_httpx_client("post", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="400"):
                client.submit(_SLUG, inputs={"bad": "field"})

    # ------------------------------------------------------------------
    # get_run()
    # ------------------------------------------------------------------

    def test_get_run_reconnects(self):
        """get_run() returns a ProtocolRun with correct run_id and status."""
        payload = {
            "run_id": _RUN_ID,
            "protocol_slug": _SLUG,
            "protocol_version": 2,
            "status": "succeeded",
        }
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            run = client.get_run(_RUN_ID)

        assert isinstance(run, ProtocolRun)
        assert run.run_id == _RUN_ID
        assert run.status == "succeeded"

    # ------------------------------------------------------------------
    # run_and_wait()
    # ------------------------------------------------------------------

    def test_run_and_wait_end_to_end(self):
        """run_and_wait() submits, polls progress until succeeded, then returns results."""
        client = _make_client()

        submit_payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        results_payload = {
            "run_id": _RUN_ID,
            "status": "succeeded",
            "results": {"designed_sequences": ["MKTAY"]},
        }

        # We mock at the ProtocolRun level to avoid multi-layer HTTP mock complexity
        with patch.object(ProtocolClient, "submit") as mock_submit:
            mock_run = MagicMock(spec=ProtocolRun)
            mock_run.wait.return_value = mock_run
            mock_run.results.return_value = results_payload
            mock_submit.return_value = mock_run

            result = client.run_and_wait(_SLUG, inputs={"sequence": "MKTAY"}, show_progress=False)

        mock_submit.assert_called_once_with(_SLUG, {"sequence": "MKTAY"}, run_name=None)
        mock_run.wait.assert_called_once()
        mock_run.results.assert_called_once()
        assert result == results_payload.get("results", {})


# ---------------------------------------------------------------------------
# TestProtocolRun
# ---------------------------------------------------------------------------


class TestProtocolRun:
    """Tests for ProtocolRun methods: wait, progress, results, download_files, cancel, refresh."""

    def _make_run(self, status: str = "scheduled") -> tuple[ProtocolRun, ProtocolClient]:
        client = _make_client()
        data = dict(_RUN_DATA, status=status)
        run = ProtocolRun(data, client=client)
        return run, client

    # ------------------------------------------------------------------
    # progress()
    # ------------------------------------------------------------------

    def test_progress_returns_dict(self):
        """progress() calls GET /runs/{run_id}/progress/ and returns the dict."""
        run, client = self._make_run()
        progress_payload = {
            "status": "running",
            "progress_pct": 42,
            "tasks": [],
            "log_lines": ["starting..."],
            "result_row_count": None,
        }
        resp = _mock_response(200, progress_payload)
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = run.progress()

        assert result["status"] == "running"
        assert result["progress_pct"] == 42
        # URL should contain 'progress'
        called_url = mock_http.get.call_args[0][0]
        assert "progress" in called_url

    # ------------------------------------------------------------------
    # results()
    # ------------------------------------------------------------------

    def test_results_returns_detail(self):
        """results() calls GET /runs/{run_id}/ and returns the full detail dict."""
        run, client = self._make_run(status="succeeded")
        detail_payload = {
            "run_id": _RUN_ID,
            "status": "succeeded",
            "results": {"output": "data"},
            "failure_error": None,
        }
        resp = _mock_response(200, detail_payload)
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = run.results()

        assert result["run_id"] == _RUN_ID
        assert result["results"] == {"output": "data"}
        # URL must NOT contain 'progress' — it's the plain run detail endpoint
        called_url = mock_http.get.call_args[0][0]
        assert "progress" not in called_url

    # ------------------------------------------------------------------
    # refresh()
    # ------------------------------------------------------------------

    def test_refresh_updates_status(self):
        """refresh() fetches run detail and updates self.status in place."""
        run, _ = self._make_run(status="scheduled")
        assert run.status == "scheduled"

        resp = _mock_response(200, {"run_id": _RUN_ID, "status": "running"})
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            returned = run.refresh()

        assert run.status == "running"
        assert returned is run  # returns self

    # ------------------------------------------------------------------
    # cancel()
    # ------------------------------------------------------------------

    def test_cancel_sets_status(self):
        """cancel() calls DELETE and sets run.status = 'cancelled'."""
        run, _ = self._make_run(status="running")
        resp = _mock_response(200, {"detail": "cancelled"})
        mock_cls, mock_http = _patch_httpx_client("delete", resp)

        with patch("httpx.Client", mock_cls):
            run.cancel()

        assert run.status == "cancelled"
        assert mock_http.delete.called

    def test_cancel_http_error(self):
        """cancel() raises ProtocolRunError when the server returns 400."""
        run, _ = self._make_run(status="succeeded")
        resp = _mock_response(400, text="already in terminal state")
        mock_cls, _ = _patch_httpx_client("delete", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="400"):
                run.cancel()

    # ------------------------------------------------------------------
    # wait()
    # ------------------------------------------------------------------

    def test_wait_already_terminal_skips_ws(self):
        """wait() does not call _listen_ws() when run is already in a terminal state at entry."""
        run, _ = self._make_run()

        progress_snap = {"status": "succeeded", "channel_id": "telemetry_abc"}

        with patch.object(run, "progress", return_value=progress_snap), \
             patch.object(run, "_listen_ws") as mock_ws:
            result = run.wait(show_progress=False)

        mock_ws.assert_not_called()
        assert result is run
        assert run.status == "succeeded"

    def test_wait_via_ws_until_succeeded(self):
        """wait() calls progress() once for channel_id, then _listen_ws(), returns self."""
        run, _ = self._make_run()

        progress_snap = {"status": "running", "progress_pct": 50, "channel_id": "telemetry_abc"}

        def fake_listen(ws_url, show_progress=True, timeout=3600.0):
            run.status = "succeeded"

        with patch.object(run, "progress", return_value=progress_snap), \
             patch.object(run, "_listen_ws", side_effect=fake_listen):
            result = run.wait(show_progress=False)

        assert result is run
        assert run.status == "succeeded"

    def test_wait_raises_on_failed(self):
        """wait() raises ProtocolRunError when progress() returns a failed status."""
        run, _ = self._make_run()

        progress_snap = {"status": "failed", "channel_id": "telemetry_abc"}
        detail_resp = _mock_response(200, {"run_id": _RUN_ID, "status": "failed", "failure_error": "OOM"})
        mock_cls, _ = _patch_httpx_client("get", detail_resp)

        with patch.object(run, "progress", return_value=progress_snap), \
             patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="failed"):
                run.wait(show_progress=False)

    def test_wait_raises_on_cancelled(self):
        """wait() raises ProtocolRunError when progress() returns a cancelled status."""
        run, _ = self._make_run()

        progress_snap = {"status": "cancelled", "channel_id": "telemetry_abc"}

        with patch.object(run, "progress", return_value=progress_snap):
            with pytest.raises(ProtocolRunError, match="cancelled"):
                run.wait(show_progress=False)

    def test_wait_ws_loss_raises(self):
        """wait() raises ProtocolRunError if _listen_ws raises before terminal status."""
        run, _ = self._make_run()

        progress_snap = {"status": "running", "channel_id": "telemetry_abc"}

        def fake_listen(ws_url, show_progress=True, timeout=3600.0):
            raise ProtocolRunError(
                f"WebSocket connection lost for run {run.run_id}: connection refused. "
                "Last known status: running"
            )

        with patch.object(run, "progress", return_value=progress_snap), \
             patch.object(run, "_listen_ws", side_effect=fake_listen):
            with pytest.raises(ProtocolRunError, match="WebSocket connection lost"):
                run.wait(show_progress=False)

    def test_wait_no_progress_output(self, capsys):
        """wait(show_progress=False) prints nothing to stdout."""
        run, _ = self._make_run()

        progress_snap = {"status": "running", "channel_id": "telemetry_abc"}

        def fake_listen(ws_url, show_progress=True, timeout=3600.0):
            run.status = "succeeded"

        with patch.object(run, "progress", return_value=progress_snap), \
             patch.object(run, "_listen_ws", side_effect=fake_listen):
            run.wait(show_progress=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_wait_prints_progress(self, capsys):
        """wait(show_progress=True) passes show_progress=True into _listen_ws."""
        run, _ = self._make_run()

        progress_snap = {"status": "running", "progress_pct": 25, "channel_id": "telemetry_abc"}

        def fake_listen(ws_url, show_progress=True, timeout=3600.0):
            if show_progress:
                print(f"  [{run.run_id}] running 25%")
            run.status = "succeeded"

        with patch.object(run, "progress", return_value=progress_snap), \
             patch.object(run, "_listen_ws", side_effect=fake_listen):
            run.wait(show_progress=True)

        captured = capsys.readouterr()
        assert _RUN_ID in captured.out
        assert "25" in captured.out

    # ------------------------------------------------------------------
    # to_dataframe()
    # ------------------------------------------------------------------

    def test_to_dataframe_happy_path(self, tmp_path):
        """to_dataframe() downloads the zip, parses the CSV, and returns a DataFrame."""
        run, _ = self._make_run(status="succeeded")
        zip_path = tmp_path / f"{_RUN_ID}_results.csv.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("results.csv", "sequence,score\nMKTAY,0.9\nMKLAY,0.7\n")

        with patch.object(run, "download", return_value=zip_path):
            df = run.to_dataframe(output_dir=str(tmp_path))

        assert list(df.columns) == ["sequence", "score"]
        assert len(df) == 2
        assert df["sequence"].tolist() == ["MKTAY", "MKLAY"]

    def test_to_dataframe_no_pandas(self, tmp_path):
        """to_dataframe() raises ImportError with helpful message when pandas is missing."""
        import sys

        run, _ = self._make_run(status="succeeded")

        with patch.dict(sys.modules, {"pandas": None}):
            with pytest.raises(ImportError, match="pandas"):
                run.to_dataframe(output_dir=str(tmp_path))

    def test_to_dataframe_no_csv_in_zip(self, tmp_path):
        """to_dataframe() raises ProtocolRunError when the zip contains no CSV file."""
        run, _ = self._make_run(status="succeeded")
        zip_path = tmp_path / f"{_RUN_ID}_results.csv.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("results.jsonl", '{"sequence": "MKTAY"}\n')

        with patch.object(run, "download", return_value=zip_path):
            with pytest.raises(ProtocolRunError, match="No CSV"):
                run.to_dataframe(output_dir=str(tmp_path))

    # ------------------------------------------------------------------
    # download_files() — simple alias for download()
    # ------------------------------------------------------------------

    def test_download_files_is_alias_for_download(self, tmp_path):
        """download_files() calls download() with the same args and returns its result."""
        run, _ = self._make_run(status="succeeded")
        expected_path = tmp_path / f"{_RUN_ID}_results.csv.zip"

        with patch.object(run, "download", return_value=expected_path) as mock_dl:
            result = run.download_files(output_dir=str(tmp_path), file_type="csv", overwrite=False)

        mock_dl.assert_called_once_with(output_dir=str(tmp_path), file_type="csv", overwrite=False)
        assert result == expected_path


# ---------------------------------------------------------------------------
# TestTopLevel
# ---------------------------------------------------------------------------


class TestTopLevel:
    """Tests for the top-level biolmai.run_protocol() convenience function."""

    def test_top_level_convenience(self, monkeypatch):
        """run_protocol() builds a ProtocolClient, calls run_and_wait, returns results."""
        import biolmai

        monkeypatch.setenv("BIOLMAI_TOKEN", _TOKEN)

        expected_results = {"designed_sequences": ["MKTAY", "MKLAY"]}

        with patch.object(ProtocolClient, "run_and_wait", return_value=expected_results) as mock_raw:
            result = biolmai.run_protocol(
                _SLUG,
                inputs={"sequence": "MKTAY"},
                run_name="top-level-test",
                api_key=_TOKEN,
                show_progress=False,
            )

        assert result == expected_results
        mock_raw.assert_called_once_with(
            _SLUG,
            {"sequence": "MKTAY"},
            run_name="top-level-test",
            timeout=3600.0,
            show_progress=False,
        )


# ===========================================================================
# Integration tests — skipped unless BIOLMAI_TOKEN is set
#
# These make REAL HTTP requests to a live BioLM server (local or prod).
# No mocking. The full client → server → Celery → Modal → results path is
# exercised.
#
# Usage (against local dev stack):
#   BIOLMAI_TOKEN=your-token BIOLMAI_BASE_DOMAIN=http://localhost:7777 \
#       pytest tests/test_protocol_client.py::TestProtocolClientIntegration -v
#
# Usage (against prod):
#   BIOLMAI_TOKEN=your-token \
#       pytest tests/test_protocol_client.py::TestProtocolClientIntegration -v
#
# Required env vars:
#   BIOLMAI_TOKEN              — Knox API token for an account with protocol access
#   BIOLMAI_BASE_DOMAIN        — base URL (default: https://biolm.ai)
#
# Optional env vars:
#   INTEGRATION_PROTOCOL_SLUG  — protocol slug to test (default: test-echo-protocol)
#   INTEGRATION_POLL_TIMEOUT   — max seconds to wait for completion (default: 300)
# ===========================================================================

import os
import pytest


def _integration_enabled():
    """Both an explicit opt-in flag AND a token must be present."""
    has_token = bool(os.environ.get("BIOLMAI_TOKEN") or os.environ.get("BIOLM_TOKEN"))
    opted_in = os.environ.get("BIOLMAI_INTEGRATION") == "1"
    return has_token and opted_in


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="Skipped: set BIOLMAI_TOKEN and BIOLMAI_INTEGRATION=1 to run against a live server",
)
class TestProtocolClientIntegration:
    """End-to-end integration tests using a real ProtocolClient against a live server.

    No HTTP mocking. Tests submit real protocol runs, poll /progress/ until
    terminal, and verify results are returned. Requires biolm_web PR #161 to
    be merged and the target protocol to be published.
    """

    _SLUG = os.environ.get("INTEGRATION_PROTOCOL_SLUG", "test-echo-protocol")
    _TIMEOUT = int(os.environ.get("INTEGRATION_POLL_TIMEOUT", 300))

    @pytest.fixture(autouse=True)
    def client(self):
        from biolmai import ProtocolClient
        base_url = os.environ.get("BIOLMAI_BASE_DOMAIN")
        kwargs = {"api_key": _integration_token()}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = ProtocolClient(**kwargs)

    def test_list_returns_results(self):
        """client.list() hits the real server and returns a valid paginated response."""
        result = self._client.list()
        assert "results" in result
        assert "count" in result
        assert isinstance(result["results"], list)

    def test_get_protocol_schema(self):
        """client.get(slug) returns inputs_schema for the target protocol.

        Skip gracefully if the protocol doesn't exist on this server.
        """
        from biolmai import ProtocolNotFoundError
        try:
            schema = self._client.get(self._SLUG)
        except ProtocolNotFoundError:
            pytest.skip(
                f"Protocol '{self._SLUG}' not found on this server. "
                "Publish it via the console or set INTEGRATION_PROTOCOL_SLUG."
            )
        assert "inputs_schema" in schema
        assert schema["slug"] == self._SLUG

    def test_submit_and_wait_for_completion(self):
        """Submit a real run and poll until succeeded. Verify results are populated.

        Uses the protocol's own inputs_schema to build minimal valid inputs,
        so this test stays protocol-agnostic.
        """
        from biolmai import ProtocolNotFoundError, ProtocolRunError

        # Fetch schema to build inputs
        try:
            schema = self._client.get(self._SLUG)
        except ProtocolNotFoundError:
            pytest.skip(f"Protocol '{self._SLUG}' not found — cannot submit.")

        # Build minimal inputs: use initial/default values from schema where available,
        # otherwise skip required fields that have no default (test will fail clearly)
        inputs = {}
        for field, spec in schema.get("inputs_schema", {}).items():
            if "initial" in spec:
                inputs[field] = spec["initial"]
            elif spec.get("required"):
                pytest.skip(
                    f"Required field '{field}' has no default — provide a test protocol "
                    "with example_inputs or set INTEGRATION_PROTOCOL_SLUG to one that does."
                )

        # Submit
        run = self._client.submit(self._SLUG, inputs=inputs, run_name="sdk-integration-test")
        assert run.run_id.startswith("ALY_"), f"Unexpected run_id: {run.run_id}"
        assert run.status in ("scheduled", "running")

        # Wait for completion
        try:
            run.wait(timeout=self._TIMEOUT, show_progress=True)
        except ProtocolRunError as exc:
            pytest.fail(f"Run {run.run_id} failed: {exc}")

        assert run.status == "succeeded"

        # Verify results
        detail = run.results()
        assert detail["status"] == "succeeded"
        assert detail.get("results") is not None, (
            f"Run {run.run_id} succeeded but results is null — "
            "protocol may not be writing return_json."
        )

    def test_progress_shape_during_run(self):
        """Progress endpoint returns correctly shaped data for a live run."""
        from biolmai import ProtocolNotFoundError

        try:
            schema = self._client.get(self._SLUG)
        except ProtocolNotFoundError:
            pytest.skip(f"Protocol '{self._SLUG}' not found.")

        inputs = {
            field: spec["initial"]
            for field, spec in schema.get("inputs_schema", {}).items()
            if "initial" in spec
        }

        run = self._client.submit(self._SLUG, inputs=inputs)

        # Poll once immediately — run may be scheduled or already running
        prog = run.progress()
        assert "status" in prog
        assert "tasks" in prog
        assert "log_lines" in prog
        assert "progress_pct" in prog
        assert isinstance(prog["tasks"], list)
        assert isinstance(prog["log_lines"], list)

        # Clean up — wait or cancel
        if prog["status"] not in ("succeeded", "failed", "cancelled"):
            try:
                run.wait(timeout=self._TIMEOUT, show_progress=False)
            except Exception:
                run.cancel()

    def test_get_run_reconnect(self):
        """client.get_run(run_id) reconnects to a prior run and returns current status."""
        from biolmai import ProtocolNotFoundError

        try:
            schema = self._client.get(self._SLUG)
        except ProtocolNotFoundError:
            pytest.skip(f"Protocol '{self._SLUG}' not found.")

        inputs = {
            field: spec["initial"]
            for field, spec in schema.get("inputs_schema", {}).items()
            if "initial" in spec
        }

        # Submit, then reconnect by run_id
        original = self._client.submit(self._SLUG, inputs=inputs)
        run_id = original.run_id

        reconnected = self._client.get_run(run_id)
        assert reconnected.run_id == run_id
        assert reconnected.status in ("scheduled", "running", "succeeded", "failed", "cancelled")

        # Clean up
        try:
            reconnected.wait(timeout=self._TIMEOUT, show_progress=False)
        except Exception:
            reconnected.cancel()
