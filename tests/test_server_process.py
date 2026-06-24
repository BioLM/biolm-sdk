"""Tests for biolm server process management."""
import os
import signal
from unittest.mock import patch

from biolm.server.process import (
    fetch_server_health,
    get_server_runtime_status,
    read_pid_file,
    register_current_process,
    remove_pid_file,
    resolve_server_pid,
    stop_server,
    write_pid_file,
)


def test_pid_file_roundtrip(tmp_path):
    path = tmp_path / "server.pid"
    write_pid_file(4242, "127.0.0.1", 8787, path=path)
    info = read_pid_file(path=path)
    assert info is not None
    assert info.pid == 4242
    assert info.host == "127.0.0.1"
    assert info.port == 8787
    remove_pid_file(path=path)
    assert read_pid_file(path=path) is None


def test_register_current_process_cleans_up_on_exit(tmp_path):
    path = tmp_path / "server.pid"
    callbacks = []

    def capture(fn):
        callbacks.append(fn)

    with patch("biolm.server.process.atexit.register", side_effect=capture):
        register_current_process("127.0.0.1", 8787, path=path)

    assert read_pid_file(path=path).pid == os.getpid()
    assert callbacks
    callbacks[0]()
    assert read_pid_file(path=path) is None


def test_stop_server_uses_pid_file(tmp_path):
    path = tmp_path / "server.pid"
    write_pid_file(99999, "127.0.0.1", 8787, path=path)

    with patch("biolm.server.process.is_process_running", return_value=True), patch(
        "biolm.server.process.os.kill"
    ) as mock_kill:
        pid = stop_server("127.0.0.1", 8787, path=path)

    assert pid == 99999
    mock_kill.assert_called_once_with(99999, signal.SIGTERM)
    assert read_pid_file(path=path) is None


def test_resolve_server_pid_falls_back_to_health_check(tmp_path):
    with patch("biolm.server.process.read_pid_file", return_value=None), patch(
        "biolm.server.process.find_listener_pid", return_value=1234
    ), patch("biolm.server.process.is_biolm_server_running", return_value=True):
        assert resolve_server_pid("127.0.0.1", 8787, path=tmp_path / "missing.pid") == 1234


def test_get_server_runtime_status_when_not_running():
    with patch("biolm.server.process.fetch_server_health", return_value=None):
        status = get_server_runtime_status("127.0.0.1", 8787)
    assert status.running is False
    assert status.pid is None
    assert status.url == "http://127.0.0.1:8787"


def test_get_server_runtime_status_when_running():
    with patch("biolm.server.process.fetch_server_health", return_value={"status": "ok"}), patch(
        "biolm.server.process.resolve_server_pid", return_value=4242
    ):
        status = get_server_runtime_status("127.0.0.1", 8787)
    assert status.running is True
    assert status.pid == 4242


def test_fetch_server_health_returns_payload():
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"status": "ok", "modal_environment": "qa"}

    with patch("biolm.server.process.httpx.get", return_value=FakeResponse()):
        data = fetch_server_health("127.0.0.1", 8787)
    assert data["modal_environment"] == "qa"
