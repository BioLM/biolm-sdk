"""Tests for detached server startup."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from biolm.server.runner import start_detached_server
from biolm.server.settings import ServerSettings


@pytest.fixture
def settings():
    return ServerSettings(
        host="127.0.0.1",
        port=8787,
        auth_mode="none",
        modal_environment="qa",
        refresh_seconds=60,
    )


def test_start_detached_server_rejects_existing_server(settings, tmp_path):
    with patch("biolm.server.runner.is_biolm_server_running", return_value=True):
        with pytest.raises(RuntimeError, match="already running"):
            start_detached_server(settings, pid_path=tmp_path / "server.pid", log_path=tmp_path / "server.log")


def test_start_detached_server_spawns_worker(settings, tmp_path):
    proc = MagicMock()
    proc.pid = 4242
    proc.poll.return_value = None

    with patch("biolm.server.runner.is_biolm_server_running", return_value=False), patch(
        "biolm.server.runner.subprocess.Popen", return_value=proc
    ) as mock_popen, patch(
        "biolm.server.runner.wait_for_server_health", return_value=True
    ), patch("biolm.server.runner.write_pid_file") as mock_write_pid:
        pid = start_detached_server(
            settings,
            pid_path=tmp_path / "server.pid",
            log_path=tmp_path / "server.log",
        )

    assert pid == 4242
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert args[0][-2:] == ["-m", "biolm.server"]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["BIOLM_SERVER_MODAL_ENV"] == "qa"
    mock_write_pid.assert_called_once_with(4242, "127.0.0.1", 8787, path=tmp_path / "server.pid")
