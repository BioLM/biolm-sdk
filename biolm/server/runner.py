"""Start and run the biolm server process."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from biolm.server.process import (
    SERVER_PID_PATH,
    fetch_server_health,
    is_biolm_server_running,
    register_current_process,
    write_pid_file,
)
from biolm.server.settings import ServerSettings

SERVER_LOG_PATH = Path(os.path.expanduser("~")) / ".biolm" / "server.log"
HEALTH_WAIT_SECONDS = 120


def run_server_foreground(settings: ServerSettings) -> None:
    """Run uvicorn in the current process."""
    import uvicorn

    from biolm.server.app import create_app

    settings.validate()
    app = create_app(settings)
    register_current_process(settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


def _settings_env(settings: ServerSettings) -> dict[str, str]:
    env = {
        "BIOLM_SERVER_HOST": settings.host,
        "BIOLM_SERVER_PORT": str(settings.port),
        "BIOLM_SERVER_AUTH": settings.auth_mode,
        "BIOLM_SERVER_REFRESH_SECONDS": str(settings.refresh_seconds),
        "BIOLM_SERVER_MODAL_ENV": settings.modal_environment,
    }
    if settings.server_token:
        env["BIOLM_SERVER_TOKEN"] = settings.server_token
    if settings.models_env:
        env["BIOLM_SERVER_MODELS"] = settings.models_env
    if settings.config_path:
        env["BIOLM_SERVER_CONFIG_PATH"] = settings.config_path
    return env


def wait_for_server_health(host: str, port: int, timeout: float = HEALTH_WAIT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fetch_server_health(host, port):
            return True
        time.sleep(0.5)
    return False


def start_detached_server(
    settings: ServerSettings,
    *,
    log_path: Optional[Path] = None,
    pid_path: Optional[Path] = None,
) -> int:
    """Spawn a background server process and wait until /health responds."""
    settings.validate()
    if is_biolm_server_running(settings.host, settings.port):
        raise RuntimeError(f"biolm server already running at {settings.base_url}")

    log_file_path = log_path or SERVER_LOG_PATH
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(_settings_env(settings))

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "biolm.server"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    write_pid_file(proc.pid, settings.host, settings.port, path=pid_path or SERVER_PID_PATH)

    if proc.poll() is not None:
        raise RuntimeError(
            f"Detached server exited immediately (code {proc.returncode}). "
            f"See {log_file_path}"
        )

    if not wait_for_server_health(settings.host, settings.port):
        if proc.poll() is None:
            proc.terminate()
        raise RuntimeError(
            f"Detached server failed health check within {HEALTH_WAIT_SECONDS}s. "
            f"See {log_file_path}"
        )

    return proc.pid
