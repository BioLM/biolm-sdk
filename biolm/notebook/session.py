"""Persist local notebook session metadata for start/stop."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from biolm.core.paths import ensure_user_config_dir


SESSION_FILENAME = "notebook-local.json"


class NotebookSessionError(Exception):
    """Raised for local notebook session lifecycle problems."""


@dataclass
class LocalNotebookSession:
    """Record of a local JupyterLab process started by ``biolm notebook``."""

    pid: int
    port: int
    url: str
    detached: bool
    started_at: str
    notebook_dir: Optional[str] = None
    target: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalNotebookSession":
        return cls(
            pid=int(data["pid"]),
            port=int(data["port"]),
            url=str(data["url"]),
            detached=bool(data.get("detached", False)),
            started_at=str(data.get("started_at") or ""),
            notebook_dir=data.get("notebook_dir"),
            target=str(data.get("target") or "local"),
        )


def session_path() -> Path:
    return ensure_user_config_dir() / SESSION_FILENAME


def load_session(path: Optional[Path] = None) -> Optional[LocalNotebookSession]:
    target = path or session_path()
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "pid" not in data:
        return None
    try:
        return LocalNotebookSession.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def save_session(session: LocalNotebookSession, path: Optional[Path] = None) -> Path:
    target = path or session_path()
    target.write_text(
        json.dumps(session.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def clear_session(path: Optional[Path] = None) -> bool:
    target = path or session_path()
    if not target.exists():
        return False
    try:
        target.unlink()
    except OSError:
        return False
    return True


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_session(path: Optional[Path] = None) -> Optional[LocalNotebookSession]:
    """Return the session only if its pid is still alive; clear stale files."""
    session = load_session(path)
    if session is None:
        return None
    if pid_is_running(session.pid):
        return session
    clear_session(path)
    return None


def make_session(
    *,
    pid: int,
    port: int,
    detached: bool,
    notebook_dir: Optional[str] = None,
) -> LocalNotebookSession:
    return LocalNotebookSession(
        pid=pid,
        port=port,
        url=f"http://127.0.0.1:{port}/lab",
        detached=detached,
        started_at=datetime.now(timezone.utc).isoformat(),
        notebook_dir=notebook_dir,
        target="local",
    )


def stop_process(pid: int, *, timeout_s: float = 5.0) -> bool:
    """Send SIGTERM, then SIGKILL if needed. Return True if process is gone."""
    if not pid_is_running(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError as exc:
        raise NotebookSessionError(
            f"Permission denied stopping notebook process {pid}"
        ) from exc

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pid_is_running(pid):
            return True
        time.sleep(0.1)

    if not pid_is_running(pid):
        return True
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError as exc:
        raise NotebookSessionError(
            f"Permission denied killing notebook process {pid}"
        ) from exc

    time.sleep(0.1)
    return not pid_is_running(pid)
