"""Track and stop a running biolm server process."""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

SERVER_PID_PATH = Path(os.path.expanduser("~")) / ".biolm" / "server.pid"


@dataclass
class ServerProcessInfo:
    pid: int
    host: str
    port: int


def _pid_path(path: Optional[Path] = None) -> Path:
    return path or SERVER_PID_PATH


def write_pid_file(pid: int, host: str, port: int, path: Optional[Path] = None) -> None:
    pid_path = _pid_path(path)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n{host}\n{port}\n", encoding="utf-8")


def read_pid_file(path: Optional[Path] = None) -> Optional[ServerProcessInfo]:
    pid_path = _pid_path(path)
    if not pid_path.exists():
        return None
    try:
        lines = pid_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            return None
        return ServerProcessInfo(pid=int(lines[0]), host=lines[1], port=int(lines[2]))
    except (OSError, ValueError):
        return None


def remove_pid_file(path: Optional[Path] = None) -> None:
    pid_path = _pid_path(path)
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def is_biolm_server_running(host: str, port: int) -> bool:
    return fetch_server_health(host, port) is not None


def fetch_server_health(host: str, port: int) -> Optional[dict]:
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=2.0)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "ok":
            return data
    except Exception:
        return None
    return None


def find_listener_pid(port: int) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    for line in out.strip().splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


@dataclass
class ServerRuntimeStatus:
    running: bool
    pid: Optional[int] = None
    url: str = ""


def get_server_runtime_status(host: str, port: int, path: Optional[Path] = None) -> ServerRuntimeStatus:
    """Return whether the local biolm server proxy is listening."""
    url = f"http://{host}:{port}"
    if not is_biolm_server_running(host, port):
        return ServerRuntimeStatus(running=False, url=url)
    pid = resolve_server_pid(host, port, path=path)
    return ServerRuntimeStatus(running=True, pid=pid, url=url)


def resolve_server_pid(host: str, port: int, path: Optional[Path] = None) -> Optional[int]:
    info = read_pid_file(path)
    if info and info.host == host and info.port == port and is_process_running(info.pid):
        return info.pid

    listener = find_listener_pid(port)
    if listener and is_biolm_server_running(host, port):
        return listener
    return None


def register_current_process(host: str, port: int, path: Optional[Path] = None) -> None:
    """Record this process as the active server and clean up on exit."""
    pid = os.getpid()
    write_pid_file(pid, host, port, path=path)

    def _cleanup() -> None:
        info = read_pid_file(path)
        if info and info.pid == pid:
            remove_pid_file(path)

    atexit.register(_cleanup)


def stop_server(host: str, port: int, *, force: bool = False, path: Optional[Path] = None) -> int:
    """Stop the biolm server. Returns the PID that was signaled, or 0 if none found."""
    pid = resolve_server_pid(host, port, path=path)
    if not pid:
        return 0

    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)

    info = read_pid_file(path)
    if info and info.pid == pid:
        remove_pid_file(path)
    return pid
