"""Launch and stop a local JupyterLab session with BioLM auth env normalized."""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

from biolm.notebook.session import (
    NotebookSessionError,
    active_session,
    clear_session,
    make_session,
    save_session,
    stop_process,
)


TOKEN_ENV_KEYS = ("BIOLM_TOKEN", "BIOLMAI_TOKEN", "BIOLM_API_KEY")

INSTALL_HINT = "pip install 'biolm-sdk[notebook]'"

DEFAULT_PORT = 8888


def notebook_install_hint(python_executable: Optional[str] = None) -> str:
    """Return an install command pinned to the active interpreter."""
    import sys

    exe = python_executable or sys.executable
    return f"{exe} -m pip install 'biolm-sdk[notebook]'"


class NotebookDepsError(Exception):
    """Raised when JupyterLab / jupyterlab-biolm are not available."""

    def __init__(self, missing: Sequence[str], python_executable: Optional[str] = None):
        import sys

        self.missing = list(missing)
        self.python_executable = python_executable or sys.executable
        self.install_hint = notebook_install_hint(self.python_executable)
        missing_list = ", ".join(self.missing)
        super().__init__(
            f"Missing notebook dependencies: {missing_list}. "
            f"Install with: {self.install_hint}"
        )


def resolve_notebook_token_env(
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return env updates so SDK, legacy, and extension token names agree.

    If any of ``BIOLM_TOKEN``, ``BIOLMAI_TOKEN``, or ``BIOLM_API_KEY`` is set,
    fill the unset ones with that value. Existing values are never overwritten.
    When none are set, returns ``{}`` (credentials file / in-Lab Settings still work).
    """
    env = os.environ if environ is None else environ
    token = ""
    for key in TOKEN_ENV_KEYS:
        value = env.get(key) or ""
        if value:
            token = value
            break
    if not token:
        return {}
    updates: Dict[str, str] = {}
    for key in TOKEN_ENV_KEYS:
        if not (env.get(key) or ""):
            updates[key] = token
    return updates


def apply_notebook_token_env(
    environ: Optional[MutableMapping[str, str]] = None,
) -> Dict[str, str]:
    """Apply :func:`resolve_notebook_token_env` to ``environ`` (default ``os.environ``)."""
    target: MutableMapping[str, str] = os.environ if environ is None else environ
    updates = resolve_notebook_token_env(target)
    target.update(updates)
    return updates


def check_notebook_dependencies() -> tuple[List[str], Optional[str]]:
    """Return ``(missing_names, jupyter_executable)``.

    ``missing_names`` is empty when JupyterLab, jupyterlab-biolm, and the
    ``jupyter`` CLI are all available.

    Prefers ``jupyter`` next to ``sys.executable`` so a different env's
    ``jupyter`` on PATH does not mask a venv missing Lab packages.
    """
    import sys

    missing: List[str] = []
    try:
        import jupyterlab  # noqa: F401
    except ImportError:
        missing.append("jupyterlab")
    try:
        import jupyterlab_biolm  # noqa: F401
    except ImportError:
        missing.append("jupyterlab-biolm")

    venv_jupyter = os.path.join(os.path.dirname(sys.executable), "jupyter")
    if os.path.isfile(venv_jupyter) and os.access(venv_jupyter, os.X_OK):
        jupyter_bin: Optional[str] = venv_jupyter
    else:
        jupyter_bin = shutil.which("jupyter")
    if not jupyter_bin:
        missing.append("jupyter")
    return missing, jupyter_bin


def build_lab_argv(
    jupyter_bin: str,
    *,
    port: Optional[int] = None,
    notebook_dir: Optional[str] = None,
    browser: bool = True,
    extra_args: Sequence[str] = (),
) -> List[str]:
    """Build ``jupyter lab ...`` argv (including executable as argv[0])."""
    argv: List[str] = [
        jupyter_bin,
        "lab",
        # Answer yes to the Ctrl-C shutdown prompt so the session exits cleanly.
        "--ServerApp.answer_yes=True",
    ]
    if port is not None:
        argv.extend(["--port", str(port)])
    if notebook_dir:
        argv.extend(["--notebook-dir", str(notebook_dir)])
    if not browser:
        argv.append("--no-browser")
    argv.extend(extra_args)
    return argv


def apply_notebook_session_env(
    environ: Optional[MutableMapping[str, str]] = None,
) -> Dict[str, str]:
    """Apply token normalization and Lab-oriented CLI theme defaults.

    JupyterLab's default UI/terminal is light; seed ``BIOLM_CLI_THEME=light``
    when unset so SDK help inside Lab terminals stays readable. Explicit user
    overrides are preserved.
    """
    target: MutableMapping[str, str] = os.environ if environ is None else environ
    updates = resolve_notebook_token_env(target)
    if not (target.get("BIOLM_CLI_THEME") or "").strip():
        updates["BIOLM_CLI_THEME"] = "light"
    target.update(updates)
    return updates


def _prepare_launch(
    *,
    port: Optional[int],
    notebook_dir: Optional[str],
    browser: bool,
    extra_args: Sequence[str],
    environ: Optional[MutableMapping[str, str]],
) -> tuple[List[str], int, MutableMapping[str, str]]:
    """Normalize env, check deps, and build argv. Return ``(argv, port, env)``."""
    env: MutableMapping[str, str]
    if environ is None:
        apply_notebook_session_env(os.environ)
        env = os.environ
    else:
        apply_notebook_session_env(environ)
        env = environ

    missing, jupyter_bin = check_notebook_dependencies()
    if missing or not jupyter_bin:
        raise NotebookDepsError(missing or ["jupyter"])

    resolved_port = DEFAULT_PORT if port is None else port
    existing = active_session()
    if existing is not None:
        raise NotebookSessionError(
            f"A local notebook is already running (pid {existing.pid}, "
            f"{existing.url}). Stop it with: biolm notebook stop --local"
        )

    argv = build_lab_argv(
        jupyter_bin,
        port=resolved_port,
        notebook_dir=notebook_dir,
        browser=browser,
        extra_args=extra_args,
    )
    return argv, resolved_port, env


def start_local_notebook(
    *,
    port: Optional[int] = None,
    notebook_dir: Optional[str] = None,
    browser: bool = True,
    detach: bool = False,
    extra_args: Sequence[str] = (),
    environ: Optional[MutableMapping[str, str]] = None,
    exec_fn: Optional[Callable[[str, Sequence[str]], None]] = None,
    popen_fn: Optional[Callable[..., subprocess.Popen]] = None,
    open_browser_fn: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Start a local JupyterLab session.

    Foreground (default): write session metadata for the current pid, then
    replace the process with JupyterLab via ``exec_fn`` (default ``os.execvp``).
    Returns ``None`` because control does not return on success.

    Detached (``detach=True``): spawn JupyterLab in the background, persist
    session metadata, optionally open a browser, and return the Lab URL.
    """
    argv, resolved_port, env = _prepare_launch(
        port=port,
        notebook_dir=notebook_dir,
        browser=browser if not detach else False,
        extra_args=extra_args,
        environ=environ,
    )

    if detach:
        runner = popen_fn or subprocess.Popen
        # Pass a concrete env mapping; os.environ is fine when environ is None.
        child_env = dict(env)
        proc = runner(
            argv,
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        session = make_session(
            pid=proc.pid,
            port=resolved_port,
            detached=True,
            notebook_dir=notebook_dir,
        )
        save_session(session)
        if browser:
            opener = open_browser_fn or webbrowser.open
            opener(session.url)
        return session.url

    # Foreground: session pid becomes Jupyter after exec (same process id).
    session = make_session(
        pid=os.getpid(),
        port=resolved_port,
        detached=False,
        notebook_dir=notebook_dir,
    )
    save_session(session)
    runner = exec_fn or os.execvp
    runner(argv[0], argv)
    return None


def stop_local_notebook(
    *,
    timeout_s: float = 5.0,
) -> Optional[str]:
    """Stop the tracked local notebook session.

    Returns a short status message. Raises ``NotebookSessionError`` when no
    active session exists.
    """
    session = active_session()
    if session is None:
        # Clear any stale file just in case.
        clear_session()
        raise NotebookSessionError(
            "No local notebook session is running. "
            "Start one with: biolm notebook start --local"
        )

    stopped = stop_process(session.pid, timeout_s=timeout_s)
    clear_session()
    if not stopped:
        raise NotebookSessionError(
            f"Could not stop notebook process {session.pid}"
        )
    return f"Stopped local notebook (pid {session.pid}, {session.url})"


# Backwards-compatible alias used by earlier drafts / tests.
def launch_local_notebook(**kwargs):
    """Alias for :func:`start_local_notebook` (foreground by default)."""
    kwargs.setdefault("detach", False)
    return start_local_notebook(**kwargs)
