"""Local JupyterLab session bootstrap for BioLM.

Install::

    pip install 'biolm-sdk[notebook]'

Then::

    biolm notebook start --local
    biolm notebook start --local -d
    biolm notebook stop --local
"""

from __future__ import annotations

from biolm.notebook.local import (
    DEFAULT_PORT,
    INSTALL_HINT,
    NotebookDepsError,
    apply_notebook_session_env,
    apply_notebook_token_env,
    build_lab_argv,
    check_notebook_dependencies,
    launch_local_notebook,
    notebook_install_hint,
    resolve_notebook_token_env,
    start_local_notebook,
    stop_local_notebook,
)
from biolm.notebook.session import (
    LocalNotebookSession,
    NotebookSessionError,
    active_session,
    clear_session,
    load_session,
    save_session,
)

__all__ = [
    "DEFAULT_PORT",
    "INSTALL_HINT",
    "LocalNotebookSession",
    "NotebookDepsError",
    "NotebookSessionError",
    "active_session",
    "apply_notebook_session_env",
    "apply_notebook_token_env",
    "build_lab_argv",
    "check_notebook_dependencies",
    "clear_session",
    "launch_local_notebook",
    "load_session",
    "notebook_install_hint",
    "resolve_notebook_token_env",
    "save_session",
    "start_local_notebook",
    "stop_local_notebook",
]
