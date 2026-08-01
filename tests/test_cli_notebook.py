"""Tests for biolm notebook start/stop launcher."""
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from biolm.cli import cli
from biolm.notebook.local import (
    DEFAULT_PORT,
    INSTALL_HINT,
    NotebookDepsError,
    apply_notebook_token_env,
    build_lab_argv,
    check_notebook_dependencies,
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
    make_session,
    pid_is_running,
    save_session,
)


@pytest.fixture
def session_file(tmp_path, monkeypatch):
    path = tmp_path / "notebook-local.json"
    monkeypatch.setattr("biolm.notebook.session.session_path", lambda: path)
    monkeypatch.setattr("biolm.notebook.local.active_session", active_session)
    yield path
    clear_session(path)


def test_resolve_token_env_fills_missing_aliases():
    updates = resolve_notebook_token_env({"BIOLM_TOKEN": "secret"})
    assert updates == {
        "BIOLMAI_TOKEN": "secret",
        "BIOLM_API_KEY": "secret",
    }


def test_resolve_token_env_from_api_key():
    updates = resolve_notebook_token_env({"BIOLM_API_KEY": "ext-key"})
    assert updates == {
        "BIOLM_TOKEN": "ext-key",
        "BIOLMAI_TOKEN": "ext-key",
    }


def test_resolve_token_env_does_not_overwrite():
    env = {
        "BIOLM_TOKEN": "a",
        "BIOLMAI_TOKEN": "b",
        "BIOLM_API_KEY": "",
    }
    updates = resolve_notebook_token_env(env)
    assert updates == {"BIOLM_API_KEY": "a"}
    assert env["BIOLMAI_TOKEN"] == "b"


def test_resolve_token_env_empty_when_unset():
    assert resolve_notebook_token_env({}) == {}


def test_apply_notebook_token_env_mutates_mapping():
    env = {"BIOLM_TOKEN": "tok"}
    applied = apply_notebook_token_env(env)
    assert applied["BIOLM_API_KEY"] == "tok"
    assert env["BIOLMAI_TOKEN"] == "tok"
    assert env["BIOLM_API_KEY"] == "tok"


def test_build_lab_argv_defaults():
    assert build_lab_argv("/usr/bin/jupyter") == [
        "/usr/bin/jupyter",
        "lab",
        "--ServerApp.answer_yes=True",
    ]


def test_build_lab_argv_options_and_extras():
    argv = build_lab_argv(
        "jupyter",
        port=8888,
        notebook_dir="/tmp/nb",
        browser=False,
        extra_args=("--ServerApp.token=",),
    )
    assert argv == [
        "jupyter",
        "lab",
        "--ServerApp.answer_yes=True",
        "--port",
        "8888",
        "--notebook-dir",
        "/tmp/nb",
        "--no-browser",
        "--ServerApp.token=",
    ]


def test_check_notebook_dependencies_reports_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("jupyterlab", "jupyterlab_biolm") or name.startswith(
            ("jupyterlab.", "jupyterlab_biolm.")
        ):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("biolm.notebook.local.shutil.which", lambda _: None)
    monkeypatch.setattr("biolm.notebook.local.os.path.isfile", lambda _: False)

    missing, jupyter_bin = check_notebook_dependencies()
    assert jupyter_bin is None
    assert "jupyterlab" in missing
    assert "jupyterlab-biolm" in missing
    assert "jupyter" in missing


def test_start_foreground_execs_and_saves_session(session_file, monkeypatch):
    calls = []

    def fake_exec(path, argv):
        calls.append((path, list(argv)))

    env = {"BIOLM_TOKEN": "abc"}
    monkeypatch.setattr(
        "biolm.notebook.local.check_notebook_dependencies",
        lambda: ([], "/opt/jupyter"),
    )
    monkeypatch.setattr("biolm.notebook.local.os.getpid", lambda: 4242)
    start_local_notebook(
        port=9999,
        browser=False,
        environ=env,
        exec_fn=fake_exec,
    )
    assert env["BIOLM_API_KEY"] == "abc"
    assert env["BIOLMAI_TOKEN"] == "abc"
    assert env["BIOLM_CLI_THEME"] == "light"
    assert calls == [
        (
            "/opt/jupyter",
            [
                "/opt/jupyter",
                "lab",
                "--ServerApp.answer_yes=True",
                "--port",
                "9999",
                "--no-browser",
            ],
        )
    ]
    session = load_session(session_file)
    assert session is not None
    assert session.pid == 4242
    assert session.port == 9999
    assert session.detached is False


def test_start_defaults_port(session_file, monkeypatch):
    monkeypatch.setattr(
        "biolm.notebook.local.check_notebook_dependencies",
        lambda: ([], "/opt/jupyter"),
    )
    monkeypatch.setattr("biolm.notebook.local.os.getpid", lambda: 1)
    captured = []
    start_local_notebook(
        environ={"BIOLM_TOKEN": "x"},
        exec_fn=lambda path, argv: captured.append(argv),
    )
    assert f"--port" in captured[0]
    assert str(DEFAULT_PORT) in captured[0]


def test_start_preserves_explicit_cli_theme(session_file, monkeypatch):
    monkeypatch.setattr(
        "biolm.notebook.local.check_notebook_dependencies",
        lambda: ([], "/opt/jupyter"),
    )
    monkeypatch.setattr("biolm.notebook.local.os.getpid", lambda: 1)
    env = {"BIOLM_TOKEN": "abc", "BIOLM_CLI_THEME": "dark"}
    start_local_notebook(environ=env, exec_fn=lambda *_: None)
    assert env["BIOLM_CLI_THEME"] == "dark"


def test_start_raises_on_missing_deps(session_file, monkeypatch):
    monkeypatch.setattr(
        "biolm.notebook.local.check_notebook_dependencies",
        lambda: (["jupyterlab"], None),
    )
    with pytest.raises(NotebookDepsError) as excinfo:
        start_local_notebook(exec_fn=lambda *_: None)
    assert "jupyterlab" in str(excinfo.value)
    assert INSTALL_HINT in str(excinfo.value)


def test_start_detach_spawns_and_returns_url(session_file, monkeypatch):
    monkeypatch.setattr(
        "biolm.notebook.local.check_notebook_dependencies",
        lambda: ([], "/opt/jupyter"),
    )
    proc = MagicMock()
    proc.pid = 7777
    opened = []

    url = start_local_notebook(
        detach=True,
        port=8899,
        browser=True,
        environ={"BIOLM_TOKEN": "t"},
        popen_fn=lambda *a, **k: proc,
        open_browser_fn=lambda u: opened.append(u) or True,
    )
    assert url == "http://127.0.0.1:8899/lab"
    assert opened == [url]
    session = load_session(session_file)
    assert session is not None
    assert session.pid == 7777
    assert session.detached is True


def test_start_refuses_when_already_running(session_file, monkeypatch):
    monkeypatch.setattr(
        "biolm.notebook.local.check_notebook_dependencies",
        lambda: ([], "/opt/jupyter"),
    )
    monkeypatch.setattr("biolm.notebook.session.pid_is_running", lambda pid: True)
    save_session(make_session(pid=9, port=8888, detached=True), session_file)
    with pytest.raises(NotebookSessionError, match="already running"):
        start_local_notebook(exec_fn=lambda *_: None, environ={"BIOLM_TOKEN": "t"})


def test_stop_local_notebook(session_file, monkeypatch):
    monkeypatch.setattr("biolm.notebook.session.pid_is_running", lambda pid: True)
    save_session(make_session(pid=555, port=8888, detached=True), session_file)
    killed = []

    def fake_stop(pid, timeout_s=5.0):
        killed.append(pid)
        return True

    monkeypatch.setattr("biolm.notebook.local.stop_process", fake_stop)
    message = stop_local_notebook()
    assert "555" in message
    assert killed == [555]
    assert load_session(session_file) is None


def test_stop_without_session_raises(session_file):
    with pytest.raises(NotebookSessionError, match="No local notebook"):
        stop_local_notebook()


def test_active_session_clears_stale(session_file, monkeypatch):
    monkeypatch.setattr("biolm.notebook.session.pid_is_running", lambda pid: False)
    save_session(make_session(pid=1, port=8888, detached=True), session_file)
    assert active_session(session_file) is None
    assert not session_file.exists()


def test_cli_notebook_help_lists_start_stop():
    result = CliRunner().invoke(cli, ["notebook", "--help"])
    assert result.exit_code == 0, result.output
    assert "start" in result.output
    assert "stop" in result.output


def test_cli_top_level_help_includes_notebook_start_stop():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Notebook" in result.output
    assert "notebook start" in result.output
    assert "notebook stop" in result.output


def test_cli_start_requires_local():
    result = CliRunner().invoke(cli, ["notebook", "start"])
    assert result.exit_code != 0
    assert "--local" in result.output


@patch("biolm.notebook.start_local_notebook")
def test_cli_notebook_start_forwards_options(mock_start, tmp_path):
    mock_start.return_value = None
    result = CliRunner().invoke(
        cli,
        [
            "notebook",
            "start",
            "--local",
            "--port",
            "8889",
            "--dir",
            str(tmp_path),
            "--no-browser",
            "--",
            "--debug",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_start.assert_called_once()
    kwargs = mock_start.call_args.kwargs
    assert kwargs["port"] == 8889
    assert kwargs["notebook_dir"] == str(tmp_path)
    assert kwargs["browser"] is False
    assert kwargs["detach"] is False
    assert kwargs["extra_args"] == ("--debug",)


@patch("biolm.notebook.start_local_notebook")
def test_cli_notebook_start_detach(mock_start):
    mock_start.return_value = "http://127.0.0.1:8888/lab"
    result = CliRunner().invoke(cli, ["notebook", "start", "--local", "-d"])
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8888/lab" in result.output
    assert mock_start.call_args.kwargs["detach"] is True


@patch("biolm.notebook.start_local_notebook")
def test_cli_notebook_start_missing_deps_message(mock_start):
    fake_python = "/tmp/fake-venv/bin/python"
    mock_start.side_effect = NotebookDepsError(
        ["jupyterlab", "jupyterlab-biolm"],
        python_executable=fake_python,
    )
    result = CliRunner().invoke(cli, ["--no-color", "notebook", "start", "--local"])
    assert result.exit_code == 1
    assert "Missing dependencies" in result.output
    assert fake_python in result.output
    assert f"{fake_python} -m pip install 'biolm-sdk[notebook]'" in result.output


@patch("biolm.notebook.stop_local_notebook")
def test_cli_notebook_stop(mock_stop):
    mock_stop.return_value = "Stopped local notebook (pid 1, http://127.0.0.1:8888/lab)"
    result = CliRunner().invoke(cli, ["notebook", "stop", "--local"])
    assert result.exit_code == 0, result.output
    assert "Stopped" in result.output
    mock_stop.assert_called_once()


def test_notebook_install_hint_uses_interpreter():
    from biolm.notebook.local import notebook_install_hint

    hint = notebook_install_hint("/opt/venv/bin/python")
    assert hint == "/opt/venv/bin/python -m pip install 'biolm-sdk[notebook]'"


def test_pid_is_running_current():
    import os

    assert pid_is_running(os.getpid()) is True
