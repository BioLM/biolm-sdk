"""Tests for terminal-aware CLI theming."""
import os

import pytest

from biolm.cli.theme import (
    build_theme,
    create_console,
    no_color_requested,
    resolve_theme_mode,
    terminal_is_dark,
)


def test_no_color_when_env_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert no_color_requested() is True
    console = create_console()
    assert console.no_color is True
    # Custom style names must still resolve (markup must not crash)
    assert console.get_style("brand") is not None


def test_resolve_theme_mode_from_env(monkeypatch):
    monkeypatch.delenv("BIOLM_CLI_THEME", raising=False)
    assert resolve_theme_mode() == "auto"
    monkeypatch.setenv("BIOLM_CLI_THEME", "light")
    assert resolve_theme_mode() == "light"
    monkeypatch.setenv("BIOLM_CLI_THEME", "dark")
    assert resolve_theme_mode() == "dark"


def test_terminal_is_dark_from_colorfgbg(monkeypatch):
    monkeypatch.delenv("JPY_PARENT_PID", raising=False)
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    monkeypatch.delenv("JUPYTER_SERVER_ROOT", raising=False)
    monkeypatch.delenv("JUPYTERHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert terminal_is_dark() is True
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert terminal_is_dark() is False
    # Solarized Dark often reports bg=8 (bright black) — still a dark background
    monkeypatch.setenv("COLORFGBG", "12;8")
    assert terminal_is_dark() is True


def test_terminal_defaults_light_inside_jupyter_without_colorfgbg(monkeypatch):
    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    monkeypatch.delenv("JUPYTER_SERVER_ROOT", raising=False)
    monkeypatch.delenv("JUPYTERHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("JPY_PARENT_PID", "12345")
    assert terminal_is_dark() is False


def test_jupyter_colorfgbg_still_controls_theme(monkeypatch):
    monkeypatch.setenv("JPY_PARENT_PID", "12345")
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert terminal_is_dark() is True


def test_dark_theme_uses_readable_text_style():
    theme = build_theme(dark=True)
    assert "bright_white" in str(theme.styles["text"].color)
    assert "bright_cyan" in str(theme.styles["brand"].color)


def test_light_theme_uses_ansi_text():
    theme = build_theme(dark=False)
    assert "black" in str(theme.styles["text"].color)
    assert "blue" in str(theme.styles["brand"].color)


def test_auto_theme_does_not_force_color_on_pipe(monkeypatch):
    """Non-TTY stdout must not look like a terminal to Rich (CliRunner / CI)."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("BIOLM_CLI_THEME", raising=False)
    monkeypatch.setattr("biolm.cli.theme._stdout_is_tty", lambda stream=None: False)
    console = create_console()
    assert console.is_terminal is False
    assert console._color_system is None
    from io import StringIO

    buf = StringIO()
    console.file = buf
    with console.status("Fetching models..."):
        pass
    console.print("[error]Missing dependencies[/error]")
    console.print('[{"a": 1}]')
    out = buf.getvalue()
    assert "\x1b" not in out
    assert "Missing dependencies" in out
    assert '[{"a": 1}]' in out


def test_click_runner_json_not_polluted_by_status(monkeypatch):
    """Regression: Rich+Click wrapper must not inject spinner into JSON."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("BIOLM_CLI_THEME", raising=False)

    import json
    from unittest.mock import patch

    from click.testing import CliRunner

    from biolm.cli import cli

    models = [
        {
            "model_name": "ESM2-8M",
            "model_slug": "esm2-8m",
            "encoder": True,
            "predictor": False,
            "generator": False,
        }
    ]
    with patch("biolm.cli.list_models", return_value=models):
        result = CliRunner().invoke(cli, ["model", "list", "--format", "json"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert json.loads(result.output)[0]["model_slug"] == "esm2-8m"


def test_explicit_theme_forces_ansi256(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    console = create_console(theme_mode="light")
    assert console.is_terminal is True
    assert console._color_system is not None
