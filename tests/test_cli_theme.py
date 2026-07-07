"""Tests for terminal-aware CLI theming."""
import os

import pytest

from biolm.cli_theme import (
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
    monkeypatch.setenv("COLORFGBG", "15;0")
    assert terminal_is_dark() is True
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert terminal_is_dark() is False


def test_dark_theme_uses_readable_text_style():
    theme = build_theme(dark=True)
    assert theme.styles["text"].color is None or theme.styles["text"].color.name == "default"
    assert "bright_blue" in str(theme.styles["brand.bright"].color)


def test_light_theme_uses_hex_text():
    theme = build_theme(dark=False)
    assert "171717" in str(theme.styles["text"].color)
