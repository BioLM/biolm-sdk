"""Terminal-aware Rich theme for the biolm CLI."""
from __future__ import annotations

import os
from typing import Literal

from rich.console import Console
from rich.theme import Theme

ThemeMode = Literal["auto", "light", "dark"]

# Light background — brand hex colors (docs/web palette)
_LIGHT_THEME = Theme(
    {
        "brand": "#558BF7",
        "brand.bold": "#558BF7 bold",
        "brand.bright": "#2563EB",
        "brand.dark": "#131443",
        "text": "#171717",
        "text.muted": "#666666",
        "success": "#10B981",
        "success.bold": "#10B981 bold",
        "error": "#F59E0B",
        "warning": "#F59E0B",
        "accent": "#8B5CF6",
        "border": "#666666",
    }
)

# Dark background — ANSI names for readable contrast on black terminals
_DARK_THEME = Theme(
    {
        "brand": "bright_blue",
        "brand.bold": "bold bright_blue",
        "brand.bright": "bold bright_blue",
        "brand.dark": "blue",
        "text": "default",
        "text.muted": "dim",
        "success": "green",
        "success.bold": "bold green",
        "error": "yellow",
        "warning": "yellow",
        "accent": "magenta",
        "border": "dim",
    }
)


def no_color_requested() -> bool:
    """True when NO_COLOR is set (https://no-color.org/)."""
    return os.environ.get("NO_COLOR", "").strip() != ""


def resolve_theme_mode(explicit: ThemeMode | None = None) -> ThemeMode:
    """Resolve theme mode from flag, BIOLM_CLI_THEME env, or auto."""
    if explicit and explicit != "auto":
        return explicit
    env = os.environ.get("BIOLM_CLI_THEME", "auto").strip().lower()
    if env in ("light", "dark"):
        return env  # type: ignore[return-value]
    return "auto"


def terminal_is_dark() -> bool:
    """Best-effort detection of dark terminal background (xterm COLORFGBG)."""
    colorfgbg = os.environ.get("COLORFGBG", "").strip()
    if colorfgbg:
        try:
            bg = int(colorfgbg.split(";")[-1])
            # 0–7 dark, 8–15 light in the 16-color xterm palette
            if bg <= 7:
                return True
            if bg >= 8:
                return False
        except ValueError:
            pass
    # Conservative default: most dev terminals are dark
    return True


def build_theme(*, dark: bool, plain: bool = False) -> Theme:
    if plain:
        # Keep style names resolvable when NO_COLOR is set (markup must not crash)
        return Theme(
            {
                "brand": "",
                "brand.bold": "bold",
                "brand.bright": "bold",
                "brand.dark": "",
                "text": "",
                "text.muted": "dim",
                "success": "",
                "success.bold": "bold",
                "error": "",
                "warning": "",
                "accent": "",
                "border": "",
            }
        )
    return _DARK_THEME if dark else _LIGHT_THEME


def create_console(
    *,
    no_color: bool | None = None,
    theme_mode: ThemeMode | None = None,
) -> Console:
    """Create a Rich Console with terminal-appropriate colors."""
    if no_color is None:
        no_color = no_color_requested()

    mode = resolve_theme_mode(theme_mode)
    if mode == "dark":
        use_dark = True
    elif mode == "light":
        use_dark = False
    else:
        use_dark = terminal_is_dark()

    theme = build_theme(dark=use_dark, plain=no_color)
    return Console(no_color=no_color, highlight=not no_color, theme=theme)
