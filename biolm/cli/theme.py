"""Terminal-aware Rich theme for the biolm CLI."""
from __future__ import annotations

import os
import sys
from typing import Literal, Optional, TextIO

from rich.console import Console
from rich.theme import Theme

ThemeMode = Literal["auto", "light", "dark"]

# Light background — ANSI colors (JupyterLab/xterm often ignore truecolor hex)
_LIGHT_THEME = Theme(
    {
        "brand": "blue",
        "brand.bold": "bold blue",
        "brand.bright": "bold blue",
        "brand.dark": "black",
        "text": "black",
        "text.muted": "bright_black",
        "success": "green",
        "success.bold": "bold green",
        "error": "red",
        "warning": "yellow",
        "accent": "magenta",
        "border": "bright_black",
    }
)

# Dark background — ANSI names for readable contrast on black terminals
_DARK_THEME = Theme(
    {
        "brand": "bright_cyan",
        "brand.bold": "bold bright_cyan",
        "brand.bright": "bold bright_cyan",
        "brand.dark": "bright_blue",
        "text": "bright_white",
        "text.muted": "bright_black",
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
    """Best-effort detection of dark terminal background (xterm COLORFGBG).

    Only treats unambiguously light backgrounds as light. Solarized Dark and
    similar themes often use palette index 8–11 for the background, which is
    still visually dark but was previously misclassified as light.

    JupyterLab's default terminal is light and often omits COLORFGBG; when we
    can tell we are inside a Jupyter session and COLORFGBG is unset, prefer
    the light theme.
    """
    colorfgbg = os.environ.get("COLORFGBG", "").strip()
    if colorfgbg:
        try:
            bg = int(colorfgbg.split(";")[-1])
            # 7 = light gray, 15 = bright white — clearly light backgrounds
            if bg in (7, 15):
                return False
            return True
        except ValueError:
            pass
    if _in_jupyter_session():
        return False
    # Conservative default: most standalone dev terminals are dark
    return True


def _in_jupyter_session() -> bool:
    """True when running under Jupyter / IPython kernel or Lab terminal."""
    return bool(
        os.environ.get("JPY_PARENT_PID")
        or os.environ.get("JPY_SESSION_NAME")
        or os.environ.get("JUPYTER_SERVER_ROOT")
        or os.environ.get("JUPYTERHUB_API_TOKEN")
    )


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


def _is_click_runner_stream(stream: Optional[TextIO] = None) -> bool:
    """True for Click CliRunner's captured stdout/stderr wrappers."""
    file = sys.stdout if stream is None else stream
    return type(file).__name__ == "_NamedTextIOWrapper"


def _stdout_is_tty(stream: Optional[TextIO] = None) -> bool:
    file = sys.stdout if stream is None else stream
    if _is_click_runner_stream(file):
        return False
    try:
        return bool(file.isatty())
    except Exception:
        return False


def create_console(
    *,
    no_color: bool | None = None,
    theme_mode: ThemeMode | None = None,
    force_color: bool = False,
) -> Console:
    """Create a Rich Console with terminal-appropriate colors.

    Color/ANSI is enabled only for a real TTY, or when ``force_color`` is set
    (``biolm --color``). Theme mode (light/dark) chooses the palette but does
    not by itself force terminal control codes — otherwise ``BIOLM_CLI_THEME``
    and CliRunner tests fight each other under ``pytest -s``.
    """
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
    is_tty = _stdout_is_tty()

    if no_color:
        force_terminal: bool | None = False
        color_system = None
        highlight = False
    elif force_color:
        # User passed ``--color``: allow ANSI even when stdout is piped.
        force_terminal = True
        color_system = "256"
        highlight = True
    elif is_tty:
        # Real interactive terminal (incl. JupyterLab). Pin ANSI-256 so Lab/xterm
        # get readable styles instead of ignored truecolor hex.
        force_terminal = None
        color_system = "256"
        highlight = True
    else:
        # Pipes / Click CliRunner: never emit spinner or color into captures.
        force_terminal = False
        color_system = None
        highlight = False

    return Console(
        no_color=no_color,
        highlight=highlight,
        theme=theme,
        color_system=color_system,
        force_terminal=force_terminal,
    )
