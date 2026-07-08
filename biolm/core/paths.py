"""Canonical user config paths under ~/.biolm with ~/.biolmai fallbacks."""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_LEGACY_WARNED: set[str] = set()


def user_config_dir() -> Path:
    """Return the canonical user config directory (~/.biolm)."""
    return Path.home() / ".biolm"


def legacy_user_config_dir() -> Path:
    """Return the deprecated user config directory (~/.biolmai)."""
    return Path.home() / ".biolmai"


def warn_deprecated_path(legacy_path: PathLike, canonical_path: PathLike) -> None:
    """Emit a DeprecationWarning once per legacy path."""
    key = str(legacy_path)
    if key in _LEGACY_WARNED:
        return
    _LEGACY_WARNED.add(key)
    warnings.warn(
        f"Path {legacy_path} is deprecated; use {canonical_path} instead.",
        DeprecationWarning,
        stacklevel=4,
    )


def resolve_user_path(
    relative_path: PathLike,
    *,
    warn_legacy: bool = True,
) -> Path:
    """Resolve a path under the user config dir with legacy fallback.

    If the canonical path exists, it is returned. Otherwise, if the legacy
    path exists, it is returned after emitting a deprecation warning. If
    neither exists, the canonical path is returned (for new writes).
    """
    rel = Path(relative_path)
    canonical = user_config_dir() / rel
    legacy = legacy_user_config_dir() / rel

    if canonical.exists():
        return canonical
    if legacy.exists():
        if warn_legacy:
            warn_deprecated_path(legacy, canonical)
        return legacy
    return canonical


def resolve_user_file(
    relative_path: PathLike,
    *,
    warn_legacy: bool = True,
) -> str:
    """Like :func:`resolve_user_path` but returns a string path."""
    return str(resolve_user_path(relative_path, warn_legacy=warn_legacy))


def ensure_user_config_dir() -> Path:
    """Create and return the canonical user config directory."""
    path = user_config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
