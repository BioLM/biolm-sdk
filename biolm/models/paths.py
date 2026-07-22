"""Local registry paths for BioLM model packages."""
from __future__ import annotations

from pathlib import Path

from biolm.core.paths import user_config_dir


def user_models_dir() -> Path:
    """Return ``~/.biolm/models``."""
    return user_config_dir() / "models"
