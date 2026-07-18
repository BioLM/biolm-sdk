"""Discovery roots for local datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from biolm.core.paths import user_config_dir
from biolm.hub.config import read_config

PathLike = Union[str, Path]


def user_datasets_dir() -> Path:
    """Return ``~/.biolm/datasets``."""
    return user_config_dir() / "datasets"


def project_datasets_dir(cwd: Optional[PathLike] = None) -> Path:
    """Return ``<cwd>/.biolm/datasets``."""
    base = Path(cwd) if cwd is not None else Path.cwd()
    return base / ".biolm" / "datasets"


def config_dataset_roots() -> List[Path]:
    """Extra roots from ``~/.biolm/config.yaml`` key ``dataset_roots``."""
    raw = read_config().get("dataset_roots") or []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    roots: List[Path] = []
    for item in raw:
        if isinstance(item, (str, Path)) and str(item).strip():
            roots.append(Path(item).expanduser().resolve())
    return roots


def default_discovery_roots(
    *,
    cwd: Optional[PathLike] = None,
    include_config: bool = True,
) -> List[Path]:
    """Default discovery roots in priority order (project, then user, then config).

    Explicit client roots are prepended by DatasetClient and take precedence.
    """
    roots: List[Path] = [
        project_datasets_dir(cwd).resolve(),
        user_datasets_dir().resolve(),
    ]
    if include_config:
        for root in config_dataset_roots():
            if root not in roots:
                roots.append(root)
    return roots


def normalize_roots(
    roots: Optional[Sequence[PathLike]] = None,
    *,
    cwd: Optional[PathLike] = None,
    include_defaults: bool = True,
) -> List[Path]:
    """Build the ordered, deduplicated list of discovery roots."""
    ordered: List[Path] = []
    seen: set[Path] = set()

    def _add(items: Iterable[PathLike]) -> None:
        for item in items:
            path = Path(item).expanduser().resolve()
            if path not in seen:
                seen.add(path)
                ordered.append(path)

    if roots:
        _add(roots)
    if include_defaults:
        _add(default_discovery_roots(cwd=cwd, include_config=True))
    return ordered
