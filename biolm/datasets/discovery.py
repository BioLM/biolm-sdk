"""Discover local datasets under configured roots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

from biolm.datasets.errors import DuplicateDatasetIdError, DatasetSchemaError
from biolm.datasets.schema import DATASET_YAML, DatasetMeta, load_dataset_yaml

PathLike = Union[str, Path]


def iter_dataset_yaml_paths(roots: Sequence[PathLike]) -> List[Path]:
    """Recursively find dataset.yaml files under *roots* only."""
    found: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for yaml_path in root_path.rglob(DATASET_YAML):
            if not yaml_path.is_file():
                continue
            resolved = yaml_path.resolve()
            if resolved in seen:
                continue
            # Skip nested .git / junk if any; still under designated roots only
            seen.add(resolved)
            found.append(resolved)
    return found


def discover_datasets(
    roots: Sequence[PathLike],
) -> Dict[str, Path]:
    """Return mapping of dataset id → dataset directory.

    Raises DuplicateDatasetIdError if the same id appears more than once.
    Invalid yaml files are skipped.
    """
    by_id: Dict[str, List[Path]] = {}
    for yaml_path in iter_dataset_yaml_paths(roots):
        try:
            meta = load_dataset_yaml(yaml_path)
        except DatasetSchemaError:
            continue
        dataset_dir = yaml_path.parent.resolve()
        by_id.setdefault(meta.id, []).append(dataset_dir)

    duplicates = {did: paths for did, paths in by_id.items() if len(paths) > 1}
    if duplicates:
        lines = []
        for did, paths in sorted(duplicates.items()):
            joined = ", ".join(str(p) for p in paths)
            lines.append(f"  {did}: {joined}")
        raise DuplicateDatasetIdError(
            "Duplicate dataset id(s) found under discovery roots:\n" + "\n".join(lines)
        )

    return {did: paths[0] for did, paths in by_id.items()}


def path_is_under_roots(path: PathLike, roots: Sequence[PathLike]) -> bool:
    """Return True if *path* is equal to or under any discovery root."""
    target = Path(path).resolve()
    for root in roots:
        root_path = Path(root).resolve()
        try:
            target.relative_to(root_path)
            return True
        except ValueError:
            continue
    return False


def load_discovered_meta(
    roots: Sequence[PathLike],
) -> Dict[str, DatasetMeta]:
    """Discover datasets and return id → metadata."""
    mapping = discover_datasets(roots)
    return {did: load_dataset_yaml(path) for did, path in mapping.items()}
