"""DatasetClient: create, discover, and resolve local datasets."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from biolm.datasets.backends import get_backend
from biolm.datasets.dataset import Dataset
from biolm.datasets.discovery import discover_datasets, path_is_under_roots
from biolm.datasets.errors import (
    DatasetError,
    DatasetExistsError,
    DatasetNotFoundError,
    DatasetSchemaError,
)
from biolm.datasets.paths import normalize_roots, user_datasets_dir
from biolm.datasets.schema import (
    DATASET_YAML,
    DEFAULT_TYPE,
    build_meta,
    load_dataset_yaml,
    write_dataset_yaml,
)

PathLike = Union[str, Path]


class DatasetClient:
    """Client for local dataset inventory and push/pull."""

    def __init__(
        self,
        roots: Optional[Sequence[PathLike]] = None,
        *,
        primary_root: Optional[PathLike] = None,
        cwd: Optional[PathLike] = None,
    ):
        self._cwd = Path(cwd).resolve() if cwd is not None else None
        self._explicit_roots = [Path(r) for r in roots] if roots else None
        self._primary_root = (
            Path(primary_root).expanduser().resolve()
            if primary_root is not None
            else None
        )

    @property
    def roots(self) -> List[Path]:
        return normalize_roots(
            self._explicit_roots,
            cwd=self._cwd,
            include_defaults=True,
        )

    @property
    def primary_root(self) -> Path:
        if self._primary_root is not None:
            return self._primary_root
        if self._explicit_roots:
            return Path(self._explicit_roots[0]).expanduser().resolve()
        return user_datasets_dir().resolve()

    def create(
        self,
        dataset_id: str,
        *,
        type: str = DEFAULT_TYPE,
        tags: Optional[List[str]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        root: Optional[PathLike] = None,
        force: bool = False,
    ) -> Dataset:
        """Create ``<root>/<id>/dataset.yaml`` and an empty ``data/`` directory."""
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise DatasetError("dataset id must be non-empty")

        base = Path(root).expanduser().resolve() if root is not None else self.primary_root
        dataset_dir = base / dataset_id
        yaml_path = dataset_dir / DATASET_YAML
        if yaml_path.exists() and not force:
            raise DatasetExistsError(
                f"Dataset already exists at {yaml_path} (pass force=True to overwrite)"
            )

        # Ensure id uniqueness across discovery roots (excluding this path on force)
        existing = discover_datasets(self.roots)
        if dataset_id in existing and existing[dataset_id] != dataset_dir.resolve():
            raise DatasetExistsError(
                f"Dataset id '{dataset_id}' already exists at {existing[dataset_id]}"
            )

        meta = build_meta(
            dataset_id,
            description=description,
            type=type,
            tags=tags,
            attrs=attrs,
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "data").mkdir(parents=True, exist_ok=True)
        write_dataset_yaml(dataset_dir, meta)
        return Dataset(dataset_dir, meta)

    def init(
        self,
        path: PathLike,
        *,
        id: Optional[str] = None,
        type: str = DEFAULT_TYPE,
        tags: Optional[List[str]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        force: bool = False,
    ) -> Dataset:
        """Write dataset.yaml into an existing directory without moving files."""
        dataset_dir = Path(path).expanduser().resolve()
        if not dataset_dir.is_dir():
            raise DatasetError(f"Not a directory: {dataset_dir}")

        yaml_path = dataset_dir / DATASET_YAML
        if yaml_path.exists() and not force:
            raise DatasetExistsError(
                f"Dataset already exists at {yaml_path} (pass force=True to overwrite)"
            )

        dataset_id = (id or dataset_dir.name).strip()
        if not dataset_id:
            raise DatasetError("dataset id must be non-empty")

        existing = discover_datasets(self.roots)
        if dataset_id in existing and existing[dataset_id] != dataset_dir:
            raise DatasetExistsError(
                f"Dataset id '{dataset_id}' already exists at {existing[dataset_id]}"
            )

        if not path_is_under_roots(dataset_dir, self.roots):
            warnings.warn(
                f"Dataset path {dataset_dir} is outside discovery roots; "
                f"it will be reachable by path but not via list()/get(id) until "
                f"its parent is added as a dataset root.",
                UserWarning,
                stacklevel=2,
            )

        meta = build_meta(
            dataset_id,
            description=description,
            type=type,
            tags=tags,
            attrs=attrs,
        )
        write_dataset_yaml(dataset_dir, meta)
        return Dataset(dataset_dir, meta)

    def list(
        self,
        *,
        type: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dataset]:
        """List datasets discovered under configured roots."""
        mapping = discover_datasets(self.roots)
        datasets: List[Dataset] = []
        for dataset_id, path in sorted(mapping.items()):
            try:
                ds = Dataset(path)
            except DatasetSchemaError:
                continue
            if type is not None and ds.type != type:
                continue
            if tag is not None and tag not in ds.tags:
                continue
            datasets.append(ds)
        return datasets

    def get(self, id_or_path: PathLike) -> Dataset:
        """Resolve a dataset by filesystem path or by id under discovery roots."""
        raw = Path(id_or_path)
        # Path that exists as a directory (or dataset.yaml)
        if raw.exists():
            path = raw.resolve()
            if path.is_file() and path.name == DATASET_YAML:
                path = path.parent
            if path.is_dir() and (path / DATASET_YAML).is_file():
                return Dataset(path)
            if path.is_dir():
                raise DatasetNotFoundError(f"No dataset.yaml in directory: {path}")

        dataset_id = str(id_or_path)
        mapping = discover_datasets(self.roots)
        if dataset_id not in mapping:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found under discovery roots: "
                + ", ".join(str(r) for r in self.roots)
            )
        return Dataset(mapping[dataset_id])

    def pull(
        self,
        dataset_id: str,
        *,
        backend: str = "mlflow",
        path: Optional[PathLike] = None,
        force: bool = False,
        **opts: Any,
    ) -> Dataset:
        """Pull a remote dataset into a local directory and return Dataset."""
        dest = (
            Path(path).expanduser().resolve()
            if path is not None
            else (self.primary_root / dataset_id).resolve()
        )

        if dest.exists() and (dest / DATASET_YAML).is_file() and not force:
            existing = load_dataset_yaml(dest)
            if existing.id != dataset_id:
                raise DatasetExistsError(
                    f"Destination {dest} already has dataset id '{existing.id}' "
                    f"(expected '{dataset_id}'); pass force=True to overwrite"
                )

        dest.mkdir(parents=True, exist_ok=True)
        get_backend(backend).pull(dataset_id, dest, force=force, **opts)

        # Ensure dataset.yaml exists after pull (backend should write it)
        if not (dest / DATASET_YAML).is_file():
            meta = build_meta(dataset_id)
            write_dataset_yaml(dest, meta)

        return Dataset(dest)
