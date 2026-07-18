"""Dataset resource: one local self-describing dataset directory."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from biolm.datasets.backends import get_backend
from biolm.datasets.errors import DatasetError
from biolm.datasets.schema import (
    DATASET_YAML,
    DatasetMeta,
    load_dataset_yaml,
    write_dataset_yaml,
)

PathLike = Union[str, Path]


class Dataset:
    """A local dataset directory containing dataset.yaml."""

    def __init__(self, path: PathLike, meta: Optional[DatasetMeta] = None):
        self._path = Path(path).resolve()
        self._meta = meta if meta is not None else load_dataset_yaml(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def id(self) -> str:
        return self._meta.id

    @property
    def type(self) -> str:
        return self._meta.type

    @property
    def tags(self) -> List[str]:
        return list(self._meta.tags)

    @property
    def attrs(self) -> Dict[str, Any]:
        return dict(self._meta.attrs)

    @property
    def description(self) -> Optional[str]:
        return self._meta.description

    @property
    def created_at(self) -> Optional[str]:
        return self._meta.created_at

    @property
    def meta(self) -> DatasetMeta:
        return self._meta

    @property
    def data_dir(self) -> Path:
        """Preferred content directory: data/ if present, else dataset root."""
        candidate = self._path / "data"
        return candidate if candidate.is_dir() else self._path

    def refresh(self) -> "Dataset":
        """Re-read dataset.yaml from disk."""
        self._meta = load_dataset_yaml(self._path)
        return self

    def files(self) -> List[Path]:
        """Relative paths of files under the dataset (excluding dataset.yaml)."""
        results: List[Path] = []
        for path in sorted(self._path.rglob("*")):
            if not path.is_file():
                continue
            if path.name == DATASET_YAML and path.parent == self._path:
                continue
            results.append(path.relative_to(self._path))
        return results

    def add(
        self,
        source: PathLike,
        *,
        recursive: bool = False,
        dest_name: Optional[str] = None,
    ) -> Path:
        """Copy a file or directory into the dataset content directory.

        Returns the destination path.
        """
        source_path = Path(source)
        if not source_path.exists():
            raise DatasetError(f"Source not found: {source_path}")

        dest_root = self.data_dir
        dest_root.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            target = dest_root / (dest_name or source_path.name)
            shutil.copy2(source_path, target)
            return target

        if not recursive and not dest_name:
            # Copy directory tree into dest_root/<dirname>/
            target = dest_root / source_path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source_path, target)
            return target

        # recursive flag: copy contents (or named subtree)
        target = dest_root / (dest_name or source_path.name)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_path, target)
        return target

    def push(self, backend: str = "mlflow", **opts: Any) -> Dict[str, Any]:
        """Push this dataset via a registered backend."""
        return get_backend(backend).push(self, **opts)

    def open_seqframe(self):
        """Open this dataset as a :class:`~biolm.seqframe.SeqFrame`.

        Requires ``biolm-sdk[seqframe]``. Resolves ``attrs.seqframe_path`` or
        exactly one ``*.parquet`` under the dataset. See
        :mod:`biolm.seqframe.dataset_bridge`.
        """
        from biolm.seqframe.dataset_bridge import open_seqframe

        return open_seqframe(self)

    def to_dict(self) -> Dict[str, Any]:
        data = self._meta.to_dict()
        data["path"] = str(self._path)
        data["files"] = [str(p) for p in self.files()]
        return data

    def __repr__(self) -> str:
        return f"Dataset(id={self.id!r}, path={str(self._path)!r}, type={self.type!r})"
