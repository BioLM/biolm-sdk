"""Load, validate, and write dataset.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from biolm.datasets.errors import DatasetSchemaError

DATASET_YAML = "dataset.yaml"
SCHEMA_VERSION = 1
DEFAULT_TYPE = "files"

PathLike = Union[str, Path]


@dataclass
class DatasetMeta:
    """Validated metadata from dataset.yaml."""

    id: str
    schema_version: int = SCHEMA_VERSION
    description: Optional[str] = None
    created_at: Optional[str] = None
    type: str = DEFAULT_TYPE
    tags: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "type": self.type,
        }
        if self.description is not None:
            data["description"] = self.description
        if self.created_at is not None:
            data["created_at"] = self.created_at
        if self.tags:
            data["tags"] = list(self.tags)
        if self.attrs:
            data["attrs"] = dict(self.attrs)
        return data


def _require_mapping(data: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise DatasetSchemaError(
            f"{path}: dataset.yaml must be a mapping, got {type(data).__name__}"
        )
    return data


def validate_meta(data: Mapping[str, Any], *, path: Optional[Path] = None) -> DatasetMeta:
    """Validate a metadata mapping and return DatasetMeta."""
    loc = str(path) if path is not None else "dataset.yaml"
    if "id" not in data or not str(data.get("id", "")).strip():
        raise DatasetSchemaError(f"{loc}: missing required field 'id'")
    dataset_id = str(data["id"]).strip()

    schema_version = data.get("schema_version", SCHEMA_VERSION)
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise DatasetSchemaError(f"{loc}: schema_version must be an integer") from exc
    if schema_version != SCHEMA_VERSION:
        raise DatasetSchemaError(
            f"{loc}: unsupported schema_version {schema_version} "
            f"(expected {SCHEMA_VERSION})"
        )

    tags_raw = data.get("tags") or []
    if not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw):
        raise DatasetSchemaError(f"{loc}: tags must be a list of strings")

    attrs_raw = data.get("attrs") or {}
    if not isinstance(attrs_raw, Mapping):
        raise DatasetSchemaError(f"{loc}: attrs must be a mapping")

    dtype = data.get("type") or DEFAULT_TYPE
    if not isinstance(dtype, str) or not dtype.strip():
        raise DatasetSchemaError(f"{loc}: type must be a non-empty string")

    description = data.get("description")
    if description is not None and not isinstance(description, str):
        raise DatasetSchemaError(f"{loc}: description must be a string")

    created_at = data.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        raise DatasetSchemaError(f"{loc}: created_at must be a string")

    return DatasetMeta(
        id=dataset_id,
        schema_version=schema_version,
        description=description,
        created_at=created_at,
        type=dtype.strip(),
        tags=list(tags_raw),
        attrs=dict(attrs_raw),
    )


def load_dataset_yaml(path: PathLike) -> DatasetMeta:
    """Load and validate dataset.yaml from a file or dataset directory."""
    yaml_path = _resolve_yaml_path(path)
    if not yaml_path.is_file():
        raise DatasetSchemaError(f"dataset.yaml not found: {yaml_path}")
    try:
        with yaml_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise DatasetSchemaError(f"{yaml_path}: invalid YAML: {exc}") from exc
    data = _require_mapping(raw, yaml_path)
    return validate_meta(data, path=yaml_path)


def write_dataset_yaml(directory: PathLike, meta: DatasetMeta) -> Path:
    """Write dataset.yaml into *directory*; return the yaml path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    yaml_path = directory / DATASET_YAML
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta.to_dict(), f, default_flow_style=False, sort_keys=False)
    return yaml_path


def build_meta(
    dataset_id: str,
    *,
    description: Optional[str] = None,
    type: str = DEFAULT_TYPE,
    tags: Optional[List[str]] = None,
    attrs: Optional[Dict[str, Any]] = None,
    created_at: Optional[str] = None,
) -> DatasetMeta:
    """Construct metadata, defaulting created_at to now (UTC)."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return validate_meta(
        {
            "schema_version": SCHEMA_VERSION,
            "id": dataset_id,
            "description": description,
            "created_at": created_at,
            "type": type,
            "tags": tags or [],
            "attrs": attrs or {},
        }
    )


def _resolve_yaml_path(path: PathLike) -> Path:
    path = Path(path)
    if path.name == DATASET_YAML:
        return path
    return path / DATASET_YAML
