"""Thin Dataset ↔ SeqFrame bridge.

Datasets remain bags of files; SeqFrame is the typed tabular opener when a
dataset contains exactly one SeqFrame Parquet (or an explicit ``attrs.seqframe_path``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from biolm.datasets.errors import DatasetError, DatasetNotFoundError
from biolm.datasets.schema import DatasetMeta, write_dataset_yaml

if TYPE_CHECKING:
    from biolm.datasets.client import DatasetClient
    from biolm.datasets.dataset import Dataset
    from biolm.seqframe.core import SeqFrame

PathLike = Union[str, Path]

SEQFRAME_ATTR_PATH = "seqframe_path"
SEQFRAME_TYPE = "seqframe"


def _parquet_candidates(dataset: "Dataset") -> List[Path]:
    """Absolute paths to ``.parquet`` files under the dataset."""
    found: List[Path] = []
    for rel in dataset.files():
        if rel.suffix.lower() == ".parquet":
            found.append((dataset.path / rel).resolve())
    return found


def resolve_seqframe_parquet(dataset: "Dataset") -> Path:
    """Resolve the SeqFrame Parquet path inside a dataset.

    Resolution order:
    1. ``attrs["seqframe_path"]`` (relative to dataset root, or absolute)
    2. Exactly one ``*.parquet`` file under the dataset

    Raises:
        DatasetError: If zero or multiple candidates, or the pointed path is missing.
    """
    explicit = dataset.attrs.get(SEQFRAME_ATTR_PATH)
    if explicit:
        path = Path(str(explicit))
        if not path.is_absolute():
            path = (dataset.path / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise DatasetError(
                f"Dataset {dataset.id!r} attrs.{SEQFRAME_ATTR_PATH}={explicit!r} "
                f"does not exist at {path}"
            )
        return path

    candidates = _parquet_candidates(dataset)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise DatasetError(
            f"Dataset {dataset.id!r} has no Parquet files to open as SeqFrame. "
            f"Write one with SeqFrame.to_dataset() or set attrs.{SEQFRAME_ATTR_PATH}."
        )
    rels = [str(p.relative_to(dataset.path)) for p in candidates]
    raise DatasetError(
        f"Dataset {dataset.id!r} has {len(candidates)} Parquet files; "
        f"set attrs.{SEQFRAME_ATTR_PATH} to choose one. Found: {rels}"
    )


def open_seqframe(dataset: "Dataset") -> "SeqFrame":
    """Open a Dataset as a SeqFrame (requires ``biolm-sdk[seqframe]``)."""
    from biolm.seqframe.core import SeqFrame

    return SeqFrame.read(resolve_seqframe_parquet(dataset))


def seqframe_to_dataset(
    sf: "SeqFrame",
    dataset_id: str,
    *,
    client: Optional["DatasetClient"] = None,
    filename: str = "sequences.parquet",
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    attrs: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> "Dataset":
    """Materialize a SeqFrame into a local Dataset with ``type: seqframe``.

    Creates the dataset under the client's primary root when missing, writes
    Parquet under ``data/``, and records ``attrs.seqframe_path``.
    """
    from biolm.datasets.client import DatasetClient

    if client is None:
        client = DatasetClient()

    rel_path = f"data/{filename}"
    meta_attrs: Dict[str, Any] = dict(attrs or {})
    meta_attrs[SEQFRAME_ATTR_PATH] = rel_path

    try:
        ds = client.get(dataset_id)
        if not force:
            raise DatasetError(
                f"Dataset {dataset_id!r} already exists at {ds.path}. "
                f"Pass force=True to overwrite the SeqFrame artifact."
            )
    except DatasetNotFoundError:
        ds = client.create(
            dataset_id,
            type=SEQFRAME_TYPE,
            tags=tags,
            attrs=meta_attrs,
            description=description,
        )
    else:
        new_meta = DatasetMeta(
            id=ds.id,
            schema_version=ds.meta.schema_version,
            description=description if description is not None else ds.description,
            created_at=ds.created_at,
            type=SEQFRAME_TYPE,
            tags=list(tags) if tags is not None else list(ds.tags),
            attrs={**dict(ds.attrs), **meta_attrs},
        )
        write_dataset_yaml(ds.path, new_meta)
        ds = ds.refresh()

    dest = ds.data_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.io.to_parquet(dest)
    return ds.refresh()
