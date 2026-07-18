"""SeqFrame core: immutable, lazy sequence-centric dataframe."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq

from biolm import __version__ as SDK_VERSION
from biolm.seqframe.engine import DuckDBEngine
from biolm.seqframe.metadata import SEQFRAME_VERSION, SeqFrameMetadata
from biolm.seqframe.ops import Op
from biolm.seqframe.rows import dataframe_from_table, normalize_rows


class SeqFrame:
    """Sequence-centric dataframe abstraction over Parquet + DuckDB.

    SeqFrames are immutable: transforming operations return new instances.
    Query operations are lazy until materialization (collect, write, enrich).
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        ops: Tuple[Op, ...] = (),
        metadata: Optional[SeqFrameMetadata] = None,
    ):
        self._parquet_path = Path(parquet_path).resolve()
        self._ops = ops
        self._metadata = metadata or SeqFrameMetadata(created_by=f"biolm-sdk/{SDK_VERSION}")

    @property
    def schema(self) -> SeqFrameMetadata:
        return self._metadata

    @property
    def columns(self) -> List[str]:
        return list(self.collect().columns)

    @property
    def shape(self) -> Tuple[int, int]:
        nrows = len(self)
        ncols = len(self.head(0).columns)
        return nrows, ncols

    def __len__(self) -> int:
        return DuckDBEngine.count(self._parquet_path, self._ops)

    def __repr__(self) -> str:
        return (
            f"SeqFrame(rows={len(self)}, ops={len(self._ops)}, "
            f"molecule_type={self._metadata.molecule_type!r})"
        )

    def _with_ops(self, *new_ops: Op) -> "SeqFrame":
        return SeqFrame(self._parquet_path, self._ops + new_ops, self._metadata)

    def _with_metadata(self, metadata: SeqFrameMetadata) -> "SeqFrame":
        return SeqFrame(self._parquet_path, self._ops, metadata)

    def _materialize_to(
        self,
        path: Optional[Path] = None,
        *,
        clear_ops: bool = True,
    ) -> "SeqFrame":
        if path is None:
            fd, tmp = tempfile.mkstemp(suffix=".parquet")
            import os

            os.close(fd)
            path = Path(tmp)
        out_path, ops = DuckDBEngine.materialize(
            self._parquet_path,
            self._ops,
            path,
            self._metadata,
        )
        if clear_ops:
            return SeqFrame(out_path, ops, self._metadata)
        return SeqFrame(out_path, self._ops, self._metadata)

    def collect(self) -> pd.DataFrame:
        return DuckDBEngine.execute(self._parquet_path, self._ops)

    def head(self, n: int = 5) -> pd.DataFrame:
        from biolm.seqframe.ops import LimitOp

        if n <= 0:
            df = self.collect()
            return df.iloc[0:0]
        limited = self._with_ops(LimitOp(n))
        return limited.collect()

    def write(self, path: Union[str, Path]) -> "SeqFrame":
        return self.io.to_parquet(path)

    @classmethod
    def read(cls, path: Union[str, Path]) -> "SeqFrame":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SeqFrame file not found: {path}")
        table = pq.read_table(path)
        metadata = SeqFrameMetadata.from_parquet_metadata(dict(table.schema.metadata or {}))
        return cls(path, (), metadata)

    @classmethod
    def from_rows(
        cls,
        rows: List[Dict[str, Any]],
        *,
        sequence_column: str = "sequence",
        id_column: str = "id",
        molecule_type: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> "SeqFrame":
        df = normalize_rows(
            rows,
            sequence_column=sequence_column,
            id_column=id_column,
            molecule_type=molecule_type,
        )
        return cls.from_dataframe(df, molecule_type=molecule_type, path=path)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        molecule_type: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> "SeqFrame":
        if molecule_type is None:
            molecule_type = df.attrs.get("_molecule_type", "unknown")
        if path is None:
            fd, tmp = tempfile.mkstemp(suffix=".parquet")
            import os

            os.close(fd)
            path = Path(tmp)
        else:
            path = Path(path)

        meta = SeqFrameMetadata(
            molecule_type=molecule_type or "unknown",
            created_by=f"biolm-sdk/{SDK_VERSION}",
            version=SEQFRAME_VERSION,
        )
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        existing = dict(table.schema.metadata or {})
        existing.update(meta.to_parquet_metadata())
        table = table.replace_schema_metadata(existing)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        return cls(path, (), meta)

    @property
    def query(self):
        from biolm.seqframe.namespaces.query import QueryNamespace

        return QueryNamespace(self)

    @property
    def io(self):
        from biolm.seqframe.namespaces.io import IONamespace

        return IONamespace(self)

    @property
    def bio(self):
        from biolm.seqframe.namespaces.bio import BioNamespace

        return BioNamespace(self)

    @property
    def models(self):
        from biolm.seqframe.namespaces.models import ModelsNamespace

        return ModelsNamespace(self)

    @property
    def protocols(self):
        from biolm.seqframe.namespaces.protocols import ProtocolsNamespace

        return ProtocolsNamespace(self)

    @property
    def lab(self):
        from biolm.seqframe.namespaces.lab import LabNamespace

        return LabNamespace(self)

    @classmethod
    def from_fasta(cls, path: Union[str, Path], **kwargs) -> "SeqFrame":
        from biolm.seqframe.importers import from_fasta

        return from_fasta(path, **kwargs)

    @classmethod
    def from_csv(cls, path: Union[str, Path], **kwargs) -> "SeqFrame":
        from biolm.seqframe.importers import from_csv

        return from_csv(path, **kwargs)

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], **kwargs) -> "SeqFrame":
        from biolm.seqframe.importers import from_jsonl

        return from_jsonl(path, **kwargs)

    @classmethod
    def from_protocol(cls, run_or_path: Any, **kwargs) -> "SeqFrame":
        from biolm.seqframe.importers import from_protocol

        return from_protocol(run_or_path, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: Any) -> "SeqFrame":
        """Open a :class:`~biolm.datasets.Dataset` as a SeqFrame."""
        from biolm.seqframe.dataset_bridge import open_seqframe

        return open_seqframe(dataset)

    def to_dataset(
        self,
        dataset_id: str,
        *,
        client: Any = None,
        filename: str = "sequences.parquet",
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Any:
        """Write this SeqFrame into a local Dataset (``type: seqframe``)."""
        from biolm.seqframe.dataset_bridge import seqframe_to_dataset

        return seqframe_to_dataset(
            self,
            dataset_id,
            client=client,
            filename=filename,
            tags=tags,
            description=description,
            attrs=attrs,
            force=force,
        )

    def merge_columns(self, df: pd.DataFrame, *, on: str = "id") -> "SeqFrame":
        """Join enrichment results back into this SeqFrame by key column."""
        base = self.collect()
        merged = base.merge(df, on=on, how="left", suffixes=("", "_enriched"))
        return SeqFrame.from_dataframe(merged, molecule_type=self._metadata.molecule_type)
