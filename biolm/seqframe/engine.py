"""DuckDB query engine for SeqFrame."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import duckdb
import pandas as pd

from biolm.seqframe.metadata import SeqFrameMetadata
from biolm.seqframe.ops import Op

_SQL_DENY = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|ATTACH|DETACH|"
    r"COPY|PRAGMA|INSTALL|LOAD|EXPORT|IMPORT|READ_CSV|READ_PARQUET|"
    r"READ_JSON|GLOB)\b",
    re.IGNORECASE,
)


class DuckDBEngine:
    """Compile lazy ops to DuckDB SQL and execute."""

    @staticmethod
    def _parquet_source(path: Path) -> str:
        escaped = str(path).replace("'", "''")
        return f"SELECT * FROM read_parquet('{escaped}')"

    @classmethod
    def compile(cls, parquet_path: Path, ops: Sequence[Op]) -> str:
        sql = cls._parquet_source(parquet_path)
        for op in ops:
            sql = op.apply(sql)
        return sql

    @classmethod
    def validate_filter_expr(cls, expr: str) -> None:
        if _SQL_DENY.search(expr):
            raise ValueError(
                f"Filter expression contains disallowed SQL keyword: {expr!r}"
            )

    @classmethod
    def execute(
        cls,
        parquet_path: Path,
        ops: Sequence[Op],
        *,
        con: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> pd.DataFrame:
        sql = cls.compile(parquet_path, ops)
        own_con = con is None
        if own_con:
            con = duckdb.connect()
        try:
            return con.execute(sql).df()
        finally:
            if own_con:
                con.close()

    @classmethod
    def count(
        cls,
        parquet_path: Path,
        ops: Sequence[Op],
        *,
        con: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> int:
        sql = f"SELECT COUNT(*) AS n FROM ({cls.compile(parquet_path, ops)})"
        own_con = con is None
        if own_con:
            con = duckdb.connect()
        try:
            row = con.execute(sql).fetchone()
            return int(row[0]) if row else 0
        finally:
            if own_con:
                con.close()

    @classmethod
    def materialize(
        cls,
        parquet_path: Path,
        ops: Sequence[Op],
        output_path: Path,
        metadata: SeqFrameMetadata,
    ) -> Tuple[Path, Tuple[Op, ...]]:
        """Execute plan and write a new Parquet file with SeqFrame metadata."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = cls.execute(parquet_path, ops)
        table = pa.Table.from_pandas(df, preserve_index=False)
        meta = metadata.to_parquet_metadata()
        existing = dict(table.schema.metadata or {})
        existing.update(meta)
        table = table.replace_schema_metadata(existing)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)
        return output_path, ()
