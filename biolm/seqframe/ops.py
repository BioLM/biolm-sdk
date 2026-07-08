"""Lazy query operations for SeqFrame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame


@dataclass(frozen=True)
class Op:
    """Base class for lazy query operations."""

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class FilterOp(Op):
    expr: str

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        return f"SELECT * FROM ({source_sql}) AS {alias} WHERE {self.expr}"


@dataclass(frozen=True)
class SelectOp(Op):
    columns: Tuple[str, ...]

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        cols = ", ".join(self.columns)
        return f"SELECT {cols} FROM ({source_sql}) AS {alias}"


@dataclass(frozen=True)
class JoinOp(Op):
    other: "SeqFrame"
    on: str
    how: str = "inner"

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        from biolm.seqframe.engine import DuckDBEngine

        other_sql = DuckDBEngine.compile(self.other._parquet_path, self.other._ops)
        how = self.how.upper()
        if how not in {"INNER", "LEFT", "RIGHT", "FULL"}:
            raise ValueError(f"Unsupported join type: {self.how!r}")
        return (
            f"SELECT {alias}.*, o.* EXCLUDE ({self.on}) "
            f"FROM ({source_sql}) AS {alias} "
            f"{how} JOIN ({other_sql}) AS o USING ({self.on})"
        )


@dataclass(frozen=True)
class SortOp(Op):
    by: Tuple[str, ...]
    ascending: bool = True

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        direction = "ASC" if self.ascending else "DESC"
        order = ", ".join(f"{col} {direction}" for col in self.by)
        return f"SELECT * FROM ({source_sql}) AS {alias} ORDER BY {order}"


@dataclass(frozen=True)
class LimitOp(Op):
    n: int

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        if self.n < 0:
            raise ValueError("limit must be non-negative")
        return f"SELECT * FROM ({source_sql}) AS {alias} LIMIT {int(self.n)}"


@dataclass(frozen=True)
class GroupByOp(Op):
    columns: Tuple[str, ...]
    agg: Dict[str, str]

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        group_cols = ", ".join(self.columns)
        agg_parts = [f"{expr} AS {name}" for name, expr in self.agg.items()]
        if not agg_parts:
            raise ValueError("group_by requires at least one aggregation")
        agg_sql = ", ".join(agg_parts)
        return (
            f"SELECT {group_cols}, {agg_sql} "
            f"FROM ({source_sql}) AS {alias} "
            f"GROUP BY {group_cols}"
        )


@dataclass(frozen=True)
class AddColumnOp(Op):
    """Add a computed column via SQL expression."""

    name: str
    expr: str

    def apply(self, source_sql: str, *, alias: str = "t") -> str:
        return (
            f"SELECT {alias}.*, ({self.expr}) AS {self.name} "
            f"FROM ({source_sql}) AS {alias}"
        )


def validate_column_names(columns: Sequence[str]) -> None:
    import re

    pat = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    for col in columns:
        if not pat.match(col):
            raise ValueError(
                f"Invalid column name {col!r}: must match ^[a-zA-Z_][a-zA-Z0-9_]*$"
            )


def parse_columns(cols: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    if isinstance(cols, str):
        return tuple(c.strip() for c in cols.split(",") if c.strip())
    return tuple(cols)
