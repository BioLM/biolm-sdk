"""sf.query — lazy dataframe-style query operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Union

from biolm.seqframe.engine import DuckDBEngine
from biolm.seqframe.ops import (
    FilterOp,
    GroupByOp,
    JoinOp,
    LimitOp,
    SelectOp,
    SortOp,
    parse_columns,
    validate_column_names,
)

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame


class QueryNamespace:
    def __init__(self, sf: "SeqFrame"):
        self._sf = sf

    def filter(self, expr: str) -> "SeqFrame":
        DuckDBEngine.validate_filter_expr(expr)
        return self._sf._with_ops(FilterOp(expr))

    def select(self, columns: Union[str, Sequence[str]]) -> "SeqFrame":
        cols = parse_columns(columns)
        validate_column_names(cols)
        return self._sf._with_ops(SelectOp(cols))

    def join(
        self,
        other: "SeqFrame",
        *,
        on: str = "sequence_hash",
        how: str = "inner",
    ) -> "SeqFrame":
        validate_column_names([on])
        return self._sf._with_ops(JoinOp(other=other, on=on, how=how))

    def sort(self, by: Union[str, Sequence[str]], *, ascending: bool = True) -> "SeqFrame":
        cols = parse_columns(by)
        validate_column_names(cols)
        return self._sf._with_ops(SortOp(cols, ascending=ascending))

    def limit(self, n: int) -> "SeqFrame":
        return self._sf._with_ops(LimitOp(int(n)))

    def group_by(
        self,
        columns: Union[str, Sequence[str]],
        *,
        agg: Dict[str, str],
    ) -> "SeqFrame":
        cols = parse_columns(columns)
        validate_column_names(cols)
        validate_column_names(list(agg.keys()))
        return self._sf._with_ops(GroupByOp(cols, agg))
