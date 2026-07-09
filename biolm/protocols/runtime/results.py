"""DataFrame → normalized record list."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pipeline final DataFrame to list[dict] with sequence field."""
    if df is None or df.empty:
        return []

    out = df.copy()
    if "sequence" not in out.columns and "sequence_id" in out.columns:
        # Keep sequence_id; callers may join later — but profile expects sequence when possible
        pass

    records: list[dict[str, Any]] = []
    for row in out.to_dict(orient="records"):
        rec = {k: _json_safe(v) for k, v in row.items()}
        if "sequence" not in rec and "sequence" in out.columns:
            rec["sequence"] = row.get("sequence")
        records.append(rec)
    return records


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and (value != value):  # NaN
        return None
    return value
