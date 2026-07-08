"""Normalize rows from biolm.io loaders into canonical SeqFrame columns."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from biolm.io.fasta import _detect_sequence_type


def hash_sequence(sequence: str) -> str:
    """SHA-256[:16] of uppercased sequence (SeqFrame canonical hash)."""
    return hashlib.sha256(sequence.upper().encode()).hexdigest()[:16]


def detect_molecule_type(sequence: str) -> str:
    """Map fasta detector output to SeqFrame molecule_type values."""
    kind = _detect_sequence_type(sequence)
    mapping = {"aa": "protein", "dna": "dna", "rna": "rna"}
    return mapping.get(kind, "unknown")


def infer_molecule_type(sequences: Sequence[str]) -> str:
    types = {detect_molecule_type(s) for s in sequences if s}
    types.discard("unknown")
    if len(types) == 1:
        return types.pop()
    return "unknown"


def normalize_rows(
    rows: List[Dict[str, Any]],
    *,
    sequence_column: str = "sequence",
    id_column: str = "id",
    molecule_type: Optional[str] = None,
) -> pd.DataFrame:
    """Convert loader rows into a canonical SeqFrame DataFrame."""
    if not rows:
        raise ValueError("Cannot create SeqFrame from empty rows")

    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        seq = row.get(sequence_column) or row.get("sequence")
        if seq is None:
            raise ValueError(
                f"Row {i} missing sequence column {sequence_column!r}. "
                f"Available keys: {list(row.keys())}"
            )
        seq = str(seq)
        row_id = row.get(id_column) or row.get("id") or row.get("sequence_id")
        if row_id is None:
            row_id = f"seq_{i + 1}"
        row_id = str(row_id)

        record: Dict[str, Any] = {
            "id": row_id,
            "sequence": seq,
            "sequence_hash": hash_sequence(seq),
            "length": len(seq),
        }
        for key, value in row.items():
            if key in (sequence_column, id_column, "id", "sequence_id"):
                continue
            if key in record:
                continue
            if key == "metadata" and isinstance(value, dict):
                record["metadata"] = json.dumps(value)
            else:
                record[key] = value
        records.append(record)

    df = pd.DataFrame(records)
    if molecule_type is None:
        molecule_type = infer_molecule_type(df["sequence"].tolist())
    df.attrs["_molecule_type"] = molecule_type
    return df


def dataframe_from_table(
    df: pd.DataFrame,
    *,
    sequence_column: str = "sequence",
    id_column: str = "id",
    molecule_type: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize an arbitrary DataFrame (e.g. protocol results) into SeqFrame shape."""
    rows = df.to_dict(orient="records")
    return normalize_rows(
        rows,
        sequence_column=sequence_column,
        id_column=id_column,
        molecule_type=molecule_type,
    )


def infer_sequence_column(df: pd.DataFrame) -> str:
    for candidate in ("sequence", "seq", "protein_sequence", "dna_sequence"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not infer sequence column. Available columns: {list(df.columns)}"
    )


def infer_id_column(df: pd.DataFrame) -> str:
    for candidate in ("id", "sequence_id", "name", "identifier"):
        if candidate in df.columns:
            return candidate
    return "id"
