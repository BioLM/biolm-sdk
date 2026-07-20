"""SeqFrame importers."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from biolm.io.csv import load_csv
from biolm.io.fasta import load_fasta
from biolm.io.json import load_json
from biolm.seqframe.core import SeqFrame
from biolm.seqframe.rows import (
    dataframe_from_table,
    infer_id_column,
    infer_sequence_column,
)


def from_fasta(
    path: Union[str, Path],
    *,
    sequence_column: str = "sequence",
    id_column: str = "id",
    molecule_type: Optional[str] = None,
) -> SeqFrame:
    rows = load_fasta(path)
    return SeqFrame.from_rows(
        rows,
        sequence_column=sequence_column,
        id_column=id_column,
        molecule_type=molecule_type,
    )


def from_csv(
    path: Union[str, Path],
    *,
    sequence_column: str = "sequence",
    id_column: Optional[str] = None,
    molecule_type: Optional[str] = None,
) -> SeqFrame:
    rows = load_csv(path, sequence_key=sequence_column)
    id_col = id_column or "id"
    return SeqFrame.from_rows(
        rows,
        sequence_column=sequence_column,
        id_column=id_col,
        molecule_type=molecule_type,
    )


def from_jsonl(
    path: Union[str, Path],
    *,
    sequence_column: str = "sequence",
    id_column: Optional[str] = None,
    molecule_type: Optional[str] = None,
) -> SeqFrame:
    rows = load_json(path)
    id_col = id_column or "id"
    return SeqFrame.from_rows(
        rows,
        sequence_column=sequence_column,
        id_column=id_col,
        molecule_type=molecule_type,
    )


def from_protocol(
    run_or_path: Any,
    *,
    sequence_column: Optional[str] = None,
    id_column: Optional[str] = None,
    molecule_type: Optional[str] = None,
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
) -> SeqFrame:
    """Build a SeqFrame from a ProtocolRun or a downloaded results zip path."""
    from biolm.protocol_runs import ProtocolRun

    if isinstance(run_or_path, ProtocolRun) or hasattr(run_or_path, "to_dataframe"):
        df = run_or_path.to_dataframe(output_dir=output_dir, overwrite=overwrite)
    else:
        path = Path(run_or_path)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    raise ValueError(f"No CSV found in protocol results zip: {path}")
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(
                "from_protocol expects a ProtocolRun, .csv file, or results .zip"
            )

    seq_col = sequence_column or infer_sequence_column(df)
    id_col = id_column or infer_id_column(df)
    normalized = dataframe_from_table(
        df,
        sequence_column=seq_col,
        id_column=id_col,
        molecule_type=molecule_type,
    )
    return SeqFrame.from_dataframe(normalized, molecule_type=molecule_type)
