"""sf.io — import/export to common bioinformatics formats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

from biolm.io.csv import to_csv
from biolm.io.fasta import to_fasta
from biolm.io.json import to_json

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame


class IONamespace:
    def __init__(self, sf: "SeqFrame"):
        self._sf = sf

    def to_parquet(self, path: Union[str, Path]) -> "SeqFrame":
        return self._sf._materialize_to(Path(path))

    def to_fasta(self, path: Union[str, Path], *, sequence_key: str = "sequence") -> None:
        df = self._sf.collect()
        rows = df.to_dict(orient="records")
        to_fasta(rows, path, sequence_key=sequence_key)

    def to_csv(self, path: Union[str, Path]) -> None:
        df = self._sf.collect()
        to_csv(df.to_dict(orient="records"), path)

    def to_jsonl(self, path: Union[str, Path]) -> None:
        df = self._sf.collect()
        to_json(df.to_dict(orient="records"), path, jsonl=True)
