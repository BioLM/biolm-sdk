"""sf.bio — biological sequence semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from biolm.seqframe.ops import AddColumnOp
from biolm.seqframe.rows import detect_molecule_type

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame

_DNA_COMP = str.maketrans("ATCGNatcgn", "TAGCNtagcn")


class BioNamespace:
    def __init__(self, sf: "SeqFrame"):
        self._sf = sf

    def length(self, *, column: str = "length") -> "SeqFrame":
        seq_col = self._sf.schema.sequence_column
        return self._sf._with_ops(
            AddColumnOp(column, f"length({seq_col})")
        )

    def detect_type(self, *, column: str = "molecule_type") -> "SeqFrame":
        """Add a per-row molecule_type column (materialized on collect)."""
        df = self._sf.collect()
        seq_col = self._sf.schema.sequence_column
        df[column] = df[seq_col].map(detect_molecule_type)
        from biolm.seqframe.core import SeqFrame

        return SeqFrame.from_dataframe(df, molecule_type=self._sf.schema.molecule_type)

    def reverse_complement(self, *, column: str = "reverse_complement") -> "SeqFrame":
        df = self._sf.collect()
        if self._sf.schema.molecule_type not in ("dna", "rna", "unknown"):
            raise ValueError(
                "reverse_complement is only defined for DNA/RNA sequences. "
                f"SeqFrame molecule_type is {self._sf.schema.molecule_type!r}."
            )
        seq_col = self._sf.schema.sequence_column
        df[column] = df[seq_col].apply(
            lambda s: str(s).upper().translate(_DNA_COMP)[::-1]
        )
        from biolm.seqframe.core import SeqFrame

        return SeqFrame.from_dataframe(df, molecule_type=self._sf.schema.molecule_type)

    def translate(self, *, column: str = "translation") -> "SeqFrame":
        try:
            from Bio.Seq import Seq
        except ImportError as exc:
            raise ImportError(
                "biopython is required for translate(). "
                "Install with: pip install biopython"
            ) from exc

        df = self._sf.collect()
        seq_col = self._sf.schema.sequence_column
        df[column] = df[seq_col].apply(lambda s: str(Seq(str(s)).translate()))
        from biolm.seqframe.core import SeqFrame

        return SeqFrame.from_dataframe(df, molecule_type=self._sf.schema.molecule_type)
