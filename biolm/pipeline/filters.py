"""
Filter implementations for pipeline stages.

Filters can be:
- Per-sequence: Operate on individual sequences independently (streaming-compatible)
- Aggregate: Require all data before filtering (must batch)
"""

from __future__ import annotations

import itertools as _itertools
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Module-level counter for unique marker column names (avoids id() GC-reuse risk)
_FILTER_MARKER_COUNTER = _itertools.count()

_SAFE_COL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_sql_identifier(name: str) -> None:
    """Raise ValueError if *name* is not a safe SQL identifier.

    Prevents SQL injection when column names or model names are interpolated
    into SQL strings inside ``to_sql()`` methods.
    """
    if not _SAFE_COL_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: '{name}'")


class BaseFilter(ABC):
    """
    Base class for filters.

    Attributes:
        requires_complete_data: If True, filter needs all data before filtering.
                               If False, can filter sequences as they arrive (streaming).
    """

    requires_complete_data: bool = False  # Default: can stream

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter to DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        pass

    def to_sql(self, ws_table: str = "_filter_ws", model_name: Optional[str] = None) -> Optional[str]:
        """Return a complete SQL SELECT that yields surviving ``sequence_id`` values.

        The query **must** be scoped to the working set by JOINing with
        *ws_table* (a registered DuckDB table with a single ``sequence_id``
        column).  This ensures ranking/limit operations apply only to the
        current pipeline rows, not the entire datastore.

        Args:
            ws_table: Name of the registered temp table containing the working set.
            model_name: Optional model name to scope predictions to.

        Example return value::

            SELECT w.sequence_id
            FROM _filter_ws w
            INNER JOIN predictions p ON w.sequence_id = p.sequence_id
            WHERE p.prediction_type = 'tm' AND p.value >= 60.0

        Returns ``None`` (default) when the filter cannot be expressed as SQL.
        Filters that return ``None`` will be executed via DataFrame
        materialization.
        """
        return None

    def to_spec(self) -> dict:
        """Return a serializable dict describing this filter.

        Subclasses should override this. The base implementation raises
        ``NotImplementedError`` so that pipeline definition serialization
        fails early for unsupported filter types.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement to_spec(). "
            "Implement to_spec() to enable pipeline definition saving and "
            "DataPipeline.from_db() recovery."
        )

    @abstractmethod
    def __repr__(self) -> str:
        pass


class ThresholdFilter(BaseFilter):
    """
    Filter by column value threshold.

    Args:
        column: Column name to filter on
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        keep_na: Whether to keep rows with NaN values

    Example:
        >>> filter = ThresholdFilter('tm', min_value=60.0)
        >>> df_filtered = filter(df)
    """

    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        keep_na: bool = False,
    ):
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.keep_na = keep_na

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

        mask = pd.Series([True] * len(df), index=df.index)

        # Handle NaN values first
        if not self.keep_na:
            mask &= df[self.column].notna()

        # Apply min threshold
        if self.min_value is not None:
            if self.keep_na:
                mask &= (df[self.column] >= self.min_value) | df[self.column].isna()
            else:
                mask &= df[self.column] >= self.min_value

        # Apply max threshold
        if self.max_value is not None:
            if self.keep_na:
                mask &= (df[self.column] <= self.max_value) | df[self.column].isna()
            else:
                mask &= df[self.column] <= self.max_value

        return df[mask].copy()

    def to_sql(self, ws_table: str = "_filter_ws", model_name: Optional[str] = None) -> Optional[str]:
        # if keep_na=True and no bounds, everyone passes — return pass-all SQL immediately.
        if self.keep_na and self.min_value is None and self.max_value is None:
            return f"SELECT sequence_id FROM {ws_table}"

        _validate_sql_identifier(self.column)
        if not _SAFE_COL_RE.match(ws_table):
            raise ValueError(f"Invalid ws_table name: '{ws_table}'")
        if model_name is not None:
            _validate_sql_identifier(model_name.replace("-", "_"))

        col = self.column.replace("'", "''")
        model_clause = ""
        if model_name is not None:
            model_clause = f" AND p.model_name = '{model_name.replace(chr(39), chr(39)*2)}'"

        if self.keep_na:
            # F01/F02/F13: LEFT JOIN so sequences with no prediction row are also included.
            # Build value condition; NULL rows satisfy the OR p.value IS NULL clause.
            value_conditions = []
            if self.min_value is not None:
                value_conditions.append(f"p.value >= {float(self.min_value)}")
            if self.max_value is not None:
                value_conditions.append(f"p.value <= {float(self.max_value)}")
            value_cond = " AND ".join(value_conditions)
            where = f"({value_cond}) OR p.value IS NULL"
            return (
                f"SELECT s.sequence_id FROM {ws_table} s "
                f"LEFT JOIN predictions p ON p.sequence_id = s.sequence_id "
                f"AND p.prediction_type = '{col}' AND p.value IS NOT NULL{model_clause} "
                f"WHERE {where}"
            )
        else:
            # INNER JOIN path (keep_na=False): dropped rows have no prediction or fail bounds.
            conditions = []
            if self.min_value is not None:
                conditions.append(f"p.value >= {float(self.min_value)}")
            if self.max_value is not None:
                conditions.append(f"p.value <= {float(self.max_value)}")
            conditions.append("p.value IS NOT NULL")
            if model_name is not None:
                conditions.append(f"p.model_name = '{model_name.replace(chr(39), chr(39)*2)}'")
            if not conditions:
                return None
            where = " AND ".join(conditions)
            return (
                f"SELECT w.sequence_id FROM {ws_table} w "
                f"INNER JOIN predictions p ON w.sequence_id = p.sequence_id "
                f"WHERE p.prediction_type = '{col}' AND {where}"
            )

    def to_spec(self) -> dict:
        return {
            "type": "ThresholdFilter",
            "column": self.column,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "keep_na": self.keep_na,
        }

    def __repr__(self):
        parts = [f"column='{self.column}'"]
        if self.min_value is not None:
            parts.append(f"min={self.min_value}")
        if self.max_value is not None:
            parts.append(f"max={self.max_value}")
        return f"ThresholdFilter({', '.join(parts)})"


class SequenceLengthFilter(BaseFilter):
    """
    Filter by sequence length.

    Args:
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)

    Example:
        >>> filter = SequenceLengthFilter(min_length=50, max_length=500)
        >>> df_filtered = filter(df)
    """

    def __init__(
        self, min_length: Optional[int] = None, max_length: Optional[int] = None
    ):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sequence" not in df.columns:
            raise ValueError("DataFrame must have 'sequence' column")

        lengths = df["sequence"].str.len()
        mask = pd.Series([True] * len(df), index=df.index)

        if self.min_length is not None:
            mask &= lengths >= self.min_length

        if self.max_length is not None:
            mask &= lengths <= self.max_length

        return df[mask].copy()

    def to_sql(self, ws_table: str = "_filter_ws", model_name: Optional[str] = None) -> Optional[str]:
        if not _SAFE_COL_RE.match(ws_table):
            raise ValueError(f"Invalid ws_table name: '{ws_table}'")
        conditions = []
        if self.min_length is not None:
            conditions.append(f"s.length >= {int(self.min_length)}")
        if self.max_length is not None:
            conditions.append(f"s.length <= {int(self.max_length)}")
        if not conditions:
            return None
        where = " AND ".join(conditions)
        return (
            f"SELECT w.sequence_id FROM {ws_table} w "
            f"INNER JOIN sequences s ON w.sequence_id = s.sequence_id "
            f"WHERE {where}"
        )

    def to_spec(self) -> dict:
        return {
            "type": "SequenceLengthFilter",
            "min_length": self.min_length,
            "max_length": self.max_length,
        }

    def __repr__(self):
        parts = []
        if self.min_length is not None:
            parts.append(f"min={self.min_length}")
        if self.max_length is not None:
            parts.append(f"max={self.max_length}")
        return f"SequenceLengthFilter({', '.join(parts)})"


class HammingDistanceFilter(BaseFilter):
    """
    Filter by Hamming distance to a reference sequence.

    Args:
        reference_sequence: Reference sequence
        max_distance: Maximum Hamming distance (inclusive)
        min_distance: Minimum Hamming distance (inclusive)
        normalize: If True, use normalized distance (0-1)

    Example:
        >>> filter = HammingDistanceFilter('MKTAYIAKQ', max_distance=5)
        >>> df_filtered = filter(df)
    """

    def __init__(
        self,
        reference_sequence: str,
        max_distance: Optional[float] = None,
        min_distance: Optional[float] = None,
        normalize: bool = False,
    ):
        self.reference_sequence = reference_sequence
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.normalize = normalize

    @staticmethod
    def hamming_distance(seq1: str, seq2: str, normalize: bool = False) -> float:
        """Calculate Hamming distance between two sequences.

        For equal-length sequences this is the standard Hamming distance (count of
        positions where characters differ).

        For sequences of **different lengths** (F09 — edge-case note): the distance
        is computed as the number of mismatches in the overlapping prefix PLUS the
        absolute difference in lengths (each extra character in the longer sequence
        counts as one mismatch).  When ``normalize=True`` the result is divided by
        ``max(len(seq1), len(seq2))``.  This is a non-standard extension; callers
        comparing variable-length sequences should be aware that the normalized value
        is NOT equivalent to edit distance / Levenshtein normalized distance.
        """
        if len(seq1) != len(seq2):
            # Handle different lengths - only compare overlapping region
            min_len = min(len(seq1), len(seq2))
            distance = sum(c1 != c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
            distance += abs(len(seq1) - len(seq2))
        else:
            distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

        if normalize:
            max_len = max(len(seq1), len(seq2))
            return distance / max_len if max_len > 0 else 0.0
        return distance

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sequence" not in df.columns:
            raise ValueError("DataFrame must have 'sequence' column")

        # Calculate distances
        distances = df["sequence"].apply(
            lambda seq: self.hamming_distance(
                seq, self.reference_sequence, self.normalize
            )
        )

        # Store in DataFrame for later use
        col_name = (
            "hamming_distance_normalized" if self.normalize else "hamming_distance"
        )
        df = df.copy()
        df[col_name] = distances

        mask = pd.Series([True] * len(df), index=df.index)

        if self.max_distance is not None:
            mask &= distances <= self.max_distance

        if self.min_distance is not None:
            mask &= distances >= self.min_distance

        return df[mask].copy()

    def to_spec(self) -> dict:
        return {
            "type": "HammingDistanceFilter",
            "reference_sequence": self.reference_sequence,
            "max_distance": self.max_distance,
            "min_distance": self.min_distance,
            "normalize": self.normalize,
        }

    def __repr__(self):
        parts = [f"ref_len={len(self.reference_sequence)}"]
        if self.min_distance is not None:
            parts.append(f"min={self.min_distance}")
        if self.max_distance is not None:
            parts.append(f"max={self.max_distance}")
        if self.normalize:
            parts.append("normalized=True")
        return f"HammingDistanceFilter({', '.join(parts)})"


class RankingFilter(BaseFilter):
    """
    Filter by ranking - select top N or bottom N by a column value.

    This filter REQUIRES complete data to rank all sequences.

    Args:
        column: Column name to rank by
        n: Number of sequences to select
        ascending: If True, select lowest values; if False, select highest (default)
        method: Ranking method ('top' for top N, 'bottom' for bottom N, 'percentile' for top/bottom %)
        percentile: If method='percentile', the percentile threshold (0-100)

    Example:
        >>> # Top 100 by Tm
        >>> filter = RankingFilter('tm', n=100, ascending=False)
        >>> df_filtered = filter(df)
        >>>
        >>> # Bottom 50 by hamming distance
        >>> filter = RankingFilter('hamming_distance', n=50, ascending=True)
        >>> df_filtered = filter(df)
        >>>
        >>> # Top 10% by pLDDT
        >>> filter = RankingFilter('plddt', method='percentile', percentile=90)
        >>> df_filtered = filter(df)
    """

    requires_complete_data = True  # Must see all data to rank

    def __init__(
        self,
        column: str,
        n: Optional[int] = None,
        ascending: bool = False,
        method: str = "top",
        percentile: Optional[float] = None,
    ):
        self.column = column
        self.n = n
        self.ascending = ascending
        self.method = method
        self.percentile = percentile

        if method not in ["top", "bottom", "percentile"]:
            raise ValueError(
                f"method must be 'top', 'bottom', or 'percentile', got '{method}'"
            )

        if method == "percentile" and percentile is None:
            raise ValueError("percentile must be specified when method='percentile'")

        if method in ["top", "bottom"] and n is None:
            raise ValueError(f"n must be specified when method='{method}'")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

        # Remove NaN values
        df_clean = df[df[self.column].notna()].copy()

        if len(df_clean) == 0:
            return df_clean

        if self.method == "percentile":
            # Calculate threshold
            if self.ascending:
                # Bottom percentile
                threshold = df_clean[self.column].quantile(self.percentile / 100.0)
                return df_clean[df_clean[self.column] <= threshold].copy()
            else:
                # Top percentile
                threshold = df_clean[self.column].quantile(
                    1.0 - self.percentile / 100.0
                )
                return df_clean[df_clean[self.column] >= threshold].copy()

        else:
            # Top or bottom N
            # ascending=True means select lowest values (nsmallest)
            # ascending=False means select highest values (nlargest)
            if self.method == "bottom" or self.ascending:
                return df_clean.nsmallest(
                    min(self.n, len(df_clean)), self.column
                ).copy()
            else:
                return df_clean.nlargest(min(self.n, len(df_clean)), self.column).copy()

    def to_sql(self, ws_table: str = "_filter_ws", model_name: Optional[str] = None) -> Optional[str]:
        _validate_sql_identifier(self.column)
        if not _SAFE_COL_RE.match(ws_table):
            raise ValueError(f"Invalid ws_table name: '{ws_table}'")
        if model_name is not None:
            _validate_sql_identifier(model_name.replace("-", "_"))
        if self.method == "percentile":
            return None  # percentile requires computing quantile — not trivially SQL
        col = self.column.replace("'", "''")
        order = "ASC" if (self.method == "bottom" or self.ascending) else "DESC"
        model_clause = ""
        if model_name is not None:
            safe_model = model_name.replace("'", "''")
            model_clause = f" AND p.model_name = '{safe_model}'"
        return (
            f"SELECT w.sequence_id "
            f"FROM {ws_table} w "
            f"INNER JOIN predictions p ON w.sequence_id = p.sequence_id "
            f"WHERE p.prediction_type = '{col}' AND p.value IS NOT NULL{model_clause} "
            f"ORDER BY p.value {order}, w.sequence_id ASC LIMIT {int(self.n)}"
        )

    def to_spec(self) -> dict:
        return {
            "type": "RankingFilter",
            "column": self.column,
            "n": self.n,
            "ascending": self.ascending,
            "method": self.method,
            "percentile": self.percentile,
        }

    def __repr__(self):
        if self.method == "percentile":
            return f"RankingFilter(column='{self.column}', percentile={self.percentile}, ascending={self.ascending})"
        else:
            return f"RankingFilter(column='{self.column}', n={self.n}, method='{self.method}')"


class CustomFilter(BaseFilter):
    """
    Apply a custom filter function.

    Args:
        func: Function that takes a DataFrame and returns a filtered DataFrame
        name: Optional name for the filter (for repr)

    Example:
        >>> def my_filter(df):
        ...     return df[df['sequence'].str.contains('M')]
        >>> filter = CustomFilter(my_filter, name='contains_M')
        >>> df_filtered = filter(df)
    """

    def __init__(
        self, func: Callable[[pd.DataFrame], pd.DataFrame], name: Optional[str] = None
    ):
        self.func = func
        self.name = name or "custom"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.func(df)

    def to_spec(self) -> dict:
        # raise NotImplementedError so pipeline serialization fails loudly
        # rather than silently storing a null dict that cannot be restored.
        raise NotImplementedError(
            f"CustomFilter '{self.name}' uses a non-serializable function and cannot "
            "be serialized for pipeline recovery. Use a BaseFilter subclass instead."
        )

    def __repr__(self):
        return f"CustomFilter(name='{self.name}')"


class ConservedResidueFilter(BaseFilter):
    """
    Filter sequences that have specific residues at specific positions.

    Args:
        conserved_positions: Dict mapping position (0-indexed) to allowed residues
                            e.g., {5: ['M', 'L'], 10: ['K']}
        reference_length: Expected sequence length (optional)

    Example:
        >>> filter = ConservedResidueFilter({107: ['H'], 109: ['H'], 126: ['H']})
        >>> df_filtered = filter(df)
    """

    def __init__(
        self,
        conserved_positions: dict[int, list[str]],
        reference_length: Optional[int] = None,
    ):
        self.conserved_positions = conserved_positions
        self.reference_length = reference_length

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sequence" not in df.columns:
            raise ValueError("DataFrame must have 'sequence' column")

        def check_conserved(seq: str) -> bool:
            # use 'is not None' so reference_length=0 is handled correctly.
            if self.reference_length is not None and len(seq) != self.reference_length:
                return False

            for pos, allowed_residues in self.conserved_positions.items():
                if pos >= len(seq):
                    return False
                if seq[pos] not in allowed_residues:
                    return False

            return True

        mask = df["sequence"].apply(check_conserved)
        return df[mask].copy()

    def to_spec(self) -> dict:
        return {
            "type": "ConservedResidueFilter",
            # Keys must be strings for JSON serialization
            "conserved_positions": {str(k): v for k, v in self.conserved_positions.items()},
            "reference_length": self.reference_length,
        }

    def __repr__(self):
        return f"ConservedResidueFilter(positions={len(self.conserved_positions)})"


class DiversitySamplingFilter(BaseFilter):
    """
    Sample diverse sequences using clustering or random sampling.

    This filter REQUIRES complete data to assess diversity.

    Args:
        n_samples: Number of sequences to sample
        method: Sampling method ('random', 'spread', 'top')
        score_column: Column to use for 'top' method
        random_seed: Random seed for reproducibility
        resample: If False, only sample if not already sampled (default: True)

    Example:
        >>> filter = DiversitySamplingFilter(n_samples=1000, method='random')
        >>> df_sampled = filter(df)
    """

    requires_complete_data = True  # Must see all data for diversity

    def __init__(
        self,
        n_samples: int,
        method: str = "random",
        score_column: Optional[str] = None,
        random_seed: Optional[int] = 42,
        resample: bool = True,
    ):
        self.n_samples = n_samples
        self.method = method
        self.score_column = score_column
        self.random_seed = random_seed
        self.resample = resample
        self._sampled_marker_col = f"_sampled_{next(_FILTER_MARKER_COUNTER)}"

        if method not in ["random", "spread", "top"]:
            raise ValueError(
                f"method must be 'random', 'spread', or 'top', got '{method}'"
            )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.n_samples:
            return df.copy()

        # Check if already sampled (if resample=False)
        if not self.resample and self._sampled_marker_col in df.columns:
            # Return previously sampled + new ones up to n_samples
            marker = df[self._sampled_marker_col].map(lambda x: bool(x) if x is not None and x == x else False)
            df_already_sampled = df[marker].copy()
            n_already = len(df_already_sampled)

            if n_already >= self.n_samples:
                # Already have enough, just return those
                return df_already_sampled.head(self.n_samples)
            else:
                # Need to add more from unsampled
                df_unsampled = df[~marker].copy()
                n_needed = self.n_samples - n_already

                if len(df_unsampled) > 0:
                    # Sample from unsampled
                    if self.method == "random":
                        df_new = df_unsampled.sample(
                            n=min(n_needed, len(df_unsampled)),
                            random_state=self.random_seed,
                        )
                    elif self.method == "top":
                        if self.score_column is None:
                            raise ValueError("score_column is required when method='top'")
                        df_sorted = df_unsampled.sort_values(
                            self.score_column, ascending=False
                        )
                        df_new = df_sorted.head(n_needed)
                    else:
                        df_sorted = df_unsampled.sort_values(
                            self.score_column
                            if self.score_column in df_unsampled.columns
                            else df_unsampled.columns[0]
                        )
                        indices = np.linspace(
                            0,
                            len(df_sorted) - 1,
                            min(n_needed, len(df_sorted)),
                            dtype=int,
                        )
                        df_new = df_sorted.iloc[indices].copy()

                    df_result = pd.concat(
                        [df_already_sampled, df_new], ignore_index=True
                    )
                else:
                    df_result = df_already_sampled

                # Mark as sampled
                df_result[self._sampled_marker_col] = True
                return df_result

        # Normal sampling (resample=True or first time)
        if self.method == "random":
            df_result = df.sample(
                n=self.n_samples, random_state=self.random_seed
            ).copy()

        elif self.method == "top":
            if self.score_column is None:
                raise ValueError("score_column must be specified for 'top' method")
            if self.score_column not in df.columns:
                raise ValueError(f"Column '{self.score_column}' not found")

            indices = df[self.score_column].nlargest(self.n_samples).index
            df_result = df.loc[indices].copy()

        elif self.method == "spread":
            # Sample with spread - divide into bins and sample evenly
            if self.score_column and self.score_column in df.columns:
                sorted_indices = df[self.score_column].argsort().values
                top_indices = sorted_indices[
                    np.linspace(0, len(sorted_indices) - 1, self.n_samples, dtype=int)
                ]
                df_result = df.iloc[top_indices].copy()
            else:
                raise ValueError(
                    f"DiversitySamplingFilter with method='spread': score_column "
                    f"'{self.score_column}' not found in DataFrame columns: "
                    f"{list(df.columns)}"
                )
        else:
            df_result = df.iloc[: self.n_samples].copy()

        # Mark as sampled
        df_result[self._sampled_marker_col] = True
        return df_result

    def to_spec(self) -> dict:
        return {
            "type": "DiversitySamplingFilter",
            "n_samples": self.n_samples,
            "method": self.method,
            "score_column": self.score_column,
            "random_seed": self.random_seed,
            "resample": self.resample,
        }

    def __repr__(self):
        return f"DiversitySamplingFilter(n={self.n_samples}, method='{self.method}')"


class ValidAminoAcidFilter(BaseFilter):
    """
    Filter sequences to only those composed of valid amino acid characters.

    Uses vectorized regex matching via ``str.match()`` (C-level regex engine),
    which is ~100x faster than ``.apply(lambda)`` at million-sequence scale.

    Args:
        alphabet: String of allowed characters (default: 20 standard amino acids)
        verbose: If True, print count of removed sequences

    Example:
        >>> filter = ValidAminoAcidFilter()
        >>> df_filtered = filter(df)
    """

    def __init__(
        self,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
        verbose: bool = True,
        column: str = "sequence",
    ):
        self.alphabet = alphabet
        self.verbose = verbose
        self.column = column
        self.pattern = re.compile(f"^[{re.escape(alphabet)}]+$")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.column not in df.columns:
            raise ValueError(f"DataFrame must have '{self.column}' column")

        mask = df[self.column].str.match(self.pattern)
        n_removed = (~mask).sum()
        if n_removed and self.verbose:
            print(
                f"  [ValidAminoAcidFilter] Removed {n_removed} sequences "
                f"with characters outside '{self.alphabet}' (column='{self.column}')"
            )
        return df[mask].copy()

    def to_sql(self, ws_table: str = "_filter_ws", model_name: Optional[str] = None) -> Optional[str]:
        # non-standard column may not exist on sequences table — fall back to
        # DataFrame materialization so we never reference a missing column in SQL.
        if self.column != "sequence":
            return None

        _validate_sql_identifier(self.column)
        if not _SAFE_COL_RE.match(ws_table):
            raise ValueError(f"Invalid ws_table name: '{ws_table}'")

        # FI-02: Validate that the alphabet only contains characters safe for a
        # DuckDB RE2 character class without escaping.  Python's re.escape() uses
        # Python regex escaping rules which differ from RE2/POSIX — using it on
        # the alphabet string can produce incorrect SQL regexes for non-standard
        # characters (e.g. '*', '.', '-').  For complex alphabets we fall back to
        # DataFrame materialization rather than risk incorrect SQL.
        if not re.match(r'^[A-Za-z0-9\-]+$', self.alphabet):
            return None  # Fall back to DataFrame materialization for complex alphabets

        # regexp_full_match() is absent in DuckDB < 0.10.0.
        # Use length(regexp_replace(...)) = length(col) which is compatible with DuckDB 0.9+.
        # This removes all valid chars and checks the remainder is empty.
        # Build the character class directly — no re.escape() needed since alphabet
        # is validated above to be alphanumeric + '-' only.
        col = f'"{self.column}"'
        return (
            f"SELECT w.sequence_id FROM {ws_table} w "
            f"INNER JOIN sequences s ON w.sequence_id = s.sequence_id "
            f"WHERE length(regexp_replace(s.{col}, '[^{self.alphabet}]', '', 'g')) "
            f"= length(s.{col}) "
            f"AND s.{col} IS NOT NULL AND s.{col} != ''"
        )

    def to_spec(self) -> dict:
        return {
            "type": "ValidAminoAcidFilter",
            "alphabet": self.alphabet,
            "verbose": self.verbose,
            "column": self.column,
        }

    def __repr__(self):
        col_str = f", column='{self.column}'" if self.column != "sequence" else ""
        return f"ValidAminoAcidFilter(alphabet='{self.alphabet}'{col_str})"


class CompositeFilter(BaseFilter):
    """
    A serializable filter that chains sub-filters sequentially.

    Unlike ``CustomFilter``, ``CompositeFilter`` is fully serializable via
    ``to_spec()`` / ``filter_from_spec()`` as long as every sub-filter is
    serializable.  It also supports SQL fast-path evaluation when all
    sub-filters implement ``to_sql()`` and exactly one SQL filter is present.

    Args:
        *filters: Sub-filters applied left to right.

    Example:
        >>> combined = CompositeFilter(
        ...     ThresholdFilter('tm', min_value=60),
        ...     SequenceLengthFilter(min_length=100),
        ... )
    """

    requires_complete_data: bool  # determined dynamically below

    def __init__(self, *filters: BaseFilter):
        self.filters = list(filters)
        # requires_complete_data is True if ANY sub-filter requires complete data
        self.requires_complete_data = any(
            getattr(f, "requires_complete_data", False) for f in self.filters
        )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for f in self.filters:
            df = f(df)
        return df

    def to_sql(self, ws_table: str = "_filter_ws", **kwargs) -> Optional[str]:
        """Return SQL only when a single sub-filter supports it.

        Full CTE-chaining for multiple SQL filters is complex to implement
        correctly (each stage's output must feed the next as a temporary
        table).  For now we support the single-SQL-filter case and fall back
        to DataFrame materialization for all other combinations.
        """
        sqls = []
        for f in self.filters:
            sql = (
                f.to_sql(ws_table=ws_table, **kwargs)
                if hasattr(f, "to_sql")
                else None
            )
            if sql is None:
                return None  # Any non-SQL filter forces materialization
            sqls.append(sql)

        if not sqls:
            return f"SELECT sequence_id FROM {ws_table}"
        if len(sqls) == 1:
            return sqls[0]
        # Multiple SQL filters: fall back to materialization (CTE chaining is complex)
        return None

    def to_spec(self) -> dict:
        return {
            "type": "CompositeFilter",
            "filters": [f.to_spec() for f in self.filters],
        }

    def __repr__(self) -> str:
        parts = ", ".join(repr(f) for f in self.filters)
        return f"CompositeFilter({parts})"


# Utility function to combine filters
def combine_filters(*filters: BaseFilter) -> BaseFilter:
    """
    Combine multiple filters into a single filter (applied sequentially).

    Returns a :class:`CompositeFilter` which is serializable (unlike the
    previous ``CustomFilter``-based implementation) and supports SQL
    fast-path evaluation when all sub-filters implement ``to_sql()``.

    Args:
        *filters: Variable number of filter objects

    Returns:
        Combined filter

    Example:
        >>> filter = combine_filters(
        ...     ThresholdFilter('tm', min_value=60),
        ...     SequenceLengthFilter(min_length=100)
        ... )
    """
    return CompositeFilter(*filters)
