"""
Unit tests for filters.
"""
import json

import pytest
pytest.importorskip("duckdb")

import unittest

import numpy as np
import pandas as pd

from biolm.pipeline.filters import (
    CompositeFilter,
    ConservedResidueFilter,
    CustomFilter,
    DiversitySamplingFilter,
    HammingDistanceFilter,
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    ValidAminoAcidFilter,
    _validate_sql_identifier,
    combine_filters,
)
from biolm.pipeline.pipeline_def import filter_from_spec


class TestFilters(unittest.TestCase):
    """Test filter implementations."""

    def setUp(self):
        """Create test DataFrame."""
        self.df = pd.DataFrame(
            {
                "sequence": [
                    "MKTAYIAKQRQ",
                    "MKLAVIDSAQ",
                    "MKTAYIDSAQ",
                    "MKTAY",  # Short
                    "MKTAYIAKQRQGHQAMAEIKQGHQAMAEIKQ",  # Long
                ],
                "tm": [65.0, 55.0, 70.0, 45.0, 60.0],
                "plddt": [85.0, 75.0, 90.0, 70.0, 80.0],
                "temperature": [0.5, 1.0, 1.0, 1.5, 0.5],
            }
        )

    def test_threshold_filter_min(self):
        """Test ThresholdFilter with minimum."""
        filter_obj = ThresholdFilter("tm", min_value=60)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)
        self.assertTrue((df_filtered["tm"] >= 60).all())

    def test_threshold_filter_max(self):
        """Test ThresholdFilter with maximum."""
        filter_obj = ThresholdFilter("tm", max_value=65)

        df_filtered = filter_obj(self.df)

        # tm values: [65.0, 55.0, 70.0, 45.0, 60.0]
        # Values <= 65 (inclusive): 65.0, 55.0, 45.0, 60.0 = 4 values
        self.assertEqual(len(df_filtered), 4)
        self.assertTrue((df_filtered["tm"] <= 65).all())

    def test_threshold_filter_range(self):
        """Test ThresholdFilter with range."""
        filter_obj = ThresholdFilter("tm", min_value=55, max_value=65)

        df_filtered = filter_obj(self.df)

        # tm values: [65.0, 55.0, 70.0, 45.0, 60.0]
        # Values in [55, 65] (inclusive): 65.0, 55.0, 60.0 = 3 values
        self.assertEqual(len(df_filtered), 3)
        self.assertTrue((df_filtered["tm"] >= 55).all())
        self.assertTrue((df_filtered["tm"] <= 65).all())

    def test_threshold_filter_with_nan(self):
        """Test ThresholdFilter with NaN values."""
        df = self.df.copy()
        df.loc[0, "tm"] = np.nan

        # Default: exclude NaN
        filter_obj = ThresholdFilter("tm", min_value=60, keep_na=False)
        df_filtered = filter_obj(df)
        self.assertFalse(df_filtered["tm"].isna().any())

        # Keep NaN
        filter_obj = ThresholdFilter("tm", min_value=60, keep_na=True)
        df_filtered = filter_obj(df)
        self.assertTrue(df_filtered["tm"].isna().any())

    def test_sequence_length_filter_min(self):
        """Test SequenceLengthFilter with minimum."""
        filter_obj = SequenceLengthFilter(min_length=10)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 4)  # Excludes short one
        self.assertTrue((df_filtered["sequence"].str.len() >= 10).all())

    def test_sequence_length_filter_max(self):
        """Test SequenceLengthFilter with maximum."""
        filter_obj = SequenceLengthFilter(max_length=15)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 4)  # Excludes long one
        self.assertTrue((df_filtered["sequence"].str.len() <= 15).all())

    def test_sequence_length_filter_range(self):
        """Test SequenceLengthFilter with range."""
        filter_obj = SequenceLengthFilter(min_length=10, max_length=15)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)

    def test_hamming_distance_filter(self):
        """Test HammingDistanceFilter."""
        reference = "MKTAYIAKQRQ"

        filter_obj = HammingDistanceFilter(reference, max_distance=5)

        df_filtered = filter_obj(self.df)

        # Should have hamming_distance column
        self.assertIn("hamming_distance", df_filtered.columns)
        self.assertTrue((df_filtered["hamming_distance"] <= 5).all())

    def test_hamming_distance_filter_normalized(self):
        """Test HammingDistanceFilter with normalization."""
        reference = "MKTAYIAKQRQ"

        filter_obj = HammingDistanceFilter(reference, max_distance=0.5, normalize=True)

        df_filtered = filter_obj(self.df)

        self.assertIn("hamming_distance_normalized", df_filtered.columns)
        self.assertTrue((df_filtered["hamming_distance_normalized"] <= 0.5).all())

    def test_conserved_residue_filter(self):
        """Test ConservedResidueFilter."""
        # All sequences start with 'MK'
        filter_obj = ConservedResidueFilter({0: ["M"], 1: ["K"]})

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 5)  # All pass

        # Stricter filter
        filter_obj = ConservedResidueFilter(
            {
                2: ["T"],  # Position 2 must be T
            }
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 4)  # One has L at position 2

    def test_diversity_sampling_filter_random(self):
        """Test DiversitySamplingFilter with random method."""
        filter_obj = DiversitySamplingFilter(
            n_samples=3, method="random", random_seed=42
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)

    def test_diversity_sampling_filter_top(self):
        """Test DiversitySamplingFilter with top method."""
        filter_obj = DiversitySamplingFilter(
            n_samples=3, method="top", score_column="tm"
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)
        # Should be top 3 by tm
        expected_tms = sorted(self.df["tm"], reverse=True)[:3]
        actual_tms = sorted(df_filtered["tm"], reverse=True)
        self.assertEqual(actual_tms, expected_tms)

    def test_diversity_sampling_filter_spread(self):
        """Test DiversitySamplingFilter with spread method."""
        filter_obj = DiversitySamplingFilter(
            n_samples=3, method="spread", score_column="tm"
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)

    def test_custom_filter(self):
        """Test CustomFilter."""

        def my_filter(df):
            return df[df["tm"] > 60]

        filter_obj = CustomFilter(my_filter, name="tm_above_60")

        df_filtered = filter_obj(self.df)

        self.assertTrue((df_filtered["tm"] > 60).all())
        self.assertIn("tm_above_60", str(filter_obj))

    def test_combine_filters(self):
        """Test combining filters."""
        filter1 = ThresholdFilter("tm", min_value=55)
        filter2 = SequenceLengthFilter(min_length=10)

        combined = combine_filters(filter1, filter2)

        df_filtered = combined(self.df)

        # Should satisfy both filters
        self.assertTrue((df_filtered["tm"] >= 55).all())
        self.assertTrue((df_filtered["sequence"].str.len() >= 10).all())

    def test_filter_repr(self):
        """Test filter __repr__ methods."""
        f1 = ThresholdFilter("tm", min_value=60)
        self.assertIn("tm", str(f1))
        self.assertIn("60", str(f1))

        f2 = SequenceLengthFilter(min_length=10)
        self.assertIn("10", str(f2))

        f3 = HammingDistanceFilter("MKTAY", max_distance=5)
        self.assertIn("5", str(f3))


class TestFilterEdgeCases(unittest.TestCase):
    """Test filter edge cases."""

    def test_empty_dataframe(self):
        """Test filters on empty DataFrame."""
        df = pd.DataFrame({"sequence": [], "tm": []})

        filter_obj = ThresholdFilter("tm", min_value=60)
        df_filtered = filter_obj(df)

        self.assertEqual(len(df_filtered), 0)

    def test_all_filtered_out(self):
        """Test when all rows are filtered out."""
        df = pd.DataFrame({"sequence": ["MKTAY", "MKLAY"], "tm": [30.0, 35.0]})

        filter_obj = ThresholdFilter("tm", min_value=60)
        df_filtered = filter_obj(df)

        self.assertEqual(len(df_filtered), 0)

    def test_missing_column(self):
        """Test filter on missing column."""
        df = pd.DataFrame({"sequence": ["MKTAY"]})

        filter_obj = ThresholdFilter("tm", min_value=60)

        with self.assertRaises(ValueError):
            filter_obj(df)

    def test_sampling_more_than_available(self):
        """Test sampling more items than available."""
        df = pd.DataFrame({"sequence": ["MKTAY", "MKLAY"], "tm": [60.0, 65.0]})

        filter_obj = DiversitySamplingFilter(n_samples=10, method="random")
        df_filtered = filter_obj(df)

        # Should return all available
        self.assertEqual(len(df_filtered), 2)


    def test_filter_streamability_flags(self):
        """Filters have correct requires_complete_data flags."""
        from biolm.pipeline.filters import RankingFilter, ThresholdFilter
        assert ThresholdFilter("col").requires_complete_data is False
        assert RankingFilter("col", n=10).requires_complete_data is True

    def test_all_filter_subclasses_have_streamability_flag(self):
        """Every BaseFilter subclass must declare requires_complete_data."""
        import inspect
        from biolm.pipeline import filters
        subclasses = [
            cls for name, cls in inspect.getmembers(filters, inspect.isclass)
            if issubclass(cls, filters.BaseFilter) and cls is not filters.BaseFilter
        ]
        for cls in subclasses:
            self.assertIsInstance(
                cls.requires_complete_data, bool,
                f"{cls.__name__}.requires_complete_data must be bool"
            )


class TestValidateSqlIdentifier(unittest.TestCase):
    """D3 — _validate_sql_identifier security boundary tests."""

    # --- INVALID: must raise ValueError ---

    def test_rejects_semicolon_injection(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier("col; DROP TABLE--")

    def test_rejects_single_quote_injection(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier("col' OR '1'='1")

    def test_rejects_backtick(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier("col`")

    def test_rejects_double_quote_injection(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier('col"; --')

    def test_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier("")

    def test_rejects_space_in_name(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier("col with space")

    def test_rejects_leading_digit(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier("123col")

    def test_rejects_null_byte(self):
        with self.assertRaises(ValueError):
            _validate_sql_identifier("col\x00")

    def test_rejects_hyphen(self):
        """Hyphens are not valid SQL identifier chars per the regex."""
        with self.assertRaises(ValueError):
            _validate_sql_identifier("col-name")

    def test_rejects_dot(self):
        """Dots are not valid SQL identifier chars."""
        with self.assertRaises(ValueError):
            _validate_sql_identifier("schema.table")

    # --- VALID: must not raise ---

    def test_accepts_simple_name(self):
        _validate_sql_identifier("col")  # must not raise

    def test_accepts_name_with_underscores(self):
        _validate_sql_identifier("my_column")

    def test_accepts_alphanumeric_with_trailing_digit(self):
        _validate_sql_identifier("col_123")

    def test_accepts_leading_underscore(self):
        _validate_sql_identifier("_priv")

    def test_accepts_mixed_case(self):
        _validate_sql_identifier("MyColumn")

    def test_accepts_all_caps(self):
        _validate_sql_identifier("TM")


# ---------------------------------------------------------------------------
# Small helper DataFrame for round-trip tests
# ---------------------------------------------------------------------------

_RT_DF = pd.DataFrame(
    {
        "sequence": [
            "MKTAYIAKQRQ",
            "MKLAVIDSAQW",
            "ACDEFGHIKLM",
            "NQRSTVWACDE",
            "AAAABBBBCCCC",
        ],
        "tm": [65.0, 55.0, 70.0, 45.0, 60.0],
        "plddt": [85.0, 75.0, 90.0, 70.0, 80.0],
    }
)


class TestFilterToSpecRoundTrip(unittest.TestCase):
    """D2 — to_spec() round-trips for every serializable filter class."""

    # --- helpers ---

    def _assert_spec_json_serializable(self, spec: dict):
        """Assert spec is a dict and JSON-round-trips cleanly."""
        self.assertIsInstance(spec, dict)
        json_str = json.dumps(spec)
        restored = json.loads(json_str)
        self.assertEqual(spec, restored)

    def _roundtrip(self, f):
        """to_spec() → filter_from_spec() → apply to _RT_DF; assert same result."""
        spec = f.to_spec()
        self._assert_spec_json_serializable(spec)
        f2 = filter_from_spec(spec)
        result1 = f(_RT_DF.copy())
        result2 = f2(_RT_DF.copy())
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True),
            result2.reset_index(drop=True),
            check_like=True,
        )

    # --- ThresholdFilter ---

    def test_threshold_filter_to_spec_roundtrip(self):
        f = ThresholdFilter("tm", min_value=55.0, max_value=70.0, keep_na=False)
        self._roundtrip(f)

    def test_threshold_filter_to_spec_has_type_key(self):
        f = ThresholdFilter("tm", min_value=60.0)
        spec = f.to_spec()
        self.assertEqual(spec["type"], "ThresholdFilter")
        self.assertEqual(spec["column"], "tm")
        self.assertEqual(spec["min_value"], 60.0)

    # --- SequenceLengthFilter ---

    def test_sequence_length_filter_to_spec_roundtrip(self):
        f = SequenceLengthFilter(min_length=10, max_length=15)
        self._roundtrip(f)

    def test_sequence_length_filter_to_spec_has_type_key(self):
        f = SequenceLengthFilter(min_length=5)
        spec = f.to_spec()
        self.assertEqual(spec["type"], "SequenceLengthFilter")
        self.assertEqual(spec["min_length"], 5)

    # --- HammingDistanceFilter ---

    def test_hamming_distance_filter_to_spec_roundtrip(self):
        f = HammingDistanceFilter("MKTAYIAKQRQ", max_distance=5, normalize=False)
        self._roundtrip(f)

    def test_hamming_distance_filter_to_spec_normalized_roundtrip(self):
        f = HammingDistanceFilter("MKTAYIAKQRQ", max_distance=0.5, normalize=True)
        self._roundtrip(f)

    def test_hamming_distance_filter_to_spec_has_type_key(self):
        f = HammingDistanceFilter("ACDEF", max_distance=3)
        spec = f.to_spec()
        self.assertEqual(spec["type"], "HammingDistanceFilter")
        self.assertEqual(spec["reference_sequence"], "ACDEF")

    # --- RankingFilter ---

    def test_ranking_filter_to_spec_roundtrip(self):
        f = RankingFilter("tm", n=3, ascending=False, method="top")
        self._roundtrip(f)

    def test_ranking_filter_to_spec_has_type_key(self):
        f = RankingFilter("plddt", n=2, method="bottom")
        spec = f.to_spec()
        self.assertEqual(spec["type"], "RankingFilter")
        self.assertEqual(spec["column"], "plddt")
        self.assertEqual(spec["n"], 2)

    # --- ConservedResidueFilter ---

    def test_conserved_residue_filter_to_spec_roundtrip(self):
        # All sequences in _RT_DF start with 'M' at position 0
        f = ConservedResidueFilter({0: ["M"]})
        self._roundtrip(f)

    def test_conserved_residue_filter_to_spec_keys_are_strings(self):
        """JSON requires string keys; conserved_positions ints must be stringified."""
        f = ConservedResidueFilter({0: ["M"], 1: ["K"]})
        spec = f.to_spec()
        self.assertEqual(spec["type"], "ConservedResidueFilter")
        # Keys must be strings for JSON round-trip
        for k in spec["conserved_positions"]:
            self.assertIsInstance(k, str)

    def test_conserved_residue_filter_roundtrip_restores_int_keys(self):
        """filter_from_spec must convert string keys back to int."""
        f = ConservedResidueFilter({0: ["M"], 2: ["T"]})
        spec = f.to_spec()
        f2 = filter_from_spec(spec)
        self.assertIn(0, f2.conserved_positions)
        self.assertIn(2, f2.conserved_positions)

    # --- DiversitySamplingFilter ---

    def test_diversity_sampling_filter_to_spec_roundtrip(self):
        f = DiversitySamplingFilter(n_samples=3, method="random", random_seed=0)
        spec = f.to_spec()
        self._assert_spec_json_serializable(spec)
        f2 = filter_from_spec(spec)
        # DiversitySamplingFilter adds an internal _sampled_N marker column whose
        # name includes a global counter; strip it before comparing so the column
        # name difference (a pure implementation detail) doesn't fail the test.
        def _drop_marker(df):
            marker_cols = [c for c in df.columns if c.startswith("_sampled_")]
            return df.drop(columns=marker_cols)

        result1 = _drop_marker(f(_RT_DF.copy()))
        result2 = _drop_marker(f2(_RT_DF.copy()))
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True),
            result2.reset_index(drop=True),
            check_like=True,
        )

    def test_diversity_sampling_filter_to_spec_has_type_key(self):
        f = DiversitySamplingFilter(n_samples=2, method="top", score_column="tm")
        spec = f.to_spec()
        self.assertEqual(spec["type"], "DiversitySamplingFilter")
        self.assertEqual(spec["n_samples"], 2)
        self.assertEqual(spec["score_column"], "tm")

    # --- ValidAminoAcidFilter ---

    def test_valid_amino_acid_filter_to_spec_roundtrip(self):
        f = ValidAminoAcidFilter(alphabet="ACDEFGHIKLMNPQRSTVWY", verbose=False)
        self._roundtrip(f)

    def test_valid_amino_acid_filter_to_spec_has_type_key(self):
        f = ValidAminoAcidFilter(verbose=False)
        spec = f.to_spec()
        self.assertEqual(spec["type"], "ValidAminoAcidFilter")
        self.assertIn("alphabet", spec)

    # --- CompositeFilter ---

    def test_composite_filter_to_spec_roundtrip(self):
        f = CompositeFilter(
            ThresholdFilter("tm", min_value=50.0),
            SequenceLengthFilter(min_length=5),
        )
        self._roundtrip(f)

    def test_composite_filter_to_spec_has_type_key(self):
        f = CompositeFilter(
            ThresholdFilter("tm", min_value=55.0),
            SequenceLengthFilter(max_length=20),
        )
        spec = f.to_spec()
        self.assertEqual(spec["type"], "CompositeFilter")
        self.assertIsInstance(spec["filters"], list)
        self.assertEqual(len(spec["filters"]), 2)

    def test_composite_filter_nested_specs_are_json_serializable(self):
        f = CompositeFilter(
            HammingDistanceFilter("MKTAY", max_distance=10),
            ValidAminoAcidFilter(verbose=False),
        )
        spec = f.to_spec()
        # Full JSON round-trip including nested filter specs
        json_str = json.dumps(spec)
        restored = json.loads(json_str)
        self.assertEqual(restored["type"], "CompositeFilter")
        self.assertEqual(len(restored["filters"]), 2)

    # --- CustomFilter: must raise NotImplementedError ---

    def test_custom_filter_to_spec_raises_not_implemented(self):
        f = CustomFilter(lambda df: df, name="my_custom")
        with self.assertRaises(NotImplementedError):
            f.to_spec()

    def test_custom_filter_to_spec_error_mentions_name(self):
        f = CustomFilter(lambda df: df, name="special_filter")
        try:
            f.to_spec()
        except NotImplementedError as e:
            self.assertIn("special_filter", str(e))
        else:
            self.fail("Expected NotImplementedError")

    # --- combine_filters (alias for CompositeFilter) ---

    def test_combine_filters_returns_composite_filter(self):
        f = combine_filters(
            ThresholdFilter("tm", min_value=50.0),
            SequenceLengthFilter(min_length=5),
        )
        self.assertIsInstance(f, CompositeFilter)
        spec = f.to_spec()
        self.assertEqual(spec["type"], "CompositeFilter")


if __name__ == "__main__":
    unittest.main()
