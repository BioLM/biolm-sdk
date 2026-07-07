"""
Unit tests for RankingFilter.
"""
import pytest
pytest.importorskip("pandas")

import unittest

import numpy as np
import pandas as pd

from biolm.pipeline.filters import RankingFilter


class TestRankingFilter(unittest.TestCase):
    """Test RankingFilter."""

    def setUp(self):
        """Create test DataFrame."""
        self.df = pd.DataFrame(
            {
                "sequence": [f"SEQ{i}" for i in range(100)],
                "tm": np.random.uniform(40, 80, 100),
                "plddt": np.random.uniform(60, 95, 100),
            }
        )

    def test_ranking_filter_top_n(self):
        """Test top N selection."""
        filter_obj = RankingFilter("tm", n=10, ascending=False)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 10)

        # Should be top 10 by tm
        expected_top_10 = self.df.nlargest(10, "tm")
        pd.testing.assert_frame_equal(
            df_filtered.sort_values("tm", ascending=False).reset_index(drop=True),
            expected_top_10.sort_values("tm", ascending=False).reset_index(drop=True),
        )

    def test_ranking_filter_bottom_n(self):
        """Test bottom N selection."""
        filter_obj = RankingFilter("tm", n=10, ascending=True)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 10)

        # Should be bottom 10 by tm
        expected_bottom_10 = self.df.nsmallest(10, "tm")
        pd.testing.assert_frame_equal(
            df_filtered.sort_values("tm").reset_index(drop=True),
            expected_bottom_10.sort_values("tm").reset_index(drop=True),
        )

    def test_ranking_filter_percentile_top(self):
        """Test top percentile selection."""
        filter_obj = RankingFilter(
            "tm", method="percentile", percentile=10, ascending=False
        )

        df_filtered = filter_obj(self.df)

        # Should be top 10%
        threshold = self.df["tm"].quantile(0.9)
        expected_count = (self.df["tm"] >= threshold).sum()

        self.assertEqual(len(df_filtered), expected_count)
        self.assertTrue((df_filtered["tm"] >= threshold).all())

    def test_ranking_filter_percentile_bottom(self):
        """Test bottom percentile selection."""
        filter_obj = RankingFilter(
            "tm", method="percentile", percentile=10, ascending=True
        )

        df_filtered = filter_obj(self.df)

        # Should be bottom 10%
        threshold = self.df["tm"].quantile(0.1)
        expected_count = (self.df["tm"] <= threshold).sum()

        self.assertEqual(len(df_filtered), expected_count)
        self.assertTrue((df_filtered["tm"] <= threshold).all())

    def test_ranking_filter_with_nan(self):
        """Test with NaN values."""
        df = self.df.copy()
        df.loc[:10, "tm"] = np.nan

        filter_obj = RankingFilter("tm", n=10, ascending=False)
        df_filtered = filter_obj(df)

        # Should exclude NaN
        self.assertEqual(len(df_filtered), 10)
        self.assertFalse(df_filtered["tm"].isna().any())

    def test_ranking_filter_missing_column(self):
        """Test with missing column."""
        filter_obj = RankingFilter("nonexistent", n=10)

        with self.assertRaises(ValueError):
            filter_obj(self.df)

    def test_ranking_filter_method_validation(self):
        """Test method validation."""
        with self.assertRaises(ValueError):
            RankingFilter("tm", n=10, method="invalid")

    def test_ranking_filter_percentile_without_value(self):
        """Test percentile method without percentile value."""
        with self.assertRaises(ValueError):
            RankingFilter("tm", method="percentile")

    def test_ranking_filter_top_without_n(self):
        """Test top method without n value."""
        with self.assertRaises(ValueError):
            RankingFilter("tm", method="top")

    def test_ranking_filter_repr(self):
        """Test __repr__."""
        filter1 = RankingFilter("tm", n=10, method="top")
        self.assertIn("tm", str(filter1))
        self.assertIn("10", str(filter1))

        filter2 = RankingFilter("tm", method="percentile", percentile=10)
        self.assertIn("percentile", str(filter2))
        self.assertIn("10", str(filter2))


class TestResamplingFlag(unittest.TestCase):
    """Test resample flag in DiversitySamplingFilter."""

    def setUp(self):
        """Create test DataFrame."""
        self.df = pd.DataFrame(
            {
                "sequence": [f"SEQ{i}" for i in range(100)],
                "tm": np.random.uniform(40, 80, 100),
            }
        )

    def test_resample_true(self):
        """Test with resample=True (default)."""
        from biolm.pipeline.filters import DiversitySamplingFilter

        filter_obj = DiversitySamplingFilter(
            n_samples=10, method="random", resample=True
        )

        # First application
        df1 = filter_obj(self.df)
        self.assertEqual(len(df1), 10)

        # Second application on more data - should resample
        df_expanded = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    {
                        "sequence": [f"NEW{i}" for i in range(50)],
                        "tm": np.random.uniform(40, 80, 50),
                    }
                ),
            ],
            ignore_index=True,
        )

        df2 = filter_obj(df_expanded)
        self.assertEqual(len(df2), 10)
        # Could include new sequences

    def test_resample_false(self):
        """Test with resample=False."""
        from biolm.pipeline.filters import DiversitySamplingFilter

        filter_obj = DiversitySamplingFilter(
            n_samples=10, method="random", resample=False, random_seed=42
        )

        # First application
        df1 = filter_obj(self.df)
        self.assertEqual(len(df1), 10)
        sampled_seqs = set(df1["sequence"])

        # Second application on same data - should return same
        df2 = filter_obj(self.df)
        self.assertEqual(len(df2), 10)
        self.assertEqual(set(df2["sequence"]), sampled_seqs)

        # Third application with more data - should keep old + add new
        df_expanded = pd.concat(
            [
                df1,
                pd.DataFrame(
                    {
                        "sequence": [f"NEW{i}" for i in range(50)],
                        "tm": np.random.uniform(40, 80, 50),
                    }
                ),
            ],
            ignore_index=True,
        )

        df3 = filter_obj(df_expanded)
        self.assertEqual(len(df3), 10)
        # Should still contain original sampled sequences
        self.assertTrue(sampled_seqs.issubset(set(df3["sequence"])))


if __name__ == "__main__":
    unittest.main()
