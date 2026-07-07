"""
Unit tests for utility functions.
"""
import pytest
pytest.importorskip("pandas")

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from biolm.pipeline.utils import (
    compute_hamming_distance,
    compute_sequence_identity,
    deduplicate_sequences,
    hash_sequence,
    load_fasta,
    load_sequences_from_file,
    sample_sequences,
    split_sequences_by_length,
    validate_sequence,
    write_fasta,
)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    # File I/O tests

    def test_write_and_load_fasta(self):
        """Test writing and loading FASTA."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKTAYIDSAQ"]
        fasta_path = Path(self.test_dir) / "test.fasta"

        # Write
        write_fasta(sequences, fasta_path)

        # Read back
        loaded = load_fasta(fasta_path)

        self.assertEqual(loaded, sequences)

    def test_load_fasta_from_file(self):
        """Test loading FASTA using generic loader."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        fasta_path = Path(self.test_dir) / "test.fasta"

        write_fasta(sequences, fasta_path)

        # Load with generic loader
        loaded = load_sequences_from_file(fasta_path)

        self.assertEqual(loaded, sequences)

    def test_load_csv_from_file(self):
        """Test loading CSV."""
        df = pd.DataFrame({"sequence": ["MKTAYIAKQRQ", "MKLAVIDSAQ"]})
        csv_path = Path(self.test_dir) / "test.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_sequences_from_file(csv_path)

        self.assertEqual(loaded, df["sequence"].tolist())

    # Sequence comparison tests

    def test_compute_sequence_identity(self):
        """Test sequence identity calculation."""
        seq1 = "MKTAYIAKQRQ"
        seq2 = "MKTAYIAKQRQ"

        identity = compute_sequence_identity(seq1, seq2)
        self.assertEqual(identity, 1.0)

        seq3 = "MKTAYIDSAQ"
        identity2 = compute_sequence_identity(seq1, seq3)
        self.assertLess(identity2, 1.0)
        self.assertGreater(identity2, 0.0)

    def test_compute_hamming_distance(self):
        """Test Hamming distance."""
        seq1 = "MKTAYIAKQRQ"
        seq2 = "MKTAYIAKQRQ"

        distance = compute_hamming_distance(seq1, seq2)
        self.assertEqual(distance, 0)

        seq3 = "MKTAYIDSAQ"
        distance2 = compute_hamming_distance(seq1, seq3)
        self.assertGreater(distance2, 0)

    def test_compute_hamming_distance_normalized(self):
        """Test normalized Hamming distance."""
        seq1 = "MKTAYIAKQRQ"
        seq2 = "MKTAYIAKQRQ"

        distance = compute_hamming_distance(seq1, seq2, normalize=True)
        self.assertEqual(distance, 0.0)

        seq3 = "MKKKKKKKKKK"  # All different except first 2
        distance2 = compute_hamming_distance(seq1, seq3, normalize=True)
        self.assertGreater(distance2, 0.0)
        self.assertLessEqual(distance2, 1.0)

    # Deduplication tests

    def test_deduplicate_sequences_list(self):
        """Test deduplication of list."""
        sequences = ["MKTAY", "MKLAY", "MKTAY", "MKTAY", "MKLAY"]

        dedup = deduplicate_sequences(sequences)

        self.assertEqual(len(dedup), 2)
        self.assertEqual(set(dedup), {"MKTAY", "MKLAY"})

    def test_deduplicate_sequences_dataframe(self):
        """Test deduplication of DataFrame."""
        df = pd.DataFrame({"sequence": ["MKTAY", "MKLAY", "MKTAY"], "value": [1, 2, 3]})

        df_dedup = deduplicate_sequences(df)

        self.assertEqual(len(df_dedup), 2)

    # Hash tests

    def test_hash_sequence(self):
        """Test sequence hashing."""
        seq1 = "MKTAYIAKQRQ"

        hash1 = hash_sequence(seq1)
        hash2 = hash_sequence(seq1)

        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 32)

        # Different sequence = different hash
        hash3 = hash_sequence("MKLAVIDSAQ")
        self.assertNotEqual(hash1, hash3)

    # Validation tests

    def test_validate_sequence_protein(self):
        """Test protein sequence validation."""
        valid_seq = "MKTAYIAKQRQ"
        self.assertTrue(validate_sequence(valid_seq, "protein"))

        invalid_seq = "MKTAYIZAKQRQ"  # Z is not standard
        self.assertFalse(validate_sequence(invalid_seq, "protein"))

    def test_validate_sequence_dna(self):
        """Test DNA sequence validation."""
        valid_seq = "ATCG"
        self.assertTrue(validate_sequence(valid_seq, "dna"))

        invalid_seq = "ATCGM"
        self.assertFalse(validate_sequence(invalid_seq, "dna"))

    # Length binning tests

    def test_split_sequences_by_length(self):
        """Test splitting sequences by length."""
        sequences = [
            "MKTAY",  # 5
            "MKTAYIAKQRQ",  # 11
            "MKTAYIDSAQRQGHQAMAEIKQ",  # 22
            "M" * 150,  # 150
        ]

        bins = split_sequences_by_length(sequences, [10, 20, 100])

        self.assertEqual(len(bins["<10"]), 1)
        self.assertEqual(len(bins["10-20"]), 1)
        self.assertEqual(len(bins["20-100"]), 1)
        self.assertEqual(len(bins[">100"]), 1)

    # Sampling tests

    def test_sample_sequences_list(self):
        """Test sampling from list."""
        sequences = [f"SEQ{i}" for i in range(100)]

        sampled = sample_sequences(sequences, n=10, method="random", random_seed=42)

        self.assertEqual(len(sampled), 10)
        self.assertTrue(all(s in sequences for s in sampled))

    def test_sample_sequences_dataframe_random(self):
        """Test sampling from DataFrame (random)."""
        df = pd.DataFrame(
            {"sequence": [f"SEQ{i}" for i in range(100)], "score": list(range(100))}
        )

        df_sampled = sample_sequences(df, n=10, method="random", random_seed=42)

        self.assertEqual(len(df_sampled), 10)

    def test_sample_sequences_dataframe_top(self):
        """Test sampling from DataFrame (top)."""
        df = pd.DataFrame(
            {"sequence": [f"SEQ{i}" for i in range(100)], "score": list(range(100))}
        )

        df_sampled = sample_sequences(df, n=10, method="top", score_column="score")

        self.assertEqual(len(df_sampled), 10)
        # Should have highest scores
        self.assertEqual(df_sampled["score"].min(), 90)


if __name__ == "__main__":
    unittest.main()
