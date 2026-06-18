"""
Unit tests for DataStore (DuckDBDataStore).
"""
import pytest
pytest.importorskip("duckdb")

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from biolm.pipeline.datastore import DataStore


class TestDataStore(unittest.TestCase):
    """Test DataStore functionality."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test.db"
        self.data_dir = Path(self.test_dir) / "test_data"
        self.store = DataStore(self.db_path, self.data_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        self.store.close()
        shutil.rmtree(self.test_dir)

    # Sequence tests

    def test_add_sequence(self):
        """Test adding a sequence."""
        seq = "MKTAYIAKQRQ"
        seq_id = self.store.add_sequence(seq)

        self.assertIsNotNone(seq_id)
        self.assertIsInstance(seq_id, int)

        # Verify retrieval via get_sequence
        retrieved = self.store.get_sequence(seq_id)
        self.assertEqual(retrieved, seq)

    def test_sequence_deduplication(self):
        """Test that duplicate sequences return same ID."""
        seq = "MKTAYIAKQRQ"

        id1 = self.store.add_sequence(seq)
        id2 = self.store.add_sequence(seq)

        self.assertEqual(id1, id2)

    def test_add_sequences_batch(self):
        """Test batch sequence addition."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKTAYIDSAQ"]
        seq_ids = self.store.add_sequences_batch(sequences)

        self.assertEqual(len(seq_ids), 3)
        self.assertEqual(len(set(seq_ids)), 3)  # All unique

    def test_get_sequence_id(self):
        """Test getting sequence ID."""
        seq = "MKTAYIAKQRQ"
        seq_id = self.store.add_sequence(seq)

        retrieved_id = self.store.get_sequence_id(seq)
        self.assertEqual(retrieved_id, seq_id)

        # Non-existent sequence
        self.assertIsNone(self.store.get_sequence_id("NONEXISTENT"))

    def test_get_all_sequences(self):
        """Test getting all sequences as DataFrame."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKTAYIDSAQ"]
        self.store.add_sequences_batch(sequences)

        df = self.store.get_all_sequences()

        self.assertEqual(len(df), 3)
        self.assertIn("sequence", df.columns)
        self.assertIn("sequence_id", df.columns)

    # Generation metadata tests

    def test_add_generation_metadata(self):
        """Test adding generation metadata."""
        seq_id = self.store.add_sequence("MKTAYIAKQRQ")

        meta_id = self.store.add_generation_metadata(
            seq_id,
            model_name="proteinmpnn",
            temperature=1.0,
            sampling_params={"top_k": 10, "top_p": 0.9},
        )

        self.assertIsNotNone(meta_id)

        # Retrieve metadata
        metadata = self.store.get_generation_metadata(seq_id)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]["model_name"], "proteinmpnn")
        self.assertEqual(metadata[0]["temperature"], 1.0)

    # Prediction tests

    def test_add_prediction(self):
        """Test adding a prediction."""
        seq_id = self.store.add_sequence("MKTAYIAKQRQ")

        self.store.add_prediction(seq_id, "stability", "ddg_predictor", 2.5)

        preds = self.store.get_predictions(sequence_id=seq_id)
        self.assertEqual(len(preds), 1)

    def test_add_prediction_by_sequence(self):
        """Test adding prediction by sequence string."""
        pred_id = self.store.add_prediction_by_sequence(
            "MKTAYIAKQRQ", "stability", "ddg_predictor", 2.5
        )

        self.assertIsNotNone(pred_id)

        # Verify sequence was added
        seq_id = self.store.get_sequence_id("MKTAYIAKQRQ")
        self.assertIsNotNone(seq_id)

    def test_get_predictions(self):
        """Test getting predictions by sequence_id."""
        seq_id = self.store.add_sequence("MKTAYIAKQRQ")

        self.store.add_prediction(seq_id, "stability", "model1", 2.5)
        self.store.add_prediction(seq_id, "tm", "model2", 65.0)

        # Get all predictions for sequence
        preds = self.store.get_predictions(sequence_id=seq_id)
        self.assertEqual(len(preds), 2)

        # Filter by type
        preds_stability = self.store.get_predictions(
            sequence_id=seq_id, prediction_type="stability"
        )
        self.assertEqual(len(preds_stability), 1)
        self.assertEqual(preds_stability.iloc[0]["value"], 2.5)

    def test_get_predictions_by_sequence(self):
        """Test getting predictions by sequence string."""
        self.store.add_prediction_by_sequence(
            "MKTAYIAKQRQ", "stability", "ddg_predictor", 2.5
        )

        preds = self.store.get_predictions_by_sequence("MKTAYIAKQRQ")
        self.assertEqual(len(preds), 1)

    def test_has_prediction(self):
        """Test checking if prediction exists."""
        seq = "MKTAYIAKQRQ"

        # Before adding
        self.assertFalse(self.store.has_prediction(seq, "stability", "ddg_predictor"))

        # Add prediction
        self.store.add_prediction_by_sequence(seq, "stability", "ddg_predictor", 2.5)

        # After adding
        self.assertTrue(self.store.has_prediction(seq, "stability", "ddg_predictor"))

    # Structure tests

    def test_add_structure(self):
        """Test adding a structure."""
        seq_id = self.store.add_sequence("MKTAYIAKQRQ")

        pdb_data = "ATOM  1  N   MET A   1      20.154  16.967  15.994"

        # Positional API: (sequence_id, model_name, structure_str, format='pdb', plddt=...)
        struct_id = self.store.add_structure(
            seq_id, "esmfold", pdb_data, format="pdb", plddt=85.2
        )

        self.assertIsNotNone(struct_id)

    def test_get_structure_by_id(self):
        """Test getting a structure by its primary key."""
        seq_id = self.store.add_sequence("MKTAYIAKQRQ")

        pdb_data = "ATOM  1  N   MET A   1      20.154  16.967  15.994"

        struct_id = self.store.add_structure(
            seq_id, "esmfold", pdb_data, format="pdb", plddt=85.2
        )

        # Retrieve by structure_id
        struct_dict = self.store.get_structure_by_id(struct_id)

        self.assertIsNotNone(struct_dict)
        self.assertEqual(struct_dict["plddt"], 85.2)
        self.assertEqual(struct_dict["structure_str"], pdb_data)

    def test_get_structures_by_sequence(self):
        """Test getting structures by sequence."""
        seq = "MKTAYIAKQRQ"
        seq_id = self.store.add_sequence(seq)

        pdb_data = "ATOM  1  N   MET A   1      20.154  16.967  15.994"

        self.store.add_structure(seq_id, "esmfold", pdb_data, format="pdb")
        self.store.add_structure(seq_id, "alphafold2", pdb_data, format="pdb")

        # Get all structures
        structures = self.store.get_structures_by_sequence(seq)
        self.assertEqual(len(structures), 2)

        # Filter by model
        structures_esm = self.store.get_structures_by_sequence(
            seq, model_name="esmfold"
        )
        self.assertEqual(len(structures_esm), 1)

    # Embedding tests

    def test_add_embedding(self):
        """Test adding an embedding (returns embedding_id)."""
        seq_id = self.store.add_sequence("MKTAYIAKQRQ")

        embedding = np.random.randn(1280).astype(np.float32)

        emb_id = self.store.add_embedding(seq_id, "esm2", embedding, layer=12)

        self.assertIsNotNone(emb_id)
        self.assertIsInstance(emb_id, int)

    def test_get_embedding(self):
        """Test getting an embedding by embedding_id."""
        seq_id = self.store.add_sequence("MKTAYIAKQRQ")

        embedding = np.random.randn(1280).astype(np.float32)

        emb_id = self.store.add_embedding(seq_id, "esm2", embedding, layer=12)

        # Retrieve — returns (dict, ndarray)
        result = self.store.get_embedding(emb_id)
        self.assertIsNotNone(result)
        emb_dict, emb_array = result

        self.assertIsNotNone(emb_dict)
        self.assertEqual(emb_dict["dimension"], 1280)
        self.assertEqual(emb_dict["layer"], 12)
        np.testing.assert_array_almost_equal(emb_array, embedding)

    def test_get_embeddings_by_sequence(self):
        """Test getting embeddings by sequence."""
        seq = "MKTAYIAKQRQ"
        seq_id = self.store.add_sequence(seq)

        embedding1 = np.random.randn(1280).astype(np.float32)
        embedding2 = np.random.randn(1280).astype(np.float32)

        self.store.add_embedding(seq_id, "esm2", embedding1, layer=12)
        self.store.add_embedding(seq_id, "esm2", embedding2, layer=24)

        # Get all embeddings (metadata only)
        embs = self.store.get_embeddings_by_sequence(seq, load_data=False)
        self.assertEqual(len(embs), 2)

        # Get with data — returns List[Dict] (not tuples)
        embs_data = self.store.get_embeddings_by_sequence(seq, load_data=True)
        self.assertEqual(len(embs_data), 2)
        self.assertIsInstance(embs_data[0], dict)
        self.assertIn("embedding", embs_data[0])

    # Export tests

    def test_export_to_dataframe(self):
        """Test exporting to DataFrame (prediction_type as column name)."""
        # Add sequences with predictions
        seq1 = "MKTAYIAKQRQ"
        seq2 = "MKLAVIDSAQ"

        self.store.add_prediction_by_sequence(seq1, "stability", "model1", 2.5)
        self.store.add_prediction_by_sequence(seq2, "stability", "model1", 3.1)

        # Export — column name is prediction_type only (not pred_type_model_name)
        df = self.store.export_to_dataframe(
            include_sequences=True, include_predictions=True
        )

        self.assertEqual(len(df), 2)
        self.assertIn("sequence", df.columns)
        self.assertIn("stability", df.columns)

    def test_export_to_csv(self):
        """Test exporting to CSV."""
        seq1 = "MKTAYIAKQRQ"
        self.store.add_prediction_by_sequence(seq1, "stability", "model1", 2.5)

        output_path = Path(self.test_dir) / "export.csv"
        self.store.export_to_csv(str(output_path))

        self.assertTrue(output_path.exists())

        # Read back
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 1)

    # Pipeline run tests

    def test_create_pipeline_run(self):
        """Test creating pipeline run."""
        self.store.create_pipeline_run("test_run", "DataPipeline", {"param1": "value1"})

        run = self.store.get_pipeline_run("test_run")
        self.assertIsNotNone(run)
        self.assertEqual(run["pipeline_type"], "DataPipeline")

    def test_update_pipeline_run_status(self):
        """Test updating pipeline run status."""
        self.store.create_pipeline_run("test_run", "DataPipeline", {})

        self.store.update_pipeline_run_status("test_run", "completed")

        run = self.store.get_pipeline_run("test_run")
        self.assertEqual(run["status"], "completed")

    def test_mark_stage_complete(self):
        """Test marking stage as complete (using keyword args to avoid order bugs)."""
        self.store.create_pipeline_run("test_run", "DataPipeline", {})

        self.store.mark_stage_complete(
            run_id="test_run",
            stage_name="prediction_stage",
            stage_id="stage1",
            input_count=100,
            output_count=95,
        )

        self.assertTrue(self.store.is_stage_complete("stage1"))

    def test_is_stage_complete(self):
        """Test checking if stage is complete."""
        self.assertFalse(self.store.is_stage_complete("nonexistent"))

        self.store.create_pipeline_run("test_run", "DataPipeline", {})
        self.store.mark_stage_complete(
            run_id="test_run",
            stage_name="test",
            stage_id="stage1",
            input_count=10,
            output_count=10,
        )

        self.assertTrue(self.store.is_stage_complete("stage1"))

    # Utility tests

    def test_get_stats(self):
        """Test getting database stats."""
        self.store.add_sequence("MKTAYIAKQRQ")
        self.store.add_sequence("MKLAVIDSAQ")

        stats = self.store.get_stats()

        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["sequences"], 2)
        self.assertIn("predictions", stats)
        self.assertIn("structures", stats)


class TestDataStoreContext(unittest.TestCase):
    """Test DataStore context manager."""

    def test_context_manager(self):
        """Test using DataStore as context manager."""
        test_dir = tempfile.mkdtemp()

        try:
            db_path = Path(test_dir) / "test.db"

            with DataStore(db_path) as store:
                seq_id = store.add_sequence("MKTAYIAKQRQ")
                self.assertIsNotNone(seq_id)

            # Should be closed now
            # Open again to verify data persisted
            with DataStore(db_path) as store:
                seq = store.get_sequence(seq_id)
                self.assertEqual(seq, "MKTAYIAKQRQ")

        finally:
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
