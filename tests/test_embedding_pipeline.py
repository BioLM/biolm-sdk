"""
Tests for the Embed() pipeline path and embedding storage/retrieval.

Covers Bug 5 regression (embedding dict access), Embed() convenience function,
EmbeddingSpec extraction, embedding cache hits, and WorkingSet path for encode stages.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("duckdb")

from biolm.pipeline.base import StageResult, WorkingSet
from biolm.pipeline.data import DataPipeline, EmbeddingSpec, PredictionStage
from biolm.pipeline.datastore_duckdb import DuckDBDataStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ds(tmp_path):
    ds = DuckDBDataStore(
        db_path=tmp_path / "test.duckdb",
        data_dir=tmp_path / "data",
    )
    yield ds
    ds.close()


def _fake_embedding(dim: int = 320) -> np.ndarray:
    """Return a deterministic fake embedding vector."""
    return np.linspace(0.0, 1.0, dim, dtype=np.float32)


def _make_encode_stage(model_name: str = "esm2-8m") -> PredictionStage:
    """Return a PredictionStage configured for encode (embedding) action."""
    return PredictionStage(
        name="emb",
        model_name=model_name,
        action="encode",
        embedding_extractor=EmbeddingSpec("mean_representations"),
    )


def _make_mock_api(embedding: np.ndarray):
    """Return a mock BioLMApiClient whose encode() returns one embedding result."""
    mock_api = AsyncMock()
    mock_api.encode = AsyncMock(
        return_value=[{"mean_representations": embedding.tolist()}]
    )
    mock_api.shutdown = AsyncMock()
    return mock_api


# ---------------------------------------------------------------------------
# 1.  Embed convenience function
# ---------------------------------------------------------------------------


class TestEmbedFunction:
    """Embed() is a pipeline convenience function — returns a DataFrame."""

    def _run_embed(self, sequences, tmp_path, emb):
        """Run Embed() with a mocked BioLMApiClient."""
        from biolm.pipeline.data import Embed

        mock_instance = AsyncMock()
        # Embed() auto-detects key="embeddings" for ESM models (model_name contains "esm").
        # Return the correct key so EmbeddingSpec can extract and store the embedding.
        mock_instance.encode = AsyncMock(
            return_value=[{"embeddings": emb.tolist()}] * len(sequences)
        )
        mock_instance.shutdown = AsyncMock()

        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=mock_instance):
            df = Embed(
                model_name="esm2-8m",
                sequences=sequences,
                output_dir=tmp_path,
            )
        return df

    def test_returns_dataframe(self, tmp_path):
        import pandas as pd

        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        emb = _fake_embedding(320)
        df = self._run_embed(sequences, tmp_path, emb)
        assert isinstance(df, pd.DataFrame)

    def test_has_embedding_column(self, tmp_path):
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        emb = _fake_embedding(320)
        df = self._run_embed(sequences, tmp_path, emb)
        assert "embedding" in df.columns

    def test_has_sequence_column(self, tmp_path):
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        emb = _fake_embedding(320)
        df = self._run_embed(sequences, tmp_path, emb)
        assert "sequence" in df.columns or "sequence_id" in df.columns

    def test_row_count_matches_inputs(self, tmp_path):
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        emb = _fake_embedding(320)
        df = self._run_embed(sequences, tmp_path, emb)
        assert len(df) == len(sequences)

    def test_missing_embedding_extractor_raises(self):
        """PredictionStage for encode without embedding_extractor raises at construction."""
        with pytest.raises(ValueError, match="embedding_extractor"):
            PredictionStage(
                name="emb",
                model_name="esm2-8m",
                action="encode",
            )


# ---------------------------------------------------------------------------
# 2.  EmbeddingSpec extraction from API response (Bug 5 — dict access)
# ---------------------------------------------------------------------------


class TestEmbeddingSpecExtraction:
    def test_extracts_flat_list(self):
        spec = EmbeddingSpec("mean_representations")
        emb = [0.1, 0.2, 0.3]
        result = {"mean_representations": emb}

        extracted = spec(result)
        assert len(extracted) == 1
        arr, layer = extracted[0]
        assert layer is None
        np.testing.assert_allclose(arr, emb, rtol=1e-5)

    def test_extracts_nested_list_of_dicts(self):
        """EmbeddingSpec with layer selects correct sub-array from list-of-dicts format."""
        spec = EmbeddingSpec("embeddings", layer=6)
        # Multi-layer format: list of {layer: int, embedding: [...]} dicts
        per_layer = [
            {"layer": 5, "embedding": [0.1, 0.2]},
            {"layer": 6, "embedding": [0.5, 0.6, 0.7]},
        ]
        result = {"embeddings": per_layer}

        extracted = spec(result)
        assert len(extracted) == 1
        arr, layer = extracted[0]
        assert layer == 6
        np.testing.assert_allclose(arr, [0.5, 0.6, 0.7], rtol=1e-5)

    def test_extracts_all_layers_when_no_layer_filter(self):
        """EmbeddingSpec without layer= returns all layers."""
        spec = EmbeddingSpec("embeddings")
        per_layer = [
            {"layer": 5, "embedding": [0.1, 0.2]},
            {"layer": 6, "embedding": [0.5, 0.6, 0.7]},
        ]
        result = {"embeddings": per_layer}
        extracted = spec(result)
        assert len(extracted) == 2

    def test_missing_key_returns_empty(self):
        spec = EmbeddingSpec("mean_representations")
        extracted = spec({"other_key": [1, 2, 3]})
        assert extracted == []

    def test_non_dict_input_returns_empty(self):
        spec = EmbeddingSpec("mean_representations")
        assert spec("not_a_dict") == []
        assert spec(None) == []
        assert spec(42) == []

    def test_reduction_mean(self):
        spec = EmbeddingSpec("representations", reduction="mean")
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        result = {"representations": matrix}
        extracted = spec(result)
        assert len(extracted) == 1
        arr, _ = extracted[0]
        np.testing.assert_allclose(arr, [2.0, 3.0], rtol=1e-5)

    def test_reduction_first(self):
        spec = EmbeddingSpec("representations", reduction="first")
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        result = {"representations": matrix}
        extracted = spec(result)
        arr, _ = extracted[0]
        np.testing.assert_allclose(arr, [1.0, 2.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# 3.  Embedding stored in DuckDB and retrievable (end-to-end path)
# ---------------------------------------------------------------------------


class TestEmbeddingStorage:
    def test_add_and_retrieve_embedding(self, tmp_ds):
        """add_embedding() followed by get_embeddings_by_sequence() round-trip."""
        seq = "MKTAYIAKQRQ"
        seq_id = tmp_ds.add_sequence(seq)
        emb = _fake_embedding(320)

        tmp_ds.add_embedding(seq_id, "esm2-8m", emb)

        records = tmp_ds.get_embeddings_by_sequence(seq, "esm2-8m", load_data=True)
        assert len(records) == 1
        retrieved = records[0]["embedding"]
        np.testing.assert_allclose(retrieved, emb, rtol=1e-5)

    def test_embedding_cached_after_insert(self, tmp_ds):
        """A sequence with an embedding is treated as cached (no re-compute)."""
        seq = "MKLAVIDSAQ"
        seq_id = tmp_ds.add_sequence(seq)
        emb = _fake_embedding(320)
        tmp_ds.add_embedding(seq_id, "esm2-8m", emb)

        cached_ids = set(
            tmp_ds.conn.execute(
                "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer IS NULL",
                ["esm2-8m"],
            ).df()["sequence_id"].tolist()
        )
        assert seq_id in cached_ids

    def test_embedding_with_layer(self, tmp_ds):
        seq_id = tmp_ds.add_sequence("MKLLIV")
        emb = _fake_embedding(480)

        tmp_ds.add_embedding(seq_id, "esm2-8m", emb, layer=6)

        cached_layer = set(
            tmp_ds.conn.execute(
                "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer = ?",
                ["esm2-8m", 6],
            ).df()["sequence_id"].tolist()
        )
        assert seq_id in cached_layer

    def test_upsert_does_not_duplicate(self, tmp_ds):
        """Re-inserting the same embedding (same layer) updates rather than duplicating."""
        seq_id = tmp_ds.add_sequence("ACDEFG")
        emb1 = _fake_embedding(320)
        emb2 = _fake_embedding(320) * 2

        # Use an explicit layer so the ON CONFLICT key is unambiguous
        tmp_ds.add_embedding(seq_id, "esm2-8m", emb1, layer=33)
        tmp_ds.add_embedding(seq_id, "esm2-8m", emb2, layer=33)  # upsert

        count = tmp_ds.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE sequence_id = ? AND model_name = ? AND layer = ?",
            [seq_id, "esm2-8m", 33],
        ).fetchone()[0]
        assert count == 1

    def test_get_embeddings_bulk_returns_dict(self, tmp_ds):
        """get_embeddings_bulk() returns dict[int, np.ndarray]."""
        seqs = ["MKLLIV", "ACDEFG", "GHIKLM"]
        seq_ids = tmp_ds.add_sequences_batch(seqs)
        embs = {sid: _fake_embedding(320) * (i + 1) for i, sid in enumerate(seq_ids)}
        for sid, emb in embs.items():
            tmp_ds.add_embedding(sid, "esm2-8m", emb)

        result = tmp_ds.get_embeddings_bulk(seq_ids, "esm2-8m")
        assert isinstance(result, dict)
        assert len(result) == 3
        for sid in seq_ids:
            assert sid in result
            assert isinstance(result[sid], np.ndarray)

    def test_get_embeddings_bulk_values_correct(self, tmp_ds):
        """Embeddings fetched in bulk match the originally stored values."""
        seq = "MKLLIV"
        seq_id = tmp_ds.add_sequence(seq)
        emb = _fake_embedding(64)
        tmp_ds.add_embedding(seq_id, "esm2-8m", emb)

        result = tmp_ds.get_embeddings_bulk([seq_id], "esm2-8m")
        np.testing.assert_allclose(result[seq_id], emb, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4.  PredictionStage with action='encode' and WorkingSet transport
# ---------------------------------------------------------------------------


class TestEmbedStageProcessWs:
    def _make_pipeline(self, tmp_path, sequences):
        ds = DuckDBDataStore(
            db_path=tmp_path / "emb.duckdb",
            data_dir=tmp_path / "d",
        )
        pipeline = DataPipeline(
            sequences=sequences, datastore=ds, output_dir=tmp_path, verbose=False
        )
        return pipeline, ds

    def test_process_ws_returns_workingset(self, tmp_path):
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        pipeline, ds = self._make_pipeline(tmp_path, sequences)

        stage = _make_encode_stage()
        pipeline.add_stage(stage)

        ws = asyncio.run(pipeline._get_initial_data_ws())
        emb = _fake_embedding(320)

        with patch("biolmai.pipeline.data.BioLMApiClient") as MockApi:
            mock_instance = _make_mock_api(emb)
            mock_instance.encode = AsyncMock(
                return_value=[{"mean_representations": emb.tolist()}] * len(sequences)
            )
            MockApi.return_value = mock_instance

            ws_out, result = asyncio.run(
                stage.process_ws(ws, ds, run_id="r1", verbose=False)
            )

        assert isinstance(ws_out, WorkingSet)
        assert len(ws_out) == len(sequences)

    def test_embeddings_persisted_in_duckdb(self, tmp_path):
        """After process_ws(), embeddings exist in the embeddings table."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        pipeline, ds = self._make_pipeline(tmp_path, sequences)

        stage = _make_encode_stage()
        pipeline.add_stage(stage)

        ws = asyncio.run(pipeline._get_initial_data_ws())
        emb = _fake_embedding(320)

        with patch("biolmai.pipeline.data.BioLMApiClient") as MockApi:
            mock_instance = _make_mock_api(emb)
            mock_instance.encode = AsyncMock(
                return_value=[{"mean_representations": emb.tolist()}] * len(sequences)
            )
            MockApi.return_value = mock_instance
            asyncio.run(stage.process_ws(ws, ds, run_id="r1", verbose=False))

        count = ds.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model_name = 'esm2-8m'"
        ).fetchone()[0]
        assert count == len(sequences)

    def test_cached_sequences_not_recomputed(self, tmp_path):
        """Sequences already in embeddings table skip the API call."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        pipeline, ds = self._make_pipeline(tmp_path, sequences)

        stage = _make_encode_stage()
        pipeline.add_stage(stage)

        ws = asyncio.run(pipeline._get_initial_data_ws())

        # Pre-populate both embeddings
        for sid in ws.to_list():
            ds.add_embedding(sid, "esm2-8m", _fake_embedding(320))

        call_count = 0

        async def mock_encode(items, params=None):
            nonlocal call_count
            call_count += len(items)
            return [{"mean_representations": _fake_embedding(320).tolist()}] * len(items)

        with patch("biolmai.pipeline.data.BioLMApiClient") as MockApi:
            mock_instance = AsyncMock()
            mock_instance.encode = mock_encode
            mock_instance.shutdown = AsyncMock()
            MockApi.return_value = mock_instance

            asyncio.run(stage.process_ws(ws, ds, run_id="r1", verbose=False))

        assert call_count == 0, "All sequences were cached — API should not be called"

    def test_stage_result_reports_cached_count(self, tmp_path):
        """StageResult.cached_count reflects how many sequences hit cache."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKLLIV"]
        pipeline, ds = self._make_pipeline(tmp_path, sequences)

        stage = _make_encode_stage()
        pipeline.add_stage(stage)

        ws = asyncio.run(pipeline._get_initial_data_ws())
        seq_ids = ws.to_list()

        # Cache only the first sequence
        ds.add_embedding(seq_ids[0], "esm2-8m", _fake_embedding(320))

        emb = _fake_embedding(320)
        with patch("biolmai.pipeline.data.BioLMApiClient") as MockApi:
            mock_instance = AsyncMock()
            mock_instance.encode = AsyncMock(
                return_value=[{"mean_representations": emb.tolist()}] * 2  # 2 uncached
            )
            mock_instance.shutdown = AsyncMock()
            MockApi.return_value = mock_instance

            _, result = asyncio.run(
                stage.process_ws(ws, ds, run_id="r1", verbose=False)
            )

        assert result.cached_count == 1


# ---------------------------------------------------------------------------
# 5.  Context manager on DataPipeline (Bug — missing __enter__/__exit__)
# ---------------------------------------------------------------------------


class TestPipelineContextManager:
    def test_sync_context_manager_closes_datastore(self, tmp_path):
        ds = DuckDBDataStore(
            db_path=tmp_path / "ctx.duckdb",
            data_dir=tmp_path / "d",
        )
        with DataPipeline(
            sequences=["MKTAYIAKQRQ"],
            datastore=ds,
            output_dir=tmp_path,
            verbose=False,
        ) as pipeline:
            assert pipeline is not None

        # After __exit__, DuckDB connection should be closed
        # (conn.execute raises after close)
        try:
            ds.conn.execute("SELECT 1")
            closed = False
        except Exception:
            closed = True
        assert closed, "DuckDB connection should be closed after context manager exit"

    def test_context_manager_returns_pipeline(self, tmp_path):
        ds = DuckDBDataStore(
            db_path=tmp_path / "ctx2.duckdb",
            data_dir=tmp_path / "d2",
        )
        with DataPipeline(
            sequences=["MKTAYIAKQRQ"],
            datastore=ds,
            output_dir=tmp_path,
            verbose=False,
        ) as p:
            assert isinstance(p, DataPipeline)

    def test_context_manager_closes_on_exception(self, tmp_path):
        """Context manager closes datastore even when the body raises."""
        ds = DuckDBDataStore(
            db_path=tmp_path / "ctx3.duckdb",
            data_dir=tmp_path / "d3",
        )
        try:
            with DataPipeline(
                sequences=["MKTAYIAKQRQ"],
                datastore=ds,
                output_dir=tmp_path,
                verbose=False,
            ):
                raise RuntimeError("intentional test error")
        except RuntimeError:
            pass

        try:
            ds.conn.execute("SELECT 1")
            closed = False
        except Exception:
            closed = True
        assert closed

    def test_async_context_manager(self, tmp_path):
        ds = DuckDBDataStore(
            db_path=tmp_path / "ctx4.duckdb",
            data_dir=tmp_path / "d4",
        )

        async def _run():
            async with DataPipeline(
                sequences=["MKTAYIAKQRQ"],
                datastore=ds,
                output_dir=tmp_path,
                verbose=False,
            ) as p:
                assert isinstance(p, DataPipeline)

        asyncio.run(_run())

        try:
            ds.conn.execute("SELECT 1")
            closed = False
        except Exception:
            closed = True
        assert closed
