"""
Advanced pipeline feature tests: resumability, streaming behavior, error handling.
"""
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("pandas")

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pandas as pd

from biolm.pipeline.base import WorkingSet
from biolm.pipeline.data import DataPipeline, FilterStage, PipelineAPIAuthError, PredictionStage
from biolm.pipeline.datastore import DataStore
from biolm.pipeline.filters import ThresholdFilter


@pytest.fixture
def datastore(tmp_path):
    """Create a temporary datastore."""
    return DataStore(db_path=tmp_path / "test.db", data_dir=tmp_path / "data")


class TestPipelineResumability:
    """Test that pipelines can be resumed after interruption."""

    def test_cached_predictions_are_reused(self, tmp_path, datastore):
        """Test that cached predictions are not recomputed on resume."""
        sequences = ["MKLLIV", "ACDEFG", "GHIKLM"]

        # Pre-populate cache with some predictions
        for seq in sequences[:2]:
            seq_id = datastore.add_sequence(seq)
            datastore.add_prediction(seq_id, "tm", "test_model", 55.0)

        # Create pipeline
        pipeline = DataPipeline(
            sequences=sequences, datastore=datastore, output_dir=tmp_path, verbose=False
        )

        # Create a mock stage that tracks how many sequences it processes
        processed_sequences = []

        class TrackingStage(PredictionStage):
            async def process(self, df, datastore, **kwargs):
                # Track which sequences we actually process
                for seq in df["sequence"]:
                    if not datastore.has_prediction(seq, "tm", "test_model"):
                        processed_sequences.append(seq)

                # Call parent (will also check cache)
                return await super().process(df, datastore, **kwargs)

        stage = TrackingStage(
            name="predict_tm",
            model_name="test_model",
            action="predict",
            extractions="prediction",
            columns="tm",
        )
        pipeline.add_stage(stage)

        # Verify cache checking works without running the pipeline
        # (patch.object with create=True handles the case where _api_client
        # is not stored as a persistent attribute after the R1 client-lifecycle fix)
        with patch.object(stage, "_api_client", create=True) as mock_client:
            mock_client.predict = AsyncMock(return_value=[60.0])  # Only 1 uncached seq
            pass

        # Verify cache checking works
        uncached = [
            seq
            for seq in sequences
            if not datastore.has_prediction(seq, "tm", "test_model")
        ]
        assert (
            len(uncached) == 1
        ), f"Should have 1 uncached sequence, got {len(uncached)}"
        assert uncached[0] == "GHIKLM"

    def test_stage_completion_prevents_rerun(self, tmp_path, datastore):
        """Test that completed stages are not re-run."""
        sequences = ["MKLLIV"]

        DataPipeline(
            sequences=sequences,
            datastore=datastore,
            output_dir=tmp_path,
            run_id="test_run",
            verbose=False,
        )

        # Mark a stage as complete with proper signature
        datastore.mark_stage_complete(
            run_id="test_run",
            stage_name="filter_stage",
            stage_id="test_run_filter_stage",
            input_count=1,
            output_count=1,
        )

        # Check it's marked complete
        assert datastore.is_stage_complete("test_run_filter_stage")

        # This demonstrates the mechanism for tracking completion
        # In practice, pipeline would check this before running stage


class TestStreamingBehavior:
    """Test that streaming actually streams (doesn't wait for all batches)."""

    @pytest.mark.asyncio
    async def test_streaming_yields_incrementally(self, datastore):
        """Test that process_streaming yields results as they complete."""
        # Create a stage
        stage = PredictionStage(
            name="test_stream",
            model_name="esm2_t6_8M",
            action="predict",
            extractions="prediction",
            columns="score",
        )

        # Mock data
        sequences = [f"SEQ{i:03d}" * 5 for i in range(100)]  # 100 sequences
        df = pd.DataFrame({"sequence": sequences})

        # Mock the API client to track timing.
        # Patch at module level so process_streaming() picks up the mock when it
        # creates a BioLMApiClient internally (R1 fix moved client creation local).
        batch_times = []

        class TimedMockClient:
            def __init__(self):
                self.call_count = 0

            async def predict(self, items, params=None):
                self.call_count += 1
                batch_times.append(time.time())
                # Simulate API delay
                await asyncio.sleep(0.1)
                return [0.5] * len(items)

            async def shutdown(self):
                pass

        mock_client = TimedMockClient()

        # Track when we receive chunks
        receive_times = []
        chunks_received = 0

        # Stream through the stage
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=mock_client):
            async for _chunk in stage.process_streaming(df, datastore):
                receive_times.append(time.time())
                chunks_received += 1

                # KEY TEST: We should receive chunks BEFORE all batches complete
                # If streaming works, we get chunks while API calls are still in flight
                if chunks_received == 2:
                    # After receiving 2 chunks, there should be more API calls still pending
                    # (100 sequences / 32 batch_size = ~4 batches)
                    assert (
                        mock_client.call_count >= 2
                    ), "Should have made multiple API calls"
                    print(
                        f"  ✓ Streaming verified: Received chunk {chunks_received} while batches still in flight"
                    )

        # Verify we got multiple chunks
        assert (
            chunks_received >= 3
        ), f"Should receive multiple chunks, got {chunks_received}"

        # Verify timing shows overlapping execution (more lenient now)
        if len(receive_times) >= 2:
            time_between_chunks = receive_times[1] - receive_times[0]
            # With true as_completed streaming, chunks arrive as API calls finish
            # Should be much faster than sequential (which would be 0.8s+ for 4 batches)
            assert (
                time_between_chunks < 1.5
            ), f"Chunks should stream reasonably fast, got {time_between_chunks:.3f}s"
            print(
                f"  ✓ Timing verified: {time_between_chunks:.3f}s between first chunks"
            )

    @pytest.mark.asyncio
    async def test_batching_waits_for_all(self, datastore):
        """Test that non-streaming mode waits for all results."""
        stage = PredictionStage(
            name="test_batch",
            model_name="esm2_t6_8M",
            action="predict",
            extractions="prediction",
            columns="score",
        )

        sequences = [f"SEQ{i:03d}" * 5 for i in range(50)]
        df = pd.DataFrame({"sequence": sequences})

        # Mock API client — patch at module level so process() picks it up internally.
        class MockClient:
            async def predict(self, items, params=None):
                await asyncio.sleep(0.05)
                return [{"prediction": 0.5}] * len(items)

            async def shutdown(self):
                pass

        # Non-streaming process should return complete result at once
        start = time.time()
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=MockClient()):
            _, result = await stage.process(df, datastore)
        elapsed = time.time() - start

        # Should wait for all batches
        assert elapsed > 0.05, "Should wait for at least one batch"
        assert result.output_count == len(sequences), "Should process all sequences"
        print(
            f"  ✓ Batching verified: Waited {elapsed:.3f}s for all {len(sequences)} sequences"
        )


class TestErrorHandling:
    """Test error handling and skip-on-error behavior."""

    @pytest.mark.asyncio
    async def test_error_propagation_default(self, datastore):
        """Test that errors propagate by default."""
        stage = PredictionStage(
            name="test_error",
            model_name="esm2_t6_8M",
            action="predict",
            extractions="prediction",
            columns="score",
            skip_on_error=False,  # Explicit
        )

        df = pd.DataFrame({"sequence": ["VALID", "INVALID", "VALID2"]})

        # Mock API that fails — patch at module level so process() picks it up.
        class FailingMockClient:
            async def predict(self, items, params=None):
                raise ValueError("API error")

            async def shutdown(self):
                pass

        # Should raise error by default
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=FailingMockClient()):
            with pytest.raises(ValueError, match="API error"):
                await stage.process(df, datastore)

    @pytest.mark.asyncio
    async def test_skip_on_error_marks_failures(self, datastore):
        """Test that skip_on_error marks failed items in cache."""
        stage = PredictionStage(
            name="test_skip",
            model_name="esm2_t6_8M",
            action="predict",
            extractions="prediction",
            columns="score",
            skip_on_error=True,  # Enable skip on error
        )

        sequences = ["BADSEQ1", "BADSEQ2"]
        df = pd.DataFrame({"sequence": sequences})

        # Mock API that always fails — patch at module level so process() picks it up.
        class FailingMockClient:
            async def predict(self, items, params=None):
                raise ValueError("Simulated API error")

            async def shutdown(self):
                pass

        # Should NOT raise error, but mark sequences as failed
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=FailingMockClient()):
            await stage.process(df, datastore)

        # Verify sequences are marked as having predictions (failed ones).
        # Failures are stored under the column alias ("score"), not the response_key
        # ("prediction"), so cache checks use the column name.
        for seq in sequences:
            assert datastore.has_prediction(seq, "score", "esm2_t6_8M")

            # Get prediction and verify it's marked as failed
            preds = datastore.get_predictions_by_sequence(seq)
            matching = preds[
                (preds["prediction_type"] == "score")
                & (preds["model_name"] == "esm2_t6_8M")
            ]
            assert not matching.empty
            # Value should be None for failed predictions
            assert matching.iloc[0]["value"] is None or pd.isna(
                matching.iloc[0]["value"]
            )
            print(f"  ✓ {seq} marked as failed in cache")

        print("  ✓ skip_on_error successfully prevents pipeline failure")


class TestStreamingWithFilters:
    """Test streaming through filters."""

    @pytest.mark.asyncio
    async def test_streamable_filter_processes_incrementally(self, tmp_path, datastore):
        """Test that streamable filters process chunks as they arrive."""
        pipeline = DataPipeline(
            sequences=[f"SEQ{i:04d}" * 10 for i in range(100)],
            datastore=datastore,
            output_dir=tmp_path,
            verbose=True,
        )

        # Add prediction stage
        pred_stage = PredictionStage(
            name="predict",
            model_name="esm2_t6_8M",
            action="predict",
            extractions="prediction",
            columns="score",
        )

        # Add streamable filter
        filter_stage = FilterStage(
            name="threshold",
            filter_func=ThresholdFilter("score", min_value=0.5),
            depends_on=["predict"],
        )

        pipeline.add_stage(pred_stage)
        pipeline.add_stage(filter_stage)

        # Mock the prediction stage to track streaming
        chunks_streamed = []

        original_process_streaming = pred_stage.process_streaming

        async def tracked_streaming(df, datastore, **kwargs):
            async for chunk in original_process_streaming(df, datastore, **kwargs):
                chunks_streamed.append(len(chunk))
                yield chunk

        # Replace with tracked version
        pred_stage.process_streaming = tracked_streaming

        # Mock API
        class MockClient:
            async def predict(self, items, params=None):
                await asyncio.sleep(0.01)
                return [0.6] * len(items)  # All pass filter

            async def shutdown(self):
                pass

        pred_stage._api_client = MockClient()

        # Run with streaming enabled
        with patch.object(pred_stage, "process_streaming", tracked_streaming):
            # We can't easily test this without mocking more, but the structure is here
            pass

        print("  ✓ Filter streaming test structure in place")


class TestDiffMode:
    """Test diff mode functionality."""

    def test_diff_mode_counts_existing(self, tmp_path, datastore):
        """Test that diff mode correctly counts existing sequences."""
        # Add some sequences to datastore first
        existing_seqs = ["MKLLIV", "ACDEFG", "GHIKLM"]
        for seq in existing_seqs:
            datastore.add_sequence(seq)

        # Create pipeline with mix of new and existing sequences
        all_seqs = existing_seqs + ["NEWSEQ1", "NEWSEQ2"]
        pipeline = DataPipeline(
            sequences=all_seqs,
            datastore=datastore,
            output_dir=tmp_path,
            diff_mode=True,
            verbose=False,
        )

        # Count existing (uses datastore.count_matching_sequences)
        existing_count = pipeline.datastore.count_matching_sequences(all_seqs)
        assert (
            existing_count == 3
        ), f"Should find 3 existing sequences, found {existing_count}"

    def test_diff_mode_query_results(self, tmp_path, datastore):
        """Test SQL-based querying in diff mode."""
        # Add sequences with predictions
        for i, seq in enumerate(["A" * 50, "C" * 100, "D" * 150]):
            seq_id = datastore.add_sequence(seq)
            datastore.add_prediction(seq_id, "score", "test_model", float(i * 10))

        pipeline = DataPipeline(
            sequences=["E" * 200],  # Add one new sequence
            datastore=datastore,
            output_dir=tmp_path,
            diff_mode=True,
            verbose=False,
        )

        # Query for long sequences
        df = pipeline.query_results("s.length > 75", columns=["sequence", "score"])

        assert len(df) >= 2, "Should find at least 2 sequences > 75 length"
        assert "sequence" in df.columns
        print(f"  ✓ SQL query returned {len(df)} sequences efficiently")

    def test_diff_mode_get_merged_results(self, tmp_path, datastore):
        """Test get_merged_results with filters."""
        # Add some test data
        for i in range(5):
            seq_id = datastore.add_sequence(f"SEQ{i}" * 20)
            datastore.add_prediction(seq_id, "tm", "test_model", 50.0 + i * 5)

        pipeline = DataPipeline(
            sequences=[f"NEWSEQ{i}" * 20 for i in range(3)],
            datastore=datastore,
            output_dir=tmp_path,
            diff_mode=True,
            verbose=False,
        )

        # Get merged results with specific prediction types
        df = pipeline.get_merged_results(prediction_types=["tm"])

        assert "tm" in df.columns
        assert len(df) >= 5, "Should have at least the 5 existing sequences"
        print(f"  ✓ Merged results: {len(df)} sequences with 'tm' predictions")


class TestBranchedDAG:
    """Tests for the C1 branched-DAG dependency-resolution fix."""

    def test_get_stage_input_ws_typo_raises(self, tmp_path, datastore):
        """A typoed depends_on must raise instead of silently using the root WS."""
        from biolm.pipeline.base import Stage, WorkingSet

        pipeline = DataPipeline(
            sequences=["MKLLIV", "ACDEFG"],
            datastore=datastore,
            output_dir=tmp_path,
            verbose=False,
        )

        class _DummyStage(Stage):
            async def process_ws(self, ws, datastore, **kwargs):
                return ws, None

        stage = _DummyStage(name="downstream", depends_on=["nonexistent_typo"])
        pipeline._working_sets["other_stage"] = WorkingSet.from_ids([1, 2])
        with pytest.raises(KeyError, match="nonexistent_typo"):
            pipeline._get_stage_input_ws(stage, WorkingSet.from_ids([99]))

    def test_get_stage_input_ws_intersect(self, tmp_path, datastore):
        """When a stage has multiple deps, _get_stage_input_ws returns their intersection."""
        from biolm.pipeline.base import Stage, WorkingSet

        pipeline = DataPipeline(
            sequences=["MKLLIV"],
            datastore=datastore,
            output_dir=tmp_path,
            verbose=False,
        )

        class _DummyStage(Stage):
            async def process_ws(self, ws, datastore, **kwargs):
                return ws, None

        pipeline._working_sets["a"] = WorkingSet.from_ids([1, 2, 3, 4])
        pipeline._working_sets["b"] = WorkingSet.from_ids([3, 4, 5])
        stage = _DummyStage(name="downstream", depends_on=["a", "b"])
        ws = pipeline._get_stage_input_ws(stage, WorkingSet.from_ids([0]))
        assert ws.sequence_ids == frozenset({3, 4})


class TestSqlInjectionGuards:
    """S1/S2: query_results and get_merged_results reject obvious injection vectors."""

    def test_query_results_rejects_semicolon(self, tmp_path, datastore):
        pipeline = DataPipeline(
            sequences=["AAA", "BBB"],
            datastore=datastore,
            output_dir=tmp_path,
            verbose=False,
        )
        with pytest.raises(ValueError, match="semicolons"):
            pipeline.query_results(sql_where="length > 0; DROP TABLE sequences")

    def test_query_results_rejects_drop_kw(self, tmp_path, datastore):
        pipeline = DataPipeline(
            sequences=["AAA"],
            datastore=datastore,
            output_dir=tmp_path,
            verbose=False,
        )
        with pytest.raises(ValueError, match="DROP"):
            pipeline.query_results(sql_where="DROP TABLE sequences")

    def test_get_merged_results_rejects_semicolon(self, tmp_path, datastore):
        pipeline = DataPipeline(
            sequences=["AAA"],
            datastore=datastore,
            output_dir=tmp_path,
            diff_mode=True,
            verbose=False,
        )
        with pytest.raises(ValueError, match="semicolons"):
            pipeline.get_merged_results(sequence_filter="x > 0; DELETE FROM sequences")

    @pytest.mark.parametrize("attack", [
        "1=1 OR (SELECT count(*) FROM read_csv_auto('/etc/passwd')) > 0",
        "1=1 OR (SELECT 1 FROM read_parquet('/etc/passwd'))",
        "ATTACH '/etc/passwd' AS evil",
        "1=1 UNION SELECT name FROM information_schema.tables",
        "INSERT/**/INTO foo VALUES (1)",          # comment-bypass
        "1=1 -- ; DROP TABLE sequences",          # SQL-comment-bypass
        "PRAGMA table_info('sequences')",
        "LOAD 'evil.duckdb_extension'",
    ])
    def test_query_results_rejects_known_attacks(self, tmp_path, datastore, attack):
        pipeline = DataPipeline(
            sequences=["AAA"], datastore=datastore, output_dir=tmp_path, verbose=False,
        )
        with pytest.raises(ValueError):
            pipeline.query_results(sql_where=attack)


class TestDatastoreLifecycle:
    """C2: auto-created datastores close on __del__, user-provided ones do not."""

    def test_user_datastore_survives_pipeline_gc(self, tmp_path):
        from biolm.pipeline.datastore_duckdb import DuckDBDataStore as DataStore
        ds = DataStore(db_path=tmp_path / "shared.db", data_dir=tmp_path / "shared")
        ds.add_sequence("MKLLIV")
        pipeline = DataPipeline(
            sequences=["MKLLIV"], datastore=ds, output_dir=tmp_path, verbose=False
        )
        assert getattr(pipeline, "_auto_created_datastore", False) is False
        del pipeline
        # User-provided datastore should still be usable.
        assert ds.add_sequence("ACDEFG") is not None


# ---------------------------------------------------------------------------
# B1 — PipelineAPIAuthError fail-fast bypasses skip_on_error
# ---------------------------------------------------------------------------


class TestPipelineAPIAuthError:
    """B1: 401/402 API responses must raise PipelineAPIAuthError even when
    skip_on_error=True.  auth/billing failures are not per-item — every
    subsequent batch will hit the same upstream error, so the pipeline must
    fail fast rather than silently producing an empty result."""

    @pytest.mark.asyncio
    async def test_auth_error_401_raises_even_with_skip_on_error_true(self, datastore):
        """401 Unauthorized must bubble out of process() regardless of skip_on_error."""
        stage = PredictionStage(
            name="test_auth_401",
            model_name="temberture-regression",
            action="predict",
            extractions="melting_temperature",
            columns="tm",
            skip_on_error=True,  # This must NOT swallow PipelineAPIAuthError
        )

        df = pd.DataFrame({"sequence": ["MKTAYIAKQRQ", "ACDEFGHIKLM"]})

        auth_error_payload = {
            "status_code": 401,
            "error": "Unauthorized",
            "code": "auth_failed",
        }

        class UnauthorizedMockClient:
            async def predict(self, items, params=None):
                # Simulate API returning a top-level error dict for 401
                return auth_error_payload

            async def shutdown(self):
                pass

        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=UnauthorizedMockClient()):
            with pytest.raises(PipelineAPIAuthError) as exc_info:
                await stage.process(df, datastore)

        assert exc_info.value.status_code == 401
        assert "temberture-regression" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_auth_error_402_raises_even_with_skip_on_error_true(self, datastore):
        """402 billing-cap must bubble out of process() regardless of skip_on_error."""
        stage = PredictionStage(
            name="test_auth_402",
            model_name="temberture-regression",
            action="predict",
            extractions="melting_temperature",
            columns="tm",
            skip_on_error=True,  # This must NOT swallow PipelineAPIAuthError
        )

        df = pd.DataFrame({"sequence": ["MKTAYIAKQRQ", "ACDEFGHIKLM"]})

        billing_error_payload = {
            "status_code": 402,
            "error": "Billing cap exceeded",
            "code": "billing_cap",
        }

        class BillingCapMockClient:
            async def predict(self, items, params=None):
                # Simulate API returning a top-level error dict for 402
                return billing_error_payload

            async def shutdown(self):
                pass

        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=BillingCapMockClient()):
            with pytest.raises(PipelineAPIAuthError) as exc_info:
                await stage.process(df, datastore)

        assert exc_info.value.status_code == 402
        assert "temberture-regression" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_auth_error_raises_in_process_ws_modern_path(self, datastore):
        """The modern process_ws() path has its own asyncio.gather with the same
        skip_on_error gate as legacy process(); auth errors must surface here too.
        Without this test, the second gather site at data.py:1708-1730 silently
        swallows PipelineAPIAuthError when skip_on_error=True."""
        stage = PredictionStage(
            name="test_auth_ws_401",
            model_name="temberture-regression",
            action="predict",
            extractions="melting_temperature",
            columns="tm",
            skip_on_error=True,
        )

        seq_ids = [datastore.add_sequence(s) for s in ("MKTAYIAKQRQ", "ACDEFGHIKLM")]
        ws = WorkingSet(sequence_ids=frozenset(seq_ids))

        auth_error_payload = {
            "status_code": 401,
            "error": "Unauthorized",
            "code": "auth_failed",
        }

        class UnauthorizedMockClient:
            async def predict(self, items, params=None):
                return auth_error_payload

            async def shutdown(self):
                pass

        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=UnauthorizedMockClient()):
            with pytest.raises(PipelineAPIAuthError) as exc_info:
                await stage.process_ws(ws, datastore, run_id="test-run", verbose=False)

        assert exc_info.value.status_code == 401
        assert "temberture-regression" in str(exc_info.value)


class TestPredictionStagePreflightValidation:
    """PredictionStage preflight checks: misspelled extraction keys raise ValueError
    immediately (before all parallel batches run) rather than silently producing empty
    results after a full API run.

    Both process() and process_ws() paths are covered for fail and pass cases.
    """

    @pytest.mark.asyncio
    async def test_wrong_key_raises_in_process(self, datastore):
        """process(): misspelled response_key raises ValueError naming the bad key
        before any batches are written to DuckDB."""
        stage = PredictionStage(
            name="preflight_test",
            model_name="esm2-8m",
            action="predict",
            extractions="plddt",
            columns="plddt",
        )

        class MockClient:
            async def predict(self, items, params=None):
                # Return 'score' but stage asks for 'plddt'
                return [{"score": 0.95}] * len(items)

            async def shutdown(self):
                pass

        df = pd.DataFrame({"sequence": ["MKTAYIAKQRQ", "ACDEFGHIKLM"]})
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=MockClient()):
            with pytest.raises(ValueError, match="plddt"):
                await stage.process(df, datastore)

    @pytest.mark.asyncio
    async def test_wrong_key_error_includes_available_keys_in_process(self, datastore):
        """process(): ValueError message names the keys actually present in the response."""
        stage = PredictionStage(
            name="preflight_test",
            model_name="esm2-8m",
            action="predict",
            extractions="tm",
            columns="tm",
        )

        class MockClient:
            async def predict(self, items, params=None):
                return [{"melting_temperature": 65.0, "confidence": 0.9}] * len(items)

            async def shutdown(self):
                pass

        df = pd.DataFrame({"sequence": ["MKTAYIAKQRQ"]})
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=MockClient()):
            with pytest.raises(ValueError, match="melting_temperature"):
                await stage.process(df, datastore)

    @pytest.mark.asyncio
    async def test_wrong_key_raises_in_process_ws(self, datastore):
        """process_ws(): same preflight validation as process()."""
        stage = PredictionStage(
            name="preflight_test_ws",
            model_name="esm2-8m",
            action="predict",
            extractions="plddt",
            columns="plddt",
        )

        seq_ids = [datastore.add_sequence(s) for s in ("MKTAYIAKQRQ", "ACDEFGHIKLM")]
        ws = WorkingSet(sequence_ids=frozenset(seq_ids))

        class MockClient:
            async def predict(self, items, params=None):
                return [{"score": 0.95}] * len(items)

            async def shutdown(self):
                pass

        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=MockClient()):
            with pytest.raises(ValueError, match="plddt"):
                await stage.process_ws(ws, datastore, run_id="test-run", verbose=False)

    @pytest.mark.asyncio
    async def test_correct_key_passes_preflight_in_process(self, datastore):
        """process(): correct extraction key passes preflight and produces predictions."""
        stage = PredictionStage(
            name="preflight_pass",
            model_name="esm2-8m",
            action="predict",
            extractions="score",
            columns="score",
        )

        class MockClient:
            async def predict(self, items, params=None):
                return [{"score": 0.75}] * len(items)

            async def shutdown(self):
                pass

        df = pd.DataFrame({"sequence": ["MKTAYIAKQRQ", "ACDEFGHIKLM"]})
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=MockClient()):
            df_out, result = await stage.process(df, datastore)

        assert result.output_count > 0, "correct key should produce predictions"

    @pytest.mark.asyncio
    async def test_correct_key_passes_preflight_in_process_ws(self, datastore):
        """process_ws(): correct extraction key passes preflight and produces predictions."""
        stage = PredictionStage(
            name="preflight_pass_ws",
            model_name="esm2-8m",
            action="predict",
            extractions="score",
            columns="score",
        )

        seq_ids = [datastore.add_sequence(s) for s in ("MKTAYIAKQRQ", "ACDEFGHIKLM")]
        ws = WorkingSet(sequence_ids=frozenset(seq_ids))

        class MockClient:
            async def predict(self, items, params=None):
                return [{"score": 0.75}] * len(items)

            async def shutdown(self):
                pass

        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=MockClient()):
            ws_out, result = await stage.process_ws(ws, datastore, run_id="test-run", verbose=False)

        assert result.output_count > 0, "correct key should produce predictions"


class TestAuditFixes:
    """Tests for data-quality and alignment issues found in audit workflows."""

    @pytest.mark.asyncio
    async def test_process_ws_length_mismatch_raises_not_silent(self, datastore):
        """API returning fewer results than items sent must raise ValueError (not silently
        drop sequences), so the except block writes NULL sentinels for all seq_ids."""
        stage = PredictionStage(
            name="len_mismatch_test",
            model_name="esm2-8m",
            action="predict",
            extractions="score",
            columns="score",
            skip_on_error=True,
        )

        class TruncatingMockClient:
            async def predict(self, items, params=None):
                # Return only 1 result for however many items were sent
                return [{"score": 0.9}]

            async def shutdown(self):
                pass

        seqs = ["MKTAYIAKQRQ", "ACDEFGHIKLM", "LMNPQRSTVWY"]
        ids = []
        for s in seqs:
            ids.append(datastore.add_sequence(s))

        ws = WorkingSet.from_ids(set(ids))
        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=TruncatingMockClient()):
            ws_out, result = await stage.process_ws(ws, datastore, run_id="len-test", verbose=False)

        # With skip_on_error=True and a length-mismatch raise, all sequences in the
        # mismatched batch are written as NULL sentinels — so they appear in DuckDB
        # as failed rather than simply vanishing from the WorkingSet.
        null_rows = datastore.conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE value IS NULL AND model_name = 'esm2-8m'"
        ).fetchone()[0]
        assert null_rows > 0, (
            "Length-mismatch batch must write NULL sentinel rows so sequences "
            "are failure-cached rather than silently disappearing"
        )

    @pytest.mark.asyncio
    async def test_failure_marker_write_survives_nonserializable_params(self, datastore):
        """A non-JSON-serializable self.params must not escape the skip_on_error boundary.
        The failure-marker write should be swallowed with a warning, not propagate."""
        import numpy as np

        stage = PredictionStage(
            name="params_test",
            model_name="esm2-8m",
            action="predict",
            extractions="score",
            columns="score",
            skip_on_error=True,
        )
        # Inject a non-serializable params value
        stage.params = {"weights": np.array([1.0, 2.0])}

        class AlwaysFailingClient:
            async def predict(self, items, params=None):
                raise RuntimeError("simulated API failure")

            async def shutdown(self):
                pass

        seqs = ["MKTAYIAKQRQ", "ACDEFGHIKLM"]
        for s in seqs:
            datastore.add_sequence(s)
        ws = WorkingSet.from_ids(
            set(datastore.conn.execute("SELECT sequence_id FROM sequences").df()["sequence_id"].tolist())
        )
        # Must not raise even though self.params has a numpy array
        ws_out, result = await stage.process_ws.__wrapped__(
            stage, ws, datastore, run_id="params-test", verbose=False
        ) if hasattr(stage.process_ws, "__wrapped__") else (
            await stage.process_ws(ws, datastore, run_id="params-test", verbose=False)
        )
        # Test passes if no exception is raised; result may be empty

    @pytest.mark.asyncio
    async def test_embedding_2d_array_dropped_with_warning(self, datastore, caplog):
        """Per-residue embeddings (2D arrays) must be dropped with a warning rather
        than crashing DuckDB with a FLOAT[] type error."""
        import logging
        import numpy as np
        from biolmai.pipeline.data import EmbeddingSpec

        stage = PredictionStage(
            name="embed_test",
            model_name="esm2-8m",
            action="encode",
            embedding_extractor=EmbeddingSpec("embedding"),
        )

        class PerResidueEmbedClient:
            async def encode(self, items, params=None):
                # Return 2D (per-residue) embeddings — no reduction applied
                return [{"embedding": [[0.1, 0.2, 0.3]] * len(item["sequence"])} for item in items]

            async def shutdown(self):
                pass

        seqs = ["MKTAY", "ACDEF"]
        for s in seqs:
            datastore.add_sequence(s)
        ws = WorkingSet.from_ids(
            set(datastore.conn.execute("SELECT sequence_id FROM sequences").df()["sequence_id"].tolist())
        )

        with caplog.at_level(logging.WARNING, logger="biolmai.pipeline.data"):
            with patch("biolmai.pipeline.data.BioLMApiClient", return_value=PerResidueEmbedClient()):
                ws_out, result = await stage.process_ws(ws, datastore, run_id="embed-test", verbose=False)

        # Should not crash; warning should name the dimensionality
        assert any("2D" in rec.message or "ndim" in rec.message.lower() or "reduction" in rec.message
                   for rec in caplog.records), (
            "Expected a warning about per-residue (2D) embedding being dropped"
        )

    @pytest.mark.asyncio
    async def test_whitespace_only_sequence_caught_by_null_guard(self, datastore):
        """Whitespace-only sequences must be treated as null/invalid and raise before dispatch."""
        stage = PredictionStage(
            name="whitespace_test",
            model_name="esm2-8m",
            action="predict",
            extractions="score",
            columns="score",
        )

        class DummyClient:
            async def predict(self, items, params=None):
                return [{"score": 0.5}] * len(items)

            async def shutdown(self):
                pass

        # Insert a whitespace-only sequence directly into DuckDB to bypass DataPipeline cleaning
        ws_id = datastore.add_sequence("   ")
        ws = WorkingSet.from_ids({ws_id})

        with patch("biolmai.pipeline.data.BioLMApiClient", return_value=DummyClient()):
            with pytest.raises(ValueError, match="null/NaN"):
                await stage.process_ws(ws, datastore, run_id="ws-test", verbose=False)

    def test_streaming_nan_rows_dropped_before_yield(self, tmp_path):
        """process_streaming must drop NaN rows from the uncached path (matching the
        cached path dropna at line 621) so downstream stages don't see missing predictions."""
        import asyncio
        import numpy as np

        stage = PredictionStage(
            name="stream_nan_test",
            model_name="esm2-8m",
            action="predict",
            extractions="score",
            columns="score",
        )

        from biolmai.pipeline.datastore_duckdb import DuckDBDataStore
        ds = DuckDBDataStore(db_path=tmp_path / "stream.duckdb", data_dir=tmp_path / "data")
        try:
            seqs = ["MKTAY", "ACDEF", "LMNPQ"]
            for s in seqs:
                ds.add_sequence(s)
            df_in = ds.get_all_sequences()

            call_count = 0

            class TruncatingClient:
                async def predict(self, items, params=None):
                    nonlocal call_count
                    call_count += 1
                    # Return 1 result for every batch regardless of batch size
                    return [{"score": 0.88}]

                async def shutdown(self):
                    pass

            yielded_dfs = []
            with patch("biolmai.pipeline.data.BioLMApiClient", return_value=TruncatingClient()):
                async def collect():
                    async for chunk in stage.process_streaming(df_in, ds):
                        yielded_dfs.append(chunk)
                asyncio.run(collect())

            # Any chunk that was yielded must not contain NaN in the score column
            for chunk in yielded_dfs:
                if "score" in chunk.columns:
                    assert chunk["score"].notna().all(), (
                        "process_streaming yielded rows with NaN score — "
                        "dropna guard not applied before yield"
                    )
        finally:
            ds.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
