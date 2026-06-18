"""
Tests for GenerationStage WorkingSet path and previously-untested coverage gaps.

Covers:
  - FilterStage batch mode actually removes rows (Bug 6 regression)
  - mark_stage_complete signature (Bug 1 regression)
  - GenerativePipeline end-to-end with SequenceSourceConfig (no API)
  - GenerationStage.process_ws() returns WorkingSet
  - process_ws forwards ws_ids for structure-lookup scoping
  - GenerationStage to_spec() / roundtrip
  - Filter after generation correctly prunes
  - GenerationStage always re-runs on resume
"""

import asyncio
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("duckdb")

from biolm.pipeline.base import StageResult, WorkingSet
from biolm.pipeline.datastore_duckdb import DuckDBDataStore
from biolm.pipeline.generative import (
    DirectGenerationConfig,
    GenerationStage,
    GenerativePipeline,
    RemaskingConfig,
    SequenceSourceConfig,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ds(tmp_path):
    ds = DuckDBDataStore(
        db_path=tmp_path / "test.duckdb",
        data_dir=tmp_path / "data",
    )
    yield ds
    ds.close()


# ---------------------------------------------------------------------------
# 1.  FilterStage batch mode actually removes rows (Bug 6 regression)
# ---------------------------------------------------------------------------


class TestFilterStageBatchMode:
    """FilterStage.process_ws() must shrink the WorkingSet when sequences fail."""

    def test_length_filter_removes_short_sequences(self, tmp_path):
        from biolm.pipeline.data import DataPipeline, FilterStage
        from biolm.pipeline.filters import SequenceLengthFilter

        ds = DuckDBDataStore(db_path=tmp_path / "f1.duckdb", data_dir=tmp_path / "d1")
        sequences = ["MK", "MKTAYIAKQRQ", "MKLAVIDSAQ"]
        pipeline = DataPipeline(
            sequences=sequences, datastore=ds, output_dir=tmp_path, verbose=False
        )
        pipeline.add_stage(FilterStage(name="len_filter", filter_func=SequenceLengthFilter(min_length=5)))
        pipeline.run()

        final = pipeline.get_final_data()
        assert len(final) == 2, f"Expected 2 sequences (len>=5), got {len(final)}"
        assert all(final["sequence"].str.len() >= 5)

    def test_threshold_filter_removes_low_scoring_sequences(self, tmp_path):
        from biolm.pipeline.data import DataPipeline, FilterStage
        from biolm.pipeline.filters import ThresholdFilter

        ds = DuckDBDataStore(db_path=tmp_path / "f2.duckdb", data_dir=tmp_path / "d2")
        sequences = ["MKLLIV", "ACDEFG", "GHIKLM", "MNPQRS", "TVWXYZ"]
        pipeline = DataPipeline(
            sequences=sequences, datastore=ds, output_dir=tmp_path, verbose=False
        )

        # Seed predictions before adding the filter stage so the SQL path has data
        ws = asyncio.run(pipeline._get_initial_data_ws())
        seq_ids = sorted(ws.to_list())
        values = [80.0, 75.0, 85.0, 20.0, 15.0]
        ds.add_predictions_batch([
            {"sequence_id": sid, "prediction_type": "tm", "model_name": "mock", "value": v}
            for sid, v in zip(seq_ids, values)
        ])
        # Register the column so the SQL filter path can resolve it
        ds.register_column("tm", "mock", "predict", "def_test", "mock_stage")

        pipeline.add_stage(FilterStage(name="tm_filter", filter_func=ThresholdFilter("tm", min_value=60.0)))
        pipeline.run()

        final = pipeline.get_final_data()
        assert len(final) == 3, f"Expected 3 sequences (tm>=60), got {len(final)}"
        assert all(final["tm"] >= 60.0)

    def test_filter_does_not_pass_all_on_empty_df(self, tmp_path):
        """Calling FilterStage.process_ws with an empty WS returns empty WS."""
        from biolm.pipeline.data import FilterStage
        from biolm.pipeline.filters import SequenceLengthFilter

        ds = DuckDBDataStore(db_path=tmp_path / "f3.duckdb", data_dir=tmp_path / "d3")
        stage = FilterStage(name="f", filter_func=SequenceLengthFilter(min_length=5))
        ws_empty = WorkingSet(frozenset())

        ws_out, result = asyncio.run(stage.process_ws(ws_empty, ds))

        assert len(ws_out) == 0
        assert result.output_count == 0


# ---------------------------------------------------------------------------
# 2.  mark_stage_complete signature — Bug 1 regression
# ---------------------------------------------------------------------------


class TestMarkStageComplete:
    def test_accepts_status_kwarg(self, tmp_ds):
        """mark_stage_complete(status='completed') must not raise TypeError."""
        tmp_ds.create_pipeline_run(
            run_id="r1", pipeline_type="DataPipeline", config={}, status="running"
        )
        # This is the exact call made by BasePipeline._execute_stage_ws()
        tmp_ds.mark_stage_complete(
            stage_id="r1_stage1",
            run_id="r1",
            stage_name="stage1",
            input_count=10,
            output_count=8,
            status="completed",
        )
        assert tmp_ds.is_stage_complete("r1_stage1")

    def test_completion_row_persisted(self, tmp_ds):
        tmp_ds.create_pipeline_run(
            run_id="r2", pipeline_type="DataPipeline", config={}, status="running"
        )
        tmp_ds.mark_stage_complete(
            stage_id="r2_pred",
            run_id="r2",
            stage_name="pred",
            input_count=5,
            output_count=5,
            status="completed",
        )
        row = tmp_ds.conn.execute(
            "SELECT input_count, output_count FROM stage_completions WHERE stage_id = ?",
            ["r2_pred"],
        ).fetchone()
        assert row is not None
        assert row[0] == 5 and row[1] == 5


# ---------------------------------------------------------------------------
# 3.  GenerationStage.process_ws() returns WorkingSet (not DataFrame)
# ---------------------------------------------------------------------------


class TestGenerationStageProcessWs:
    def test_returns_workingset_type(self, tmp_ds):
        config = SequenceSourceConfig(sequences=["MKTAYIAKQRQ", "MKLAVIDSAQ"])
        stage = GenerationStage(name="gen", config=config)
        ws_in = WorkingSet(frozenset())

        ws_out, result = asyncio.run(stage.process_ws(ws_in, tmp_ds, run_id="r1"))

        assert isinstance(ws_out, WorkingSet)
        assert isinstance(result, StageResult)

    def test_correct_sequence_count(self, tmp_ds):
        config = SequenceSourceConfig(sequences=["MKTAYIAKQRQ", "MKLAVIDSAQ", "GHIKLM"])
        stage = GenerationStage(name="gen", config=config)
        ws_in = WorkingSet(frozenset())

        ws_out, result = asyncio.run(stage.process_ws(ws_in, tmp_ds, run_id="r1"))

        assert len(ws_out) == 3
        assert result.output_count == 3

    def test_empty_source_returns_empty_workingset(self, tmp_ds):
        config = SequenceSourceConfig(sequences=[])
        stage = GenerationStage(name="gen", config=config)
        ws_in = WorkingSet(frozenset())

        with warnings.catch_warnings(record=True):
            ws_out, result = asyncio.run(stage.process_ws(ws_in, tmp_ds, run_id="r1"))

        assert isinstance(ws_out, WorkingSet)
        assert len(ws_out) == 0
        assert result.output_count == 0

    def test_deduplication_in_workingset_path(self, tmp_ds):
        config = SequenceSourceConfig(
            sequences=["MKTAYIAKQRQ", "MKTAYIAKQRQ", "MKLAVIDSAQ"]
        )
        stage = GenerationStage(name="gen", config=config, deduplicate=True)
        ws_in = WorkingSet(frozenset())

        ws_out, _ = asyncio.run(stage.process_ws(ws_in, tmp_ds, run_id="r1"))

        assert len(ws_out) == 2

    def test_sequences_stored_in_duckdb(self, tmp_ds):
        """Generated sequences are persisted in DuckDB (not just in memory)."""
        config = SequenceSourceConfig(sequences=["MKTAYIAKQRQ", "MKLAVIDSAQ"])
        stage = GenerationStage(name="gen", config=config)
        ws_in = WorkingSet(frozenset())

        ws_out, _ = asyncio.run(stage.process_ws(ws_in, tmp_ds, run_id="r1"))

        df = tmp_ds.materialize_working_set(ws_out)
        assert len(df) == 2
        seqs = set(df["sequence"].str.upper())
        assert "MKTAYIAKQRQ" in seqs
        assert "MKLAVIDSAQ" in seqs


# ---------------------------------------------------------------------------
# 4.  process_ws forwards ws_ids to _dispatch_config
# ---------------------------------------------------------------------------


class TestProcessWsForwardsIds:
    def test_ws_ids_forwarded_when_ws_nonempty(self, tmp_path):
        """process_ws passes input WS IDs as ws_ids to _dispatch_config."""
        ds = DuckDBDataStore(db_path=tmp_path / "fwd.duckdb", data_dir=tmp_path / "d")
        seq_ids = ds.add_sequences_batch(["MKTAYIAKQRQ", "MKLAVIDSAQ"])
        ws_in = WorkingSet.from_ids(seq_ids)

        captured = {}

        async def mock_dispatch(cfg, datastore, df_input, context=None, run_id=None, ws_ids=None):
            captured["ws_ids"] = ws_ids
            return []

        config = SequenceSourceConfig(sequences=["MKLLIV"])
        stage = GenerationStage(name="gen", config=config)

        with patch.object(stage, "_dispatch_config", side_effect=mock_dispatch):
            with warnings.catch_warnings(record=True):
                asyncio.run(stage.process_ws(ws_in, ds, run_id="r1"))

        assert captured.get("ws_ids") is not None, "ws_ids must be forwarded from process_ws"
        assert set(captured["ws_ids"]) == set(ws_in.to_list())

    def test_ws_ids_not_forwarded_when_ws_empty(self, tmp_path):
        """process_ws does not set ws_ids when the input WorkingSet is empty."""
        ds = DuckDBDataStore(db_path=tmp_path / "fwd2.duckdb", data_dir=tmp_path / "d2")
        ws_in = WorkingSet(frozenset())

        captured = {}

        async def mock_dispatch(cfg, datastore, df_input, context=None, run_id=None, ws_ids=None):
            captured["ws_ids"] = ws_ids
            return []

        config = SequenceSourceConfig(sequences=["MKLLIV"])
        stage = GenerationStage(name="gen", config=config)

        with patch.object(stage, "_dispatch_config", side_effect=mock_dispatch):
            with warnings.catch_warnings(record=True):
                asyncio.run(stage.process_ws(ws_in, ds, run_id="r1"))

        # Empty WS → no forwarding (ws_ids stays None)
        assert captured.get("ws_ids") is None

    def test_explicit_ws_ids_kwarg_not_overwritten(self, tmp_path):
        """Caller-supplied ws_ids in kwargs takes precedence over the WorkingSet."""
        ds = DuckDBDataStore(db_path=tmp_path / "fwd3.duckdb", data_dir=tmp_path / "d3")
        seq_ids = ds.add_sequences_batch(["MKTAYIAKQRQ"])
        ws_in = WorkingSet.from_ids(seq_ids)

        captured = {}

        async def mock_dispatch(cfg, datastore, df_input, context=None, run_id=None, ws_ids=None):
            captured["ws_ids"] = ws_ids
            return []

        config = SequenceSourceConfig(sequences=["MKLLIV"])
        stage = GenerationStage(name="gen", config=config)

        explicit_ids = [999]
        with patch.object(stage, "_dispatch_config", side_effect=mock_dispatch):
            with warnings.catch_warnings(record=True):
                asyncio.run(stage.process_ws(ws_in, ds, run_id="r1", ws_ids=explicit_ids))

        assert captured.get("ws_ids") == explicit_ids, "Explicit ws_ids must not be overwritten"


# ---------------------------------------------------------------------------
# 5.  GenerativePipeline end-to-end (no network — SequenceSourceConfig)
# ---------------------------------------------------------------------------


class TestGenerativePipelineE2E:
    def test_pipeline_runs_and_returns_data(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "p1.duckdb", data_dir=tmp_path / "d1")
        pipeline = GenerativePipeline(datastore=ds, output_dir=tmp_path, verbose=False)
        pipeline.use_sequences(["MKTAYIAKQRQ", "MKLAVIDSAQ"])
        pipeline.run()

        final = pipeline.get_final_data()
        assert len(final) == 2
        assert "sequence" in final.columns

    def test_filter_after_generation_prunes_correctly(self, tmp_path):
        from biolm.pipeline.filters import SequenceLengthFilter

        ds = DuckDBDataStore(db_path=tmp_path / "p2.duckdb", data_dir=tmp_path / "d2")
        pipeline = GenerativePipeline(datastore=ds, output_dir=tmp_path, verbose=False)
        pipeline.use_sequences(["MK", "MKTAYIAKQRQ", "MKLAVIDSAQ", "A"])
        pipeline.add_filter(SequenceLengthFilter(min_length=5))
        pipeline.run()

        final = pipeline.get_final_data()
        assert all(final["sequence"].str.len() >= 5)
        assert len(final) == 2

    def test_pipeline_stage_results_populated(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "p3.duckdb", data_dir=tmp_path / "d3")
        pipeline = GenerativePipeline(datastore=ds, output_dir=tmp_path, verbose=False)
        pipeline.use_sequences(["MKTAYIAKQRQ", "MKLAVIDSAQ"])
        pipeline.run()

        assert len(pipeline.stage_results) >= 1
        gen_result = pipeline.stage_results.get("data_source")
        assert gen_result is not None
        assert gen_result.output_count == 2

    def test_pipeline_with_multiple_source_sequences(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "p4.duckdb", data_dir=tmp_path / "d4")
        seqs = [f"{'MKLLIV' * (i + 1)}"[:20] for i in range(10)]
        pipeline = GenerativePipeline(datastore=ds, output_dir=tmp_path, verbose=False)
        pipeline.use_sequences(seqs)
        pipeline.run()

        final = pipeline.get_final_data()
        # May be fewer than 10 due to deduplication
        assert len(final) >= 1
        assert len(final) <= 10

    def test_from_csv_source(self, tmp_path):
        import pandas as pd

        ds = DuckDBDataStore(db_path=tmp_path / "p5.duckdb", data_dir=tmp_path / "d5")
        csv_path = tmp_path / "seqs.csv"
        pd.DataFrame({"sequence": ["MKTAYIAKQRQ", "MKLAVIDSAQ"]}).to_csv(csv_path, index=False)

        pipeline = GenerativePipeline(datastore=ds, output_dir=tmp_path, verbose=False)
        pipeline.use_sequences(str(csv_path))
        pipeline.run()

        final = pipeline.get_final_data()
        assert len(final) == 2


# ---------------------------------------------------------------------------
# 6.  to_spec() / roundtrip for GenerationStage configs
# ---------------------------------------------------------------------------


class TestGenerationStageSpec:
    def test_sequence_source_roundtrip(self):
        from biolm.pipeline.pipeline_def import stage_from_spec

        config = SequenceSourceConfig(sequences=["MKTAYIAKQRQ", "MKLAVIDSAQ"])
        stage = GenerationStage(name="gen", config=config, deduplicate=False)
        spec = stage.to_spec()

        assert spec["type"] == "GenerationStage"
        assert spec["name"] == "gen"
        assert spec["deduplicate"] is False

        reconstructed = stage_from_spec(spec)
        assert reconstructed.name == "gen"
        assert not reconstructed.deduplicate
        assert len(reconstructed.configs) == 1

    def test_direct_generation_config_roundtrip(self):
        from biolm.pipeline.pipeline_def import stage_from_spec

        config = DirectGenerationConfig(
            model_name="protein-mpnn",
            num_sequences=50,
            temperature=0.5,
        )
        stage = GenerationStage(name="mpnn_gen", config=config)
        spec = stage.to_spec()

        reconstructed = stage_from_spec(spec)
        rc = reconstructed.configs[0]
        assert rc.model_name == "protein-mpnn"
        assert rc.num_sequences == 50
        assert rc.temperature == 0.5

    def test_remasking_config_preserves_parent_and_variants(self):
        from biolm.pipeline.pipeline_def import stage_from_spec

        config = RemaskingConfig(
            model_name="esm2-8m",
            mask_fraction=0.15,
            num_iterations=3,
            parent_sequence="MKTAYIAKQRQ",
            num_variants=20,
        )
        stage = GenerationStage(name="remask", config=config)
        spec = stage.to_spec()

        reconstructed = stage_from_spec(spec)
        rc = reconstructed.configs[0]
        assert rc.model_name == "esm2-8m"
        assert rc.parent_sequence == "MKTAYIAKQRQ"
        assert rc.num_variants == 20

    def test_multi_config_stage_spec(self):
        config1 = SequenceSourceConfig(sequences=["MKTAY"])
        config2 = DirectGenerationConfig(model_name="protein-mpnn")
        stage = GenerationStage(name="multi", configs=[config1, config2])
        spec = stage.to_spec()

        assert len(spec["configs"]) == 2

    def test_legacy_generation_config_raises_on_to_spec(self):
        """GenerationConfig (deprecated) raises NotImplementedError for to_spec."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from biolm.pipeline.generative import GenerationConfig
            with _w.catch_warnings():
                _w.simplefilter("ignore", DeprecationWarning)
                cfg = GenerationConfig(model_name="protein-mpnn")

        stage = GenerationStage(name="legacy", configs=[cfg])
        with pytest.raises(NotImplementedError):
            stage.to_spec()


# ---------------------------------------------------------------------------
# 7.  GenerationStage always re-runs on resume (stochastic — no caching)
# ---------------------------------------------------------------------------


class TestGenerationStageResume:
    def test_reload_stage_working_set_returns_none_for_generation(self, tmp_path):
        """_reload_stage_working_set must return None for GenerationStage."""
        ds = DuckDBDataStore(db_path=tmp_path / "r1.duckdb", data_dir=tmp_path / "d1")
        pipeline = GenerativePipeline(
            datastore=ds, output_dir=tmp_path, run_id="resume_run", verbose=False
        )
        pipeline.use_sequences(["MKTAYIAKQRQ"])
        gen_stage = pipeline.stages[0]

        assert isinstance(gen_stage, GenerationStage)
        ws_input = WorkingSet.from_ids([1])
        result = pipeline._reload_stage_working_set(gen_stage, ws_input)
        assert result is None, "GenerationStage must never be reloaded from cache"

    def test_pipeline_reruns_generation_on_resume(self, tmp_path):
        """Running the same GenerativePipeline twice generates fresh sequences both times."""
        ds = DuckDBDataStore(db_path=tmp_path / "r2.duckdb", data_dir=tmp_path / "d2")

        pipeline = GenerativePipeline(datastore=ds, output_dir=tmp_path, verbose=False)
        pipeline.use_sequences(["MKTAYIAKQRQ", "MKLAVIDSAQ"])
        pipeline.run()
        count_first = len(pipeline.get_final_data())

        # Second run with the same config — generation stage re-runs
        pipeline2 = GenerativePipeline(datastore=ds, output_dir=tmp_path, verbose=False)
        pipeline2.use_sequences(["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKLLIV"])
        pipeline2.run()
        count_second = len(pipeline2.get_final_data())

        # Second run has the extra sequence (dedup merges with first run's data in DB)
        assert count_second >= count_first


# ---------------------------------------------------------------------------
# 8.  WorkingSet properties used by generation flow
# ---------------------------------------------------------------------------


class TestWorkingSetProperties:
    def test_workingset_from_generation_ids_are_ints(self, tmp_ds):
        config = SequenceSourceConfig(sequences=["MKTAYIAKQRQ"])
        stage = GenerationStage(name="gen", config=config)

        ws_out, _ = asyncio.run(stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r"))

        for sid in ws_out.sequence_ids:
            assert isinstance(sid, int), f"sequence_id must be int, got {type(sid)}"

    def test_workingset_union_covers_all_generated(self, tmp_path):
        """Multiple GenerationStage runs can be combined with union."""
        ds = DuckDBDataStore(db_path=tmp_path / "u.duckdb", data_dir=tmp_path / "d")

        cfg1 = SequenceSourceConfig(sequences=["MKTAYIAKQRQ"])
        cfg2 = SequenceSourceConfig(sequences=["MKLAVIDSAQ"])
        s1 = GenerationStage(name="g1", config=cfg1)
        s2 = GenerationStage(name="g2", config=cfg2)

        ws_in = WorkingSet(frozenset())
        ws1, _ = asyncio.run(s1.process_ws(ws_in, ds, run_id="r1"))
        ws2, _ = asyncio.run(s2.process_ws(ws_in, ds, run_id="r2"))

        combined = ws1.union(ws2)
        assert len(combined) == 2


# ---------------------------------------------------------------------------
# A4.  _extract_sequences 4-format unit tests
# ---------------------------------------------------------------------------


class TestExtractSequences:
    """A4 — _extract_sequences handles the four documented response formats."""

    def test_extract_sequences_bare_dict(self):
        """Single bare dict (MPNN with unwrap_single=True) → list of one dict."""
        raw = {"sequence": "MKTAY"}
        result = GenerationStage._extract_sequences(raw)
        assert len(result) == 1
        assert result[0]["sequence"] == "MKTAY"

    def test_extract_sequences_flat_list_mpnn(self):
        """Flat list (MPNN, multiple results) → both sequences extracted."""
        raw = [{"sequence": "MKTAY"}, {"sequence": "MKLLIV"}]
        result = GenerationStage._extract_sequences(raw)
        seqs = [r["sequence"] for r in result]
        assert "MKTAY" in seqs
        assert "MKLLIV" in seqs
        assert len(result) == 2

    def test_extract_sequences_antifold_nested(self):
        """AntiFold nested format: heavy+light → sequence joined with ':'.

        The resulting dicts must include sequence='EVQLVES:DIQMTQS',
        heavy_chain='EVQLVES', and light_chain='DIQMTQS'.
        """
        raw = [
            {
                "sequences": [
                    {"heavy": "EVQLVES", "light": "DIQMTQS", "score": 0.9}
                ]
            }
        ]
        result = GenerationStage._extract_sequences(raw)
        assert len(result) == 1
        rec = result[0]
        assert rec["sequence"] == "EVQLVES:DIQMTQS"
        assert rec["heavy_chain"] == "EVQLVES"
        assert rec["light_chain"] == "DIQMTQS"

    def test_extract_sequences_dsm_double_nested(self):
        """DSM double-nested format: [[{sequence, log_prob}]] → flat list."""
        raw = [[{"sequence": "MKTAY", "log_prob": -1.2}]]
        result = GenerationStage._extract_sequences(raw)
        assert len(result) == 1
        assert result[0]["sequence"] == "MKTAY"
        assert result[0]["log_prob"] == -1.2
