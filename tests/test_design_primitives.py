"""
Tests for pipeline design primitive config types added in chance/pipeline-design-primitives:

  - SaturationMutagenesisConfig — fields, to_spec, dispatch with mocked API
  - IterativeMaskingDMSConfig   — fields, to_spec, dispatch with mocked API
  - GenerativePipeline shorthands (configs=, filters=, data_store=)
  - DirectGenerationConfig.label → generation_metadata → source_label in results()
  - BasePipeline.results() alias
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("duckdb")

from biolmai.pipeline import (
    DirectGenerationConfig,
    GenerativePipeline,
    HammingDistanceFilter,
    IterativeMaskingDMSConfig,
    SaturationMutagenesisConfig,
    SequenceSourceConfig,
    ValidAminoAcidFilter,
    combine_filters,
    DuckDBDataStore,
)
from biolmai.pipeline.base import WorkingSet
from biolmai.pipeline.generative import GenerationStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def make_logits(seq_len: int, preferred: dict, vocab: str = ALPHABET) -> list:
    """Return logits (seq_len × len(vocab)) where each position strongly
    prefers the amino acid in *preferred* and suppresses all others."""
    rows = []
    for pos in range(seq_len):
        row = [-10.0] * len(vocab)
        if pos in preferred:
            aa = preferred[pos]
            if aa in vocab:
                row[vocab.index(aa)] = 10.0
        rows.append(row)
    return rows


@pytest.fixture
def tmp_ds(tmp_path):
    ds = DuckDBDataStore(
        db_path=tmp_path / "test.duckdb",
        data_dir=tmp_path / "data",
    )
    yield ds
    ds.close()


def mock_client_cls(instance):
    """Return a class mock that produces *instance* on every instantiation."""
    cls = MagicMock()
    cls.return_value = instance
    return cls


# ---------------------------------------------------------------------------
# SaturationMutagenesisConfig — specification
# ---------------------------------------------------------------------------


class TestSaturationMutagenesisConfigSpec:
    def test_required_fields(self):
        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKTAY",
            scoring_model="thermompnn-d",
        )
        assert cfg.parent_sequence == "MKTAY"
        assert cfg.scoring_model == "thermompnn-d"

    def test_defaults(self):
        cfg = SaturationMutagenesisConfig(parent_sequence="MKTAY", scoring_model="m")
        assert cfg.alphabet == ALPHABET
        assert cfg.scoring_action == "predict"
        assert cfg.score_field == "ddg"
        assert cfg.top_n == 50
        assert cfg.ascending is True
        assert cfg.exclude_synonymous is True
        assert cfg.batch_size == 8
        assert cfg.label is None
        assert cfg.pdb_str is None
        assert cfg.chain == "A"

    def test_to_spec_roundtrip(self):
        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKTAY",
            scoring_model="thermompnn-d",
            positions=[0, 2, 4],
            top_n=10,
            ascending=False,
            label="thermo-top10",
            pdb_str="ATOM ...",
            chain="B",
        )
        spec = cfg.to_spec()
        assert spec["type"] == "SaturationMutagenesisConfig"
        assert spec["parent_sequence"] == "MKTAY"
        assert spec["positions"] == [0, 2, 4]
        assert spec["top_n"] == 10
        assert spec["ascending"] is False
        assert spec["label"] == "thermo-top10"
        assert spec["pdb_str"] == "ATOM ..."
        assert spec["chain"] == "B"

    def test_to_spec_serializable_as_json(self):
        import json
        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKTAY",
            scoring_model="thermompnn-d",
            positions=[0, 1],
            label="test",
        )
        spec = cfg.to_spec()
        json.dumps(spec)  # must not raise

    def test_generation_stage_to_spec_includes_saturation_config(self):
        cfg = SaturationMutagenesisConfig(parent_sequence="MKTAY", scoring_model="m")
        stage = GenerationStage(name="gen", config=cfg)
        spec = stage.to_spec()
        assert len(spec["configs"]) == 1
        assert spec["configs"][0]["type"] == "SaturationMutagenesisConfig"


# ---------------------------------------------------------------------------
# SaturationMutagenesisConfig — dispatch (mocked API)
# ---------------------------------------------------------------------------


class TestSaturationMutagenesisConfigDispatch:
    """Tests for _run_saturation_mutagenesis with a mocked BioLMApiClient."""

    # parent="MKTAY", positions=[0] → 19 non-synonymous substitutions at pos 0
    PARENT = "MKTAY"
    POSITIONS = [0]

    def _make_mock(self, scores: list, score_field: str = "ddg"):
        """Return a mock that yields exactly len(items) scores per batch call.

        Using side_effect (not return_value) so that each batch call gets the
        correct number of results rather than the same full list every time —
        which would cause zip() truncation to silently mask batch alignment bugs.
        """
        scores_iter = iter(scores)

        async def _predict(items, **kwargs):
            return [{score_field: next(scores_iter, None)} for _ in items]

        instance = AsyncMock()
        instance.predict = _predict
        instance.shutdown = AsyncMock()
        return instance

    def _dispatch(self, cfg, mock_instance, tmp_ds):
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(mock_instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        return ws_out, result

    def test_scores_all_non_synonymous_positions(self, tmp_ds):
        # positions=[0], parent[0]='M', alphabet has 20 AAs → 19 variants
        scores = [-float(i) for i in range(19)]  # -0, -1, ..., -18
        mock_inst = self._make_mock(scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            positions=self.POSITIONS,
            top_n=None,  # keep all
        )
        ws_out, result = self._dispatch(cfg, mock_inst, tmp_ds)
        # All 19 non-synonymous variants should be returned
        assert result.output_count == 19

    def test_top_n_selection(self, tmp_ds):
        scores = list(range(19))  # 0 is lowest (best with ascending=True)
        mock_inst = self._make_mock(scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            positions=self.POSITIONS,
            top_n=3,
            ascending=True,
        )
        ws_out, result = self._dispatch(cfg, mock_inst, tmp_ds)
        assert result.output_count == 3

    def test_top_n_descending_keeps_highest_scores(self, tmp_ds):
        """ascending=False: top_n keeps the HIGHEST scoring variants."""
        # Alphabet order at pos 0 (excl. WT 'M'): A,C,D,E,F,G,H,I,K,L,N,P,Q,R,S,T,V,W,Y
        # Scores 0..18: A=0, C=1, ..., Y=18
        scores = list(range(19))
        mock_inst = self._make_mock(scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            positions=self.POSITIONS,
            top_n=3,
            ascending=False,  # higher score = better
        )
        ws_out, result = self._dispatch(cfg, mock_inst, tmp_ds)
        assert result.output_count == 3
        df = tmp_ds.materialize_working_set(ws_out)
        top_seqs = set(df["sequence"].str.upper())
        # Highest scores: Y(18)→"YKTAY", W(17)→"WKTAY", V(16)→"VKTAY"
        for included in {"YKTAY", "WKTAY", "VKTAY"}:
            assert included in top_seqs, f"{included} should be in top-3 by descending score"
        for excluded in {"AKTAY", "CKTAY", "DKTAY"}:
            assert excluded not in top_seqs, f"{excluded} has low score but appeared in top-3"

    def test_top_n_ascending_keeps_lowest_scores(self, tmp_ds):
        # Alphabet order at pos 0 (excluding WT 'M'): A,C,D,E,F,G,H,I,K,L,N,P,Q,R,S,T,V,W,Y
        # Assign scores 18,17,...,0 in that order → 'Y' gets 0 (lowest/best)
        scores = list(range(18, -1, -1))  # 18 down to 0
        mock_inst = self._make_mock(scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            positions=self.POSITIONS,
            top_n=3,
            ascending=True,
        )
        ws_out, result = self._dispatch(cfg, mock_inst, tmp_ds)
        assert result.output_count == 3
        df = tmp_ds.materialize_working_set(ws_out)
        # Substitutions at pos 0 excluding WT 'M', in alphabet (ACDEFGHIKLMNPQRSTVWY) order:
        # A(18), C(17), D(16), E(15), F(14), G(13), H(12), I(11), K(10), L(9),
        # N(8), P(7), Q(6), R(5), S(4), T(3), V(2), W(1), Y(0)
        # Top-3 lowest scores: Y(0)→"YKTAY", W(1)→"WKTAY", V(2)→"VKTAY"
        expected = {"YKTAY", "WKTAY", "VKTAY"}
        actual = set(df["sequence"].str.upper())
        assert actual == expected, f"Expected {expected}, got {actual}"

    def test_exclude_synonymous_drops_wt_residue(self, tmp_ds):
        # With exclude_synonymous=True (default), 'M' at pos 0 is skipped → 19 variants
        # With exclude_synonymous=False, 'M' is included → 20 variants
        scores_19 = [-float(i) for i in range(19)]
        scores_20 = [-float(i) for i in range(20)]

        mock_19 = self._make_mock(scores_19)
        cfg_excl = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT, scoring_model="m",
            positions=self.POSITIONS, top_n=None, exclude_synonymous=True,
        )
        ws_out_19, result_19 = self._dispatch(cfg_excl, mock_19, tmp_ds)

        # New datastore for second run to avoid sequence dedup interference
        from pathlib import Path
        ds2 = DuckDBDataStore(db_path=Path(tmp_ds.db_path).parent / "ds2.duckdb",
                              data_dir=Path(tmp_ds.db_path).parent / "ds2")
        try:
            mock_20 = self._make_mock(scores_20)
            cfg_incl = SaturationMutagenesisConfig(
                parent_sequence=self.PARENT, scoring_model="m",
                positions=self.POSITIONS, top_n=None, exclude_synonymous=False,
            )
            stage2 = GenerationStage(name="gen", config=cfg_incl)
            with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(mock_20)):
                ws_out_20, result_20 = asyncio.run(
                    stage2.process_ws(WorkingSet(frozenset()), ds2, run_id="r2")
                )
            assert result_19.output_count == 19
            assert result_20.output_count == 20
        finally:
            ds2.close()

    def test_none_scores_are_dropped(self, tmp_ds):
        # 19 variants at pos 0 of "MKT"; every 3rd one gets None score.
        # indices with None: 1, 4, 7, 10, 13, 16 → 6 dropped, 13 kept.
        all_scores = [float(i) if i % 3 != 1 else None for i in range(19)]
        mock_inst = self._make_mock(all_scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKT",
            scoring_model="m",
            positions=[0],
            top_n=None,
            exclude_synonymous=True,
        )
        ws_out, result = self._dispatch(cfg, mock_inst, tmp_ds)
        assert result.output_count == 19 - 6

    def test_label_stored_as_source_label(self, tmp_ds):
        scores = [-float(i) for i in range(19)]
        mock_inst = self._make_mock(scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            positions=self.POSITIONS,
            top_n=3,
            label="thermo-test",
        )
        ws_out, _ = self._dispatch(cfg, mock_inst, tmp_ds)
        df = tmp_ds.materialize_working_set(ws_out)
        assert "source_label" in df.columns
        assert (df["source_label"] == "thermo-test").all()

    def test_pdb_str_builds_item_with_mutations_and_pdb(self, tmp_ds):
        """When pdb_str is set, items must use pdb + mutations list, not sequence."""
        captured_items = []

        async def capture_predict(items, **kwargs):
            captured_items.extend(items)
            return [{"ddg": -1.0} for _ in items]

        instance = AsyncMock()
        instance.predict = capture_predict
        instance.shutdown = AsyncMock()

        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKT",
            scoring_model="thermompnn-d",
            positions=[1],  # pos 1, WT='K'
            pdb_str="ATOM   1  CA  ALA A   1 ...",
            chain="A",
            top_n=None,
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            asyncio.run(stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1"))

        assert len(captured_items) > 0
        for item in captured_items:
            assert "pdb" in item, "items should contain 'pdb' when pdb_str is set"
            assert "mutations" in item, "items should contain 'mutations'"
            assert "sequence" not in item, "items should NOT contain 'sequence'"
            assert item["chain"] == "A"

    def test_nested_score_field(self, tmp_ds):
        """score_field='result.ddg' must navigate nested dict response."""
        scores_iter = iter(-float(i) for i in range(19))

        async def nested_predict(items, **kwargs):
            return [{"result": {"ddg": next(scores_iter, None)}} for _ in items]

        instance = AsyncMock()
        instance.predict = nested_predict
        instance.shutdown = AsyncMock()
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="m",
            positions=self.POSITIONS,
            score_field="result.ddg",
            top_n=5,
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        assert result.output_count == 5

    def test_empty_library_when_all_positions_synonymous(self, tmp_ds):
        """exclude_synonymous=True + single-AA alphabet → 0 variants, 0 API calls."""
        instance = AsyncMock()
        instance.predict = AsyncMock(return_value=[])
        instance.shutdown = AsyncMock()
        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKT",
            scoring_model="m",
            positions=[0],
            alphabet="M",  # only 'M' — same as WT, all excluded
            exclude_synonymous=True,
            top_n=None,
        )
        import warnings
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            with warnings.catch_warnings(record=True):
                ws_out, result = asyncio.run(
                    stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
                )
        instance.predict.assert_not_called()
        assert result.output_count == 0

    def test_positions_none_uses_all_residues(self, tmp_ds):
        """positions=None should enumerate all 3 residues in a 3-residue parent."""
        parent = "MKT"  # 3 residues, 3 × 19 = 57 non-synonymous variants
        instance = AsyncMock()
        instance.predict = AsyncMock(
            side_effect=lambda items, **kw: [{"ddg": -float(i)} for i in range(len(items))]
        )
        instance.shutdown = AsyncMock()
        cfg = SaturationMutagenesisConfig(
            parent_sequence=parent, scoring_model="m",
            positions=None, top_n=None,
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        assert result.output_count == 3 * 19  # all 57 variants

    def test_multi_batch_score_alignment(self, tmp_ds):
        """With batch_size=3 and 19 variants, scores must align to correct mutants."""
        # The first 3 variants at pos 0 (excl. WT 'M') in alphabet order:
        #   A (idx 0), C (idx 1), D (idx 2) → scores 100, 200, 300
        # All others get score 0.
        # With ascending=True and top_n=3, the 3 lowest scores (all 0s) should win —
        # meaning none of the first 3 are in the top set.
        scores = [100.0, 200.0, 300.0] + [0.0] * 16
        mock_inst = self._make_mock(scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            positions=self.POSITIONS,
            top_n=3,
            ascending=True,
            batch_size=3,  # forces 7 batches for 19 variants
        )
        ws_out, result = self._dispatch(cfg, mock_inst, tmp_ds)
        assert result.output_count == 3
        df = tmp_ds.materialize_working_set(ws_out)
        top_seqs = set(df["sequence"].str.upper())
        # High-score variants (first batch: A=100, C=200, D=300) must be excluded
        for excluded in {"AKTAY", "CKTAY", "DKTAY"}:
            assert excluded not in top_seqs, f"{excluded} has score 100+ but appeared in top-3"
        # 16 variants share score 0; stable sort picks the first 3 in enumeration order
        # (alphabet minus 'M'): E(idx 3)→"EKTAY", F(idx 4)→"FKTAY", G(idx 5)→"GKTAY"
        for included in {"EKTAY", "FKTAY", "GKTAY"}:
            assert included in top_seqs, f"{included} has score 0 but was not in top-3"

    def test_api_exception_in_batch_fills_none_scores(self, tmp_ds):
        """A batch-level API failure gracefully fills scores with None for that batch."""
        call_count = 0

        async def failing_predict(items, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated API failure")
            return [{"ddg": -1.0} for _ in items]

        instance = AsyncMock()
        instance.predict = failing_predict
        instance.shutdown = AsyncMock()

        # 19 variants, batch_size=10 → batch 0 fails (None × 10), batch 1 succeeds (9 scores)
        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="m",
            positions=self.POSITIONS,
            top_n=None,
            batch_size=10,
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        # First batch (10 items) → all None → dropped; second batch (9 items) → kept
        assert result.output_count == 9

    def test_frozen_prevents_post_construction_mutation(self):
        """frozen=True must block post-construction attribute assignment."""
        from dataclasses import FrozenInstanceError
        cfg = SaturationMutagenesisConfig(parent_sequence="MKTAY", scoring_model="m")
        with pytest.raises(FrozenInstanceError):
            cfg.positions = [999]

    def test_invalid_scoring_action_raises(self):
        with pytest.raises(ValueError, match="scoring_action"):
            SaturationMutagenesisConfig(
                parent_sequence="MKTAY",
                scoring_model="thermompnn-d",
                scoring_action="__class__",
            )

    def test_invalid_score_field_pattern_raises(self):
        with pytest.raises(ValueError, match="score_field"):
            SaturationMutagenesisConfig(
                parent_sequence="MKTAY",
                scoring_model="m",
                score_field="bad field!",
            )

    def test_reserved_score_field_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            SaturationMutagenesisConfig(
                parent_sequence="MKTAY",
                scoring_model="m",
                score_field="sequence",
            )

    def test_out_of_range_position_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            SaturationMutagenesisConfig(
                parent_sequence="MKTAY",
                scoring_model="m",
                positions=[0, 10],  # 10 >= len("MKTAY")=5
            )

    def test_negative_position_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            SaturationMutagenesisConfig(
                parent_sequence="MKTAY",
                scoring_model="m",
                positions=[-1],
            )


# ---------------------------------------------------------------------------
# IterativeMaskingDMSConfig — specification
# ---------------------------------------------------------------------------


class TestIterativeMaskingDMSConfigSpec:
    def test_required_fields(self):
        cfg = IterativeMaskingDMSConfig(
            parent_sequence="MKTAY",
            model_name="esm2-650m",
        )
        assert cfg.parent_sequence == "MKTAY"
        assert cfg.model_name == "esm2-650m"

    def test_defaults(self):
        cfg = IterativeMaskingDMSConfig(parent_sequence="MKTAY", model_name="m")
        assert cfg.positions is None
        assert cfg.rounds == 2
        assert cfg.mask_token == "<mask>"
        assert cfg.alphabet == ALPHABET
        assert cfg.exclude_synonymous is True
        assert cfg.batch_size == 32
        assert cfg.label is None
        assert cfg.action == "predict"

    def test_to_spec_roundtrip(self):
        cfg = IterativeMaskingDMSConfig(
            parent_sequence="MKTAY",
            model_name="esm2-650m",
            positions=[0, 2, 4],
            rounds=1,
            label="dms-r1",
        )
        spec = cfg.to_spec()
        assert spec["type"] == "IterativeMaskingDMSConfig"
        assert spec["rounds"] == 1
        assert spec["positions"] == [0, 2, 4]
        assert spec["label"] == "dms-r1"

    def test_to_spec_serializable_as_json(self):
        import json
        cfg = IterativeMaskingDMSConfig(parent_sequence="MKTAY", model_name="m")
        json.dumps(cfg.to_spec())  # must not raise

    def test_generation_stage_to_spec_includes_dms_config(self):
        cfg = IterativeMaskingDMSConfig(parent_sequence="MKTAY", model_name="m")
        stage = GenerationStage(name="gen", config=cfg)
        spec = stage.to_spec()
        assert spec["configs"][0]["type"] == "IterativeMaskingDMSConfig"

    def test_frozen_prevents_post_construction_mutation(self):
        """frozen=True must block post-construction attribute assignment."""
        from dataclasses import FrozenInstanceError
        cfg = IterativeMaskingDMSConfig(parent_sequence="MKTAY", model_name="esm2-650m")
        with pytest.raises(FrozenInstanceError):
            cfg.positions = [999]

    def test_rounds_greater_than_2_raises(self):
        with pytest.raises(ValueError, match="rounds > 2"):
            IterativeMaskingDMSConfig(
                parent_sequence="MKTAY",
                model_name="esm2-650m",
                rounds=3,
            )

    def test_rounds_0_raises(self):
        with pytest.raises(ValueError, match="rounds must be >= 1"):
            IterativeMaskingDMSConfig(
                parent_sequence="MKTAY",
                model_name="esm2-650m",
                rounds=0,
            )

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            IterativeMaskingDMSConfig(
                parent_sequence="MKTAY",
                model_name="esm2-650m",
                action="_headers",
            )

    def test_out_of_range_position_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            IterativeMaskingDMSConfig(
                parent_sequence="MKTAY",
                model_name="esm2-650m",
                positions=[0, 99],
            )


# ---------------------------------------------------------------------------
# IterativeMaskingDMSConfig — dispatch (mocked API)
# ---------------------------------------------------------------------------


class TestIterativeMaskingDMSConfigDispatch:
    PARENT = "MKTAY"
    POSITIONS = [0, 2]  # 2 target positions → 2 round-1 variants, 2 round-2 variants

    def _make_logits_response(self, preferred_by_position: dict) -> dict:
        """Build a model response dict with logits encoding the preferred AA."""
        return {
            "logits": make_logits(len(self.PARENT), preferred_by_position),
            "vocab_tokens": list(ALPHABET),
        }

    def _dispatch(self, cfg, mock_instance, tmp_ds):
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(mock_instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        return ws_out, result

    def test_rounds_1_produces_single_point_variants(self, tmp_ds):
        """rounds=1: each target position yields one single-mutant sequence."""
        # pos 0 (WT='M') → prefers 'A'; pos 2 (WT='T') → prefers 'G'
        r1_response = [
            self._make_logits_response({0: "A"}),
            self._make_logits_response({2: "G"}),
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(return_value=r1_response)
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esm2-650m",
            positions=self.POSITIONS,
            rounds=1,
            batch_size=100,  # explicit: ensures both positions collapse to one call
        )
        ws_out, result = self._dispatch(cfg, instance, tmp_ds)
        # 2 positions → 2 single-mutant sequences
        assert result.output_count == 2
        df = tmp_ds.materialize_working_set(ws_out)
        seqs = set(df["sequence"])
        assert "AKTAY" in seqs  # M→A at pos 0
        assert "MKGAY" in seqs  # T→G at pos 2

    def test_rounds_2_produces_two_point_variants(self, tmp_ds):
        """rounds=2: each round-1 variant yields (n_positions-1) two-point variants."""
        # Round 1: pos0→'A', pos2→'G'
        r1_response = [
            self._make_logits_response({0: "A"}),
            self._make_logits_response({2: "G"}),
        ]
        # Round 2:
        #   AKTAY: mask pos2 → prefers 'D' at pos2 → "AKDAY"
        #   MKGAY: mask pos0 → prefers 'L' at pos0 → "LKGAY"
        r2_response = [
            self._make_logits_response({2: "D"}),
            self._make_logits_response({0: "L"}),
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(side_effect=[r1_response, r2_response])
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esm2-650m",
            positions=self.POSITIONS,
            rounds=2,
            batch_size=100,  # explicit: 2 positions → both rounds each fit in 1 call
        )
        ws_out, result = self._dispatch(cfg, instance, tmp_ds)
        assert result.output_count == 2
        df = tmp_ds.materialize_working_set(ws_out)
        seqs = set(df["sequence"])
        assert "AKDAY" in seqs
        assert "LKGAY" in seqs

    def test_exclude_synonymous_drops_wt_argmax_positions(self, tmp_ds):
        """If round-1 argmax == WT residue, that position is skipped."""
        # pos 0 WT='M' → argmax returns 'M' (synonymous, excluded)
        # pos 2 WT='T' → argmax returns 'G' (change, kept)
        r1_response = [
            self._make_logits_response({0: "M"}),  # synonymous → excluded
            self._make_logits_response({2: "G"}),
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(return_value=r1_response)
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esm2-650m",
            positions=self.POSITIONS,
            rounds=1,
            exclude_synonymous=True,
            batch_size=100,
        )
        ws_out, result = self._dispatch(cfg, instance, tmp_ds)
        # Only pos2 change is kept
        assert result.output_count == 1
        df = tmp_ds.materialize_working_set(ws_out)
        assert "MKGAY" in set(df["sequence"])
        assert "MKTAY" not in set(df["sequence"])  # synonymous dropped

    def test_exclude_synonymous_false_keeps_wt_argmax(self, tmp_ds):
        """exclude_synonymous=False keeps positions even if argmax == WT."""
        r1_response = [
            self._make_logits_response({0: "M"}),  # synonymous but kept
            self._make_logits_response({2: "G"}),
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(return_value=r1_response)
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esm2-650m",
            positions=self.POSITIONS,
            rounds=1,
            exclude_synonymous=False,
            batch_size=100,
        )
        ws_out, result = self._dispatch(cfg, instance, tmp_ds)
        # Both positions kept (even the synonymous one)
        assert result.output_count == 2

    def test_label_stored_as_source_label(self, tmp_ds):
        r1_response = [
            self._make_logits_response({0: "A"}),
            self._make_logits_response({2: "G"}),
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(return_value=r1_response)
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esm2-650m",
            positions=self.POSITIONS,
            rounds=1,
            label="dms-esm2-r1",
            batch_size=100,
        )
        ws_out, _ = self._dispatch(cfg, instance, tmp_ds)
        df = tmp_ds.materialize_working_set(ws_out)
        assert "source_label" in df.columns
        assert (df["source_label"] == "dms-esm2-r1").all()

    def test_bos_eos_logits_strip(self, tmp_ds):
        """Logits with shape (seq_len + 2, vocab) should have BOS/EOS stripped."""
        seq_len = len(self.PARENT)  # 5
        vocab = list(ALPHABET)

        # Build logits with BOS/EOS padding (shape 7 × 20)
        def make_padded_logits(preferred: dict) -> list:
            rows = [[-10.0] * len(vocab)]  # BOS
            rows.extend(make_logits(seq_len, preferred, "".join(vocab)))
            rows.append([-10.0] * len(vocab))  # EOS
            return rows

        r1_response = [
            {"logits": make_padded_logits({0: "A"}), "vocab_tokens": vocab},
            {"logits": make_padded_logits({2: "G"}), "vocab_tokens": vocab},
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(return_value=r1_response)
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esmc-300m",
            positions=self.POSITIONS,
            rounds=1,
            batch_size=100,
        )
        ws_out, result = self._dispatch(cfg, instance, tmp_ds)
        # Should correctly extract argmax even with BOS/EOS padding
        assert result.output_count == 2
        df = tmp_ds.materialize_working_set(ws_out)
        seqs = set(df["sequence"])
        assert "AKTAY" in seqs
        assert "MKGAY" in seqs

    def test_deduplication_of_identical_two_point_variants(self, tmp_ds):
        """If two (pos1, pos2) pairs produce the same final sequence, it appears once."""
        # Use positions [0, 1] where the round-2 cross-products happen to collide.
        # pos0→'A' applied, then pos1→'K' (unchanged from WT): "AKTAY" + pos1="K" → "AKTAY"
        # pos1→'K' applied (synonymous if WT is also 'K'? let's use a different parent)
        # Easier: make both round-2 variants produce the same sequence
        parent = "MKTAY"
        positions = [0, 2]

        # Round 1: pos0→'A', pos2→'G'
        r1_response = [
            self._make_logits_response({0: "A"}),
            self._make_logits_response({2: "G"}),
        ]
        # Round 2: both produce "AKGAY" (contrived but valid for testing dedup)
        r2_response = [
            {"logits": make_logits(len(parent), {2: "G"}), "vocab_tokens": list(ALPHABET)},
            {"logits": make_logits(len(parent), {0: "A"}), "vocab_tokens": list(ALPHABET)},
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(side_effect=[r1_response, r2_response])
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=parent, model_name="esm2-650m",
            positions=positions, rounds=2, batch_size=100,
        )
        ws_out, result = self._dispatch(cfg, instance, tmp_ds)
        df = tmp_ds.materialize_working_set(ws_out)
        # Both round-2 branches produce "AKGAY" → deduplicated to 1
        assert len(df) == 1
        assert df["sequence"].iloc[0] == "AKGAY"

    def test_positions_none_uses_all_sequence_residues(self, tmp_ds):
        """positions=None enumerates all residues in the parent."""
        parent = "MKT"  # 3 residues
        # Round 1 with 3 positions, return a change at each
        r1_response = [
            {"logits": make_logits(3, {0: "A"}), "vocab_tokens": list(ALPHABET)},
            {"logits": make_logits(3, {1: "L"}), "vocab_tokens": list(ALPHABET)},
            {"logits": make_logits(3, {2: "G"}), "vocab_tokens": list(ALPHABET)},
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(return_value=r1_response)
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=parent, model_name="esm2-650m",
            positions=None, rounds=1, batch_size=100,
        )
        stage = GenerationStage(name="gen", config=cfg)
        ds_path = tmp_ds.db_path  # reuse fixture ds
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        # 3 positions → 3 single-mutant sequences
        assert result.output_count == 3

    def test_short_api_response_round1_padded_not_truncated(self, tmp_ds):
        """Round-1: if an earlier batch returns N-1 items, the gap is padded with {}
        so subsequent batches stay position-aligned.

        Setup: 4 positions, batch_size=3 → two batches.
          Batch 1 (pos 0,1,2): API returns 2 results — pos 2 missing.
          Batch 2 (pos 3):     API returns 1 result — pos 3 = 'V'.

        With proper padding batch 1 pads to [r0, r1, {}], so results_r1 has
        4 entries aligned to positions [0,1,2,3].  pos 3 → 'V' → "MKTVY".

        With zip-truncation results_r1 = [r0, r1, r3] (only 3 entries).
        zip([0,1,2,3], [r0,r1,r3]) produces 3 pairs: pos 2 gets r3's logits
        (wrong argmax at pos 2) and pos 3 is silently dropped → "MKTVY" absent.
        """
        parent = "MKTAY"
        positions = [0, 1, 2, 3]
        vocab = list(ALPHABET)
        batch1_resp = [
            {"logits": make_logits(5, {0: "A"}), "vocab_tokens": vocab},  # pos0→A
            {"logits": make_logits(5, {1: "L"}), "vocab_tokens": vocab},  # pos1→L
            # pos2 missing from this batch
        ]
        batch2_resp = [
            {"logits": make_logits(5, {3: "V"}), "vocab_tokens": vocab},  # pos3→V
        ]
        call_count = 0

        async def side_effect(items, **kwargs):
            nonlocal call_count
            call_count += 1
            return batch1_resp if call_count == 1 else batch2_resp

        instance = AsyncMock()
        instance.predict = side_effect
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=parent, model_name="esm2-650m",
            positions=positions, rounds=1,
            batch_size=3,  # forces two batches
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        df = tmp_ds.materialize_working_set(ws_out)
        sequences = set(df["sequence"])
        # pos3→V must be present: only reachable if alignment was maintained after the gap
        assert "MKTVY" in sequences, (
            "pos3 dropped — zip-truncation regression: batch gap shifted alignment"
        )
        # pos2 must NOT have been mis-processed using r3's logits (argmax at pos2 from
        # make_logits(5,{3:'V'}) is 'A' since row2 is all -10.0 → first AA in alphabet)
        assert not any(s[2] == "A" and s != "AKTAY" for s in sequences), (
            "pos2 got wrong argmax from r3 — zip-truncation misalignment"
        )

    def test_short_api_response_round2_padded_not_truncated(self, tmp_ds):
        """Round-2: if the first round-2 batch returns empty, the gap is padded
        so the second batch's item stays aligned to its r2_items entry.

        Setup: parent="MKT", positions=[0,1], batch_size=1 → one API call per r2 item.
          r2_items: (pos1=0,pos2=1,'A') and (pos1=1,pos2=0,'L').
          Round-2 batch 0: API returns [] (empty) — item 0 result missing.
          Round-2 batch 1: API returns logits with pos0→'G'.

        With proper padding r2_results=[{}, r]
          → item0 → {} → argmax None → skip
          → item1 → r → argmax at pos0='G' → seq "GLT"

        With zip-truncation r2_results=[r] (1 entry).
        zip(r2_items, [r]) produces 1 pair: item0 gets item1's logits.
          argmax at pos1 from make_logits(3,{0:'G'}) row1 = 'A' (all-10.0 tie → first AA).
          seq_out: pos0='A', pos1='A' → "AAT". Item1 dropped.
        """
        parent = "MKT"
        positions = [0, 1]
        vocab = list(ALPHABET)
        # batch_size=1 → 4 total API calls:
        #   call 1: round-1 batch 0 (pos 0)    → pos0→'A'
        #   call 2: round-1 batch 1 (pos 1)    → pos1→'L'
        #   call 3: round-2 batch 0 (item 0)   → EMPTY (the gap we're testing)
        #   call 4: round-2 batch 1 (item 1)   → pos0→'G'
        r1_batch0 = [{"logits": make_logits(3, {0: "A"}), "vocab_tokens": vocab}]
        r1_batch1 = [{"logits": make_logits(3, {1: "L"}), "vocab_tokens": vocab}]
        r2_item1_resp = [{"logits": make_logits(3, {0: "G"}), "vocab_tokens": vocab}]
        call_count = 0

        async def side_effect(items, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return r1_batch0    # round-1, pos0
            elif call_count == 2:
                return r1_batch1    # round-1, pos1
            elif call_count == 3:
                return []           # round-2 batch 0: empty response
            else:
                return r2_item1_resp  # round-2 batch 1

        instance = AsyncMock()
        instance.predict = side_effect
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=parent, model_name="esm2-650m",
            positions=positions, rounds=2,
            batch_size=1,  # each r2 item gets its own API call
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        df = tmp_ds.materialize_working_set(ws_out)
        sequences = set(df["sequence"])
        # item1 correctly processed → pos1=1,aa1='L',pos2=0,aa2='G' → "GLT"
        assert "GLT" in sequences, (
            "item1 dropped — zip-truncation regression: empty batch gap shifted r2 alignment"
        )
        # item0 must NOT have been mis-processed using item1's logits
        assert "AAT" not in sequences, (
            "item0 got item1's logits — zip-truncation misalignment in round-2"
        )


# ---------------------------------------------------------------------------
# Preflight validation — wrong API response fields caught immediately
# ---------------------------------------------------------------------------


class TestPreflightValidation:
    """Verify that misconfigured field names raise immediately on the first batch
    rather than silently producing empty or wrong results after a full run."""

    PARENT = "MKTAY"

    def _dispatch_sat(self, cfg, mock_instance, tmp_ds):
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(mock_instance)):
            return asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )

    def _dispatch_dms(self, cfg, mock_instance, tmp_ds):
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(mock_instance)):
            return asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )

    def test_wrong_score_field_raises_on_first_batch(self, tmp_ds):
        """If score_field is misspelled, a ValueError naming the actual response keys
        must be raised on the first batch — not silently after all batches return None."""
        instance = AsyncMock()
        # API returns 'ddg' but config asks for 'wrong_field'
        instance.score = AsyncMock(
            side_effect=lambda items, **kw: [{"ddg": -1.0} for _ in items]
        )
        instance.shutdown = AsyncMock()

        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            scoring_action="score",
            score_field="wrong_field",
            positions=[0],
        )
        with pytest.raises(ValueError, match="wrong_field"):
            self._dispatch_sat(cfg, instance, tmp_ds)

    def test_wrong_score_field_error_includes_available_keys(self, tmp_ds):
        """The ValueError message must name the keys actually present in the response."""
        instance = AsyncMock()
        instance.score = AsyncMock(
            side_effect=lambda items, **kw: [{"ddg": -1.0, "plddt": 90.0} for _ in items]
        )
        instance.shutdown = AsyncMock()

        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            scoring_action="score",
            score_field="typo_score",
            positions=[0],
        )
        with pytest.raises(ValueError, match="ddg"):
            self._dispatch_sat(cfg, instance, tmp_ds)

    def test_missing_vocab_tokens_raises_on_first_batch(self, tmp_ds):
        """IterativeMaskingDMS must raise immediately if vocab_tokens absent from response."""
        instance = AsyncMock()
        # Return logits WITHOUT vocab_tokens
        instance.predict = AsyncMock(
            return_value=[{"logits": [[0.1] * 20] * len(self.PARENT)}]
        )
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esm2-650m",
            positions=[0],
            rounds=1,
        )
        with pytest.raises(ValueError, match="vocab_tokens"):
            self._dispatch_dms(cfg, instance, tmp_ds)

    def test_valid_score_field_passes_preflight_and_runs(self, tmp_ds):
        """Happy-path: correct score_field passes preflight and produces output."""
        instance = AsyncMock()
        instance.score = AsyncMock(
            side_effect=lambda items, **kw: [{"ddg": -1.0} for _ in items]
        )
        instance.shutdown = AsyncMock()

        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            scoring_action="score",
            score_field="ddg",
            positions=[0],
        )
        ws_out, result = self._dispatch_sat(cfg, instance, tmp_ds)
        assert result.output_count > 0, "correct score_field should produce variants"

    def test_valid_vocab_tokens_passes_preflight_and_runs(self, tmp_ds):
        """Happy-path: correct vocab_tokens passes preflight and produces output."""
        from tests.test_design_primitives import ALPHABET, make_logits
        instance = AsyncMock()
        instance.predict = AsyncMock(
            return_value=[{
                "logits": make_logits(len(self.PARENT), {0: "A"}),
                "vocab_tokens": list(ALPHABET),
            }]
        )
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence=self.PARENT,
            model_name="esm2-650m",
            positions=[0],
            rounds=1,
        )
        ws_out, result = self._dispatch_dms(cfg, instance, tmp_ds)
        assert result.output_count > 0, "correct vocab_tokens should produce output"


# ---------------------------------------------------------------------------
# GenerativePipeline constructor shorthands
# ---------------------------------------------------------------------------


class TestGenerativePipelineShorthands:
    """configs=, filters=, data_store= aliases on GenerativePipeline.__init__."""

    def test_configs_alias_resolves_generation_configs(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        cfg = SequenceSourceConfig(sequences=["MKTAY"])
        pipe = GenerativePipeline(configs=[cfg], datastore=ds, verbose=False)
        assert len(pipe.generation_configs) == 1
        assert pipe.generation_configs[0] is cfg
        ds.close()

    def test_generation_configs_kwarg_still_works(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        cfg = SequenceSourceConfig(sequences=["MKTAY"])
        pipe = GenerativePipeline(generation_configs=[cfg], datastore=ds, verbose=False)
        assert len(pipe.generation_configs) == 1
        ds.close()

    def test_filters_adds_filter_stage(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        flt = ValidAminoAcidFilter()
        pipe = GenerativePipeline(
            configs=[SequenceSourceConfig(sequences=["MKTAY"])],
            filters=flt,
            datastore=ds,
            verbose=False,
        )
        from biolmai.pipeline.data import FilterStage
        filter_stages = [s for s in pipe.stages if isinstance(s, FilterStage)]
        assert len(filter_stages) == 1
        ds.close()

    def test_filters_combine_filters_object(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        combined = combine_filters(ValidAminoAcidFilter(), HammingDistanceFilter("MKTAY", min_distance=1))
        pipe = GenerativePipeline(
            configs=[SequenceSourceConfig(sequences=["AKTAY"])],
            filters=combined,
            datastore=ds,
            verbose=False,
        )
        from biolmai.pipeline.data import FilterStage
        assert any(isinstance(s, FilterStage) for s in pipe.stages)
        ds.close()

    def test_data_store_alias_sets_datastore(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        pipe = GenerativePipeline(
            configs=[SequenceSourceConfig(sequences=["MKTAY"])],
            data_store=ds,
            verbose=False,
        )
        assert pipe.datastore is ds
        ds.close()

    def test_data_store_and_datastore_conflict_uses_datastore(self, tmp_path):
        """When both data_store= and datastore= are supplied, datastore= wins
        (data_store= is only applied when 'datastore' is absent from kwargs)."""
        ds1 = DuckDBDataStore(db_path=tmp_path / "t1.duckdb", data_dir=tmp_path / "d1")
        ds2 = DuckDBDataStore(db_path=tmp_path / "t2.duckdb", data_dir=tmp_path / "d2")
        # datastore= kwarg takes precedence (data_store= is only a fallback)
        pipe = GenerativePipeline(
            configs=[SequenceSourceConfig(sequences=["MKTAY"])],
            datastore=ds1,
            data_store=ds2,
            verbose=False,
        )
        assert pipe.datastore is ds1
        ds1.close()
        ds2.close()

    def test_configs_and_filters_and_data_store_together(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        cfg = SequenceSourceConfig(sequences=["MKTAY", "AKTAY"])
        flt = ValidAminoAcidFilter()
        pipe = GenerativePipeline(configs=[cfg], filters=flt, data_store=ds, verbose=False)
        pipe.run()
        df = pipe.results()
        assert len(df) == 2
        ds.close()

    def test_no_configs_produces_empty_pipeline(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        pipe = GenerativePipeline(datastore=ds, verbose=False)
        assert len(pipe.generation_configs) == 0
        ds.close()


# ---------------------------------------------------------------------------
# DirectGenerationConfig.label → generation_metadata → source_label in results()
# ---------------------------------------------------------------------------


class TestDirectGenerationConfigLabel:
    def test_label_field_default_none(self):
        cfg = DirectGenerationConfig("hyper-mpnn")
        assert cfg.label is None

    def test_label_stored_in_to_spec(self):
        cfg = DirectGenerationConfig("hyper-mpnn", label="patch-nterm-T03")
        spec = cfg.to_spec()
        assert spec["label"] == "patch-nterm-T03"

    def test_label_none_in_to_spec(self):
        cfg = DirectGenerationConfig("hyper-mpnn")
        spec = cfg.to_spec()
        assert spec["label"] is None

    def test_label_persisted_in_generation_metadata(self, tmp_ds):
        """Rows stored via add_generation_metadata_batch carry the label through."""
        seq_id = tmp_ds.add_sequences_batch(["MKTAYIAKQRQ"])[0]
        tmp_ds.create_pipeline_run(run_id="r1", pipeline_type="T", config={}, status="running")
        tmp_ds.add_generation_metadata_batch([{
            "sequence_id": seq_id,
            "run_id": "r1",
            "model_name": "hyper-mpnn",
            "temperature": 0.3,
            "label": "nterm-T0.3",
        }])
        row = tmp_ds.conn.execute(
            "SELECT label FROM generation_metadata WHERE sequence_id = ?", [seq_id]
        ).fetchone()
        assert row is not None
        assert row[0] == "nterm-T0.3"

    def test_source_label_appears_in_materialize_when_label_set(self, tmp_ds):
        """materialize_working_set includes source_label when labels exist in generation_metadata."""
        seq_id = tmp_ds.add_sequences_batch(["MKTAYIAKQRQ"])[0]
        tmp_ds.create_pipeline_run(run_id="r1", pipeline_type="T", config={}, status="running")
        tmp_ds.add_generation_metadata_batch([{
            "sequence_id": seq_id,
            "run_id": "r1",
            "model_name": "hyper-mpnn",
            "label": "test-label",
        }])
        ws = WorkingSet.from_ids([seq_id])
        df = tmp_ds.materialize_working_set(ws)
        assert "source_label" in df.columns
        assert df["source_label"].iloc[0] == "test-label"

    def test_source_label_is_null_when_no_labels_stored(self, tmp_ds):
        """source_label column is always present; NULL when no label was set."""
        seq_id = tmp_ds.add_sequences_batch(["MKTAYIAKQRQ"])[0]
        tmp_ds.create_pipeline_run(run_id="r1", pipeline_type="T", config={}, status="running")
        tmp_ds.add_generation_metadata_batch([{
            "sequence_id": seq_id,
            "run_id": "r1",
            "model_name": "hyper-mpnn",
            "label": None,
        }])
        ws = WorkingSet.from_ids([seq_id])
        df = tmp_ds.materialize_working_set(ws)
        assert "source_label" in df.columns
        assert df["source_label"].isna().all()

    def test_multiple_labels_preserved_per_sequence(self, tmp_ds):
        """Each sequence retains its own label; different labels coexist."""
        ids = tmp_ds.add_sequences_batch(["MKTAY", "AKTAY", "GKTAY"])
        tmp_ds.create_pipeline_run(run_id="r1", pipeline_type="T", config={}, status="running")
        for sid, label in zip(ids, ["label-A", "label-B", "label-C"]):
            tmp_ds.add_generation_metadata_batch([{
                "sequence_id": sid,
                "run_id": "r1",
                "model_name": "hyper-mpnn",
                "label": label,
            }])
        ws = WorkingSet.from_ids(ids)
        df = tmp_ds.materialize_working_set(ws)
        assert "source_label" in df.columns
        assert set(df["source_label"]) == {"label-A", "label-B", "label-C"}

    def test_source_label_in_pipeline_results_end_to_end(self, tmp_path):
        """End-to-end: GenerativePipeline with labelled DirectGenerationConfig
        produces results() with source_label column."""
        ds = DuckDBDataStore(db_path=tmp_path / "e2e.duckdb", data_dir=tmp_path / "d")

        mock_instance = AsyncMock()
        # Return MPNN-style flat list (one sequence per call)
        mock_instance.generate = AsyncMock(
            return_value=[{"sequence": "AKTAY"}]
        )
        mock_instance.shutdown = AsyncMock()

        cfg = DirectGenerationConfig(
            model_name="hyper-mpnn",
            item_field="sequence",
            sequence="MKTAY",
            params={"batch_size": 1, "temperature": 0.3},
            label="hyper-nterm-T0.3",
        )
        pipe = GenerativePipeline(configs=[cfg], data_store=ds, verbose=False)

        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(mock_instance)):
            pipe.run()

        df = pipe.results()
        ds.close()

        assert "source_label" in df.columns
        assert df["source_label"].iloc[0] == "hyper-nterm-T0.3"


# ---------------------------------------------------------------------------
# BasePipeline.results() alias
# ---------------------------------------------------------------------------


class TestResultsAlias:
    def test_results_returns_same_as_get_final_data(self, tmp_path):
        from biolmai.pipeline.data import DataPipeline
        from biolmai.pipeline.filters import SequenceLengthFilter

        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        pipe = DataPipeline(
            sequences=["MKTAY", "AKTAY", "GKTAY"],
            datastore=ds,
            output_dir=tmp_path,
            verbose=False,
        )
        pipe.add_filter(SequenceLengthFilter(min_length=1))  # pass-all filter to give the pipeline a stage
        pipe.run()

        df_results = pipe.results()
        df_gfd = pipe.get_final_data()

        assert list(df_results.columns) == list(df_gfd.columns)
        assert len(df_results) == len(df_gfd)
        assert set(df_results["sequence"]) == set(df_gfd["sequence"])
        ds.close()

    def test_results_raises_before_run(self, tmp_path):
        from biolmai.pipeline.data import DataPipeline

        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        pipe = DataPipeline(
            sequences=["MKTAY"],
            datastore=ds,
            output_dir=tmp_path,
            verbose=False,
        )
        with pytest.raises(RuntimeError):
            pipe.results()
        ds.close()

    def test_results_alias_on_generative_pipeline(self, tmp_path):
        ds = DuckDBDataStore(db_path=tmp_path / "t.duckdb", data_dir=tmp_path / "d")
        pipe = GenerativePipeline(
            configs=[SequenceSourceConfig(sequences=["MKTAY", "AKTAY"])],
            data_store=ds,
            verbose=False,
        )
        pipe.run()
        df = pipe.results()
        assert len(df) == 2
        ds.close()


# ---------------------------------------------------------------------------
# pipeline_def.py round-trip: save → reconstruct via _config_from_spec()
# ---------------------------------------------------------------------------


class TestPipelineDefRoundtrip:
    """Verify that all new config types survive to_spec() → _config_from_spec()."""

    def test_saturation_mutagenesis_config_roundtrip(self):
        from biolmai.pipeline.pipeline_def import _config_from_spec

        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKTAY",
            scoring_model="thermompnn-d",
            positions=[0, 2],
            top_n=10,
            ascending=False,
            label="thermo-label",
            pdb_str="ATOM ...",
            chain="B",
            score_field="ddg",
        )
        restored = _config_from_spec(cfg.to_spec())
        assert isinstance(restored, SaturationMutagenesisConfig)
        assert restored.parent_sequence == cfg.parent_sequence
        assert restored.scoring_model == cfg.scoring_model
        assert restored.positions == cfg.positions
        assert restored.top_n == cfg.top_n
        assert restored.ascending == cfg.ascending
        assert restored.label == cfg.label
        assert restored.pdb_str == cfg.pdb_str
        assert restored.chain == cfg.chain

    def test_iterative_masking_dms_config_roundtrip(self):
        from biolmai.pipeline.pipeline_def import _config_from_spec

        cfg = IterativeMaskingDMSConfig(
            parent_sequence="MKTAY",
            model_name="esm2-650m",
            positions=[0, 2],
            rounds=2,
            label="dms-label",
            batch_size=64,
        )
        restored = _config_from_spec(cfg.to_spec())
        assert isinstance(restored, IterativeMaskingDMSConfig)
        assert restored.parent_sequence == cfg.parent_sequence
        assert restored.model_name == cfg.model_name
        assert restored.rounds == cfg.rounds
        assert restored.label == cfg.label
        assert restored.batch_size == cfg.batch_size

    def test_direct_generation_config_label_survives_roundtrip(self):
        from biolmai.pipeline.pipeline_def import _config_from_spec
        from biolmai.pipeline import DirectGenerationConfig

        cfg = DirectGenerationConfig(
            model_name="hyper-mpnn",
            item_field="pdb",
            params={"batch_size": 50, "temperature": 0.3},
            label="hyper-nterm-T0.3",
        )
        restored = _config_from_spec(cfg.to_spec())
        assert isinstance(restored, DirectGenerationConfig)
        assert restored.label == "hyper-nterm-T0.3"

    def test_direct_generation_config_label_none_survives_roundtrip(self):
        from biolmai.pipeline.pipeline_def import _config_from_spec
        from biolmai.pipeline import DirectGenerationConfig

        cfg = DirectGenerationConfig(model_name="hyper-mpnn")
        restored = _config_from_spec(cfg.to_spec())
        assert restored.label is None

    def test_saturation_mutagenesis_all_nondefault_fields_roundtrip(self):
        """Every non-default field must survive to_spec() → _config_from_spec()."""
        from biolmai.pipeline.pipeline_def import _config_from_spec

        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPFANSIHSTATAMQKFLALSGEENVQLYNIGGGVLALSNRHLRSQAGLNLALIQREAEIRDLAEVSYQSRLDRLDALRVKLRRQIVDWENAQIKRGLNAFGLALGGAGLSAVGLSMSGSGSGM",
            scoring_model="thermompnn-d",
            positions=[1, 5, 10],
            alphabet="ACDEFGHIKLM",
            scoring_action="score",
            scoring_params={"chain": "A", "extra_flag": True},
            score_field="result.ddg",
            top_n=25,
            ascending=False,
            exclude_synonymous=False,
            batch_size=16,
            label="thermo-nondefault",
            pdb_str="ATOM  1  CA  ALA A   1       1.0   2.0   3.0  1.00  0.00",
            chain="B",
        )
        restored = _config_from_spec(cfg.to_spec())
        assert isinstance(restored, SaturationMutagenesisConfig)
        assert restored.alphabet == cfg.alphabet
        assert restored.scoring_action == cfg.scoring_action
        assert restored.scoring_params == cfg.scoring_params
        assert restored.score_field == cfg.score_field
        assert restored.top_n == cfg.top_n
        assert restored.ascending == cfg.ascending
        assert restored.exclude_synonymous == cfg.exclude_synonymous
        assert restored.batch_size == cfg.batch_size
        assert restored.pdb_str == cfg.pdb_str
        assert restored.chain == cfg.chain
        assert restored.label == cfg.label

    def test_iterative_masking_dms_all_nondefault_fields_roundtrip(self):
        """Every non-default field must survive to_spec() → _config_from_spec()."""
        from biolmai.pipeline.pipeline_def import _config_from_spec

        cfg = IterativeMaskingDMSConfig(
            parent_sequence="MKTAYIAKQRQ",
            model_name="esmc-300m",
            positions=[0, 3, 7],
            rounds=1,
            mask_token="[MASK]",
            alphabet="ACDEFGHIKLM",
            exclude_synonymous=False,
            batch_size=64,
            label="dms-nondefault",
            action="predict",
        )
        restored = _config_from_spec(cfg.to_spec())
        assert isinstance(restored, IterativeMaskingDMSConfig)
        assert restored.rounds == cfg.rounds
        assert restored.mask_token == cfg.mask_token
        assert restored.alphabet == cfg.alphabet
        assert restored.exclude_synonymous == cfg.exclude_synonymous
        assert restored.batch_size == cfg.batch_size
        assert restored.label == cfg.label
        assert restored.action == cfg.action

    def test_direct_generation_config_all_nondefault_fields_roundtrip(self):
        """Every non-default field must survive to_spec() → _config_from_spec()."""
        from biolmai.pipeline.pipeline_def import _config_from_spec
        from biolmai.pipeline import DirectGenerationConfig

        cfg = DirectGenerationConfig(
            model_name="dsm-150m-base",
            structure_path="/tmp/test.pdb",
            structure_column="pdb_col",
            sequence="MKTAY",
            item_field="sequence",
            params={"num_sequences": 50, "temperature": 0.5, "remasking": "random"},
            num_sequences=50,
            temperature=0.5,
            structure_from_stage="fold_stage",
            structure_from_model="esmfold",
            n_runs=3,
            label="dsm-nondefault",
        )
        restored = _config_from_spec(cfg.to_spec())
        assert isinstance(restored, DirectGenerationConfig)
        assert restored.structure_path == cfg.structure_path
        assert restored.structure_column == cfg.structure_column
        assert restored.sequence == cfg.sequence
        assert restored.item_field == cfg.item_field
        assert restored.params == cfg.params
        assert restored.num_sequences == cfg.num_sequences
        assert restored.temperature == cfg.temperature
        assert restored.structure_from_stage == cfg.structure_from_stage
        assert restored.structure_from_model == cfg.structure_from_model
        assert restored.n_runs == cfg.n_runs
        assert restored.label == cfg.label

    def test_unknown_config_type_raises(self):
        from biolmai.pipeline.pipeline_def import _config_from_spec

        with pytest.raises(ValueError, match="Unknown generation config type"):
            _config_from_spec({"type": "NonExistentConfig"})


class TestBaseClassHierarchy:
    """Verify ScoringProtocolConfig / GenerativeProtocolConfig marker inheritance."""

    def test_saturation_mutagenesis_is_scoring(self):
        from biolmai.pipeline import SaturationMutagenesisConfig, ScoringProtocolConfig
        cfg = SaturationMutagenesisConfig(parent_sequence="MKTAY", scoring_model="esm2stabp")
        assert isinstance(cfg, ScoringProtocolConfig)

    def test_iterative_masking_is_generative(self):
        from biolmai.pipeline import IterativeMaskingDMSConfig, GenerativeProtocolConfig
        cfg = IterativeMaskingDMSConfig(parent_sequence="MKTAY", model_name="esm2-650m")
        assert isinstance(cfg, GenerativeProtocolConfig)

    def test_direct_generation_is_generative(self):
        from biolmai.pipeline import DirectGenerationConfig, GenerativeProtocolConfig
        cfg = DirectGenerationConfig(model_name="protein-mpnn")
        assert isinstance(cfg, GenerativeProtocolConfig)

    def test_saturation_mutagenesis_rejects_generate_action(self):
        """'generate' was previously allowed by the old shared _ALLOWED_API_ACTIONS; must now be rejected."""
        from biolmai.pipeline import SaturationMutagenesisConfig
        with pytest.raises(ValueError, match="scoring_action must be one of"):
            SaturationMutagenesisConfig(
                parent_sequence="MKTAY", scoring_model="esm2stabp", scoring_action="generate"
            )

    def test_iterative_masking_rejects_generate_action(self):
        """'generate' was previously allowed by the old shared _ALLOWED_API_ACTIONS; must now be rejected."""
        from biolmai.pipeline import IterativeMaskingDMSConfig
        with pytest.raises(ValueError, match="action must be one of"):
            IterativeMaskingDMSConfig(
                parent_sequence="MKTAY", model_name="esm2-650m", action="generate"
            )

    def test_iterative_masking_rejects_encode_action(self):
        from biolmai.pipeline import IterativeMaskingDMSConfig
        with pytest.raises(ValueError, match="action must be one of"):
            IterativeMaskingDMSConfig(
                parent_sequence="MKTAY", model_name="esm2-650m", action="encode"
            )


# ---------------------------------------------------------------------------
# DMS provenance persisted in generation_metadata.metadata JSON field
# ---------------------------------------------------------------------------


class TestDMSProvenanceInMetadata:
    """Verify that DMS provenance columns are packed into the metadata JSON
    field of generation_metadata so they survive session end."""

    def _dispatch_sat(self, cfg, mock_instance, tmp_ds):
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(mock_instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        return ws_out, result

    def test_saturation_mutagenesis_provenance_in_metadata(self, tmp_ds):
        """sat_position, sat_wt_aa, sat_mut_aa, and the score value must be
        stored in generation_metadata.metadata JSON for every sat-mutagenesis row."""
        import json as _json

        scores_iter = iter(-float(i) for i in range(19))

        async def _predict(items, **kwargs):
            return [{"ddg": next(scores_iter, None)} for _ in items]

        instance = AsyncMock()
        instance.predict = _predict
        instance.shutdown = AsyncMock()

        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKTAY",
            scoring_model="thermompnn-d",
            positions=[0],          # one position → 19 variants
            top_n=3,
            score_field="ddg",
        )
        ws_out, result = self._dispatch_sat(cfg, instance, tmp_ds)
        assert result.output_count == 3

        # Fetch all generation_metadata rows from DuckDB
        rows = tmp_ds.conn.execute(
            "SELECT metadata FROM generation_metadata WHERE run_id = 'r1'"
        ).fetchall()
        assert len(rows) == 3, "Expected 3 generation_metadata rows"

        for (meta_json,) in rows:
            assert meta_json is not None, "metadata JSON must not be NULL for sat-mutagenesis rows"
            meta = _json.loads(meta_json)
            assert "sat_position" in meta, f"sat_position missing from metadata: {meta}"
            assert "sat_wt_aa" in meta, f"sat_wt_aa missing from metadata: {meta}"
            assert "sat_mut_aa" in meta, f"sat_mut_aa missing from metadata: {meta}"
            assert "ddg" in meta, f"ddg score missing from metadata: {meta}"
            # pos 0 of "MKTAY" is 'M'
            assert meta["sat_position"] == 0
            assert meta["sat_wt_aa"] == "M"
            assert isinstance(meta["ddg"], float)

    def test_iterative_masking_provenance_in_metadata(self, tmp_ds):
        """dms_round, dms_pos1, dms_aa1 (and optionally dms_pos2, dms_aa2 for
        round-2) must be stored in generation_metadata.metadata for DMS rows."""
        import json as _json

        # Round 1: pos 0 → 'A', pos 2 → 'G'
        r1_response = [
            {"logits": make_logits(5, {0: "A"}), "vocab_tokens": list(ALPHABET)},
            {"logits": make_logits(5, {2: "G"}), "vocab_tokens": list(ALPHABET)},
        ]
        # Round 2: AKTAY mask pos2 → 'D'; MKGAY mask pos0 → 'L'
        r2_response = [
            {"logits": make_logits(5, {2: "D"}), "vocab_tokens": list(ALPHABET)},
            {"logits": make_logits(5, {0: "L"}), "vocab_tokens": list(ALPHABET)},
        ]
        instance = AsyncMock()
        instance.predict = AsyncMock(side_effect=[r1_response, r2_response])
        instance.shutdown = AsyncMock()

        cfg = IterativeMaskingDMSConfig(
            parent_sequence="MKTAY",
            model_name="esm2-650m",
            positions=[0, 2],
            rounds=2,
            batch_size=100,
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        assert result.output_count == 2

        rows = tmp_ds.conn.execute(
            "SELECT metadata FROM generation_metadata WHERE run_id = 'r1'"
        ).fetchall()
        assert len(rows) == 2, "Expected 2 generation_metadata rows"

        for (meta_json,) in rows:
            assert meta_json is not None, "metadata JSON must not be NULL for DMS rows"
            meta = _json.loads(meta_json)
            assert "dms_round" in meta, f"dms_round missing from metadata: {meta}"
            assert "dms_pos1" in meta, f"dms_pos1 missing from metadata: {meta}"
            assert "dms_aa1" in meta, f"dms_aa1 missing from metadata: {meta}"
            # Round-2 rows must carry both mutation sites
            assert meta["dms_round"] == 2
            assert "dms_pos2" in meta, f"dms_pos2 missing from round-2 metadata: {meta}"
            assert "dms_aa2" in meta, f"dms_aa2 missing from round-2 metadata: {meta}"
            assert meta["dms_pos1"] in (0, 2)
            assert meta["dms_pos2"] in (0, 2)
            assert meta["dms_pos1"] != meta["dms_pos2"]


# ---------------------------------------------------------------------------
# Data quality / alignment guards (audit fixes)
# ---------------------------------------------------------------------------


class TestConstructionTimeGuards:
    """Validate that invalid config values are caught at construction, not at runtime."""

    def test_satmut_empty_parent_sequence_raises(self):
        with pytest.raises(ValueError, match="parent_sequence cannot be empty"):
            SaturationMutagenesisConfig(
                parent_sequence="",
                scoring_model="thermompnn-d",
            )

    def test_satmut_positions_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            SaturationMutagenesisConfig(
                parent_sequence="MKTAY",
                scoring_model="thermompnn-d",
                positions=[],
            )

    def test_satmut_positions_none_allowed(self):
        cfg = SaturationMutagenesisConfig(
            parent_sequence="MKTAY",
            scoring_model="thermompnn-d",
            positions=None,
        )
        assert cfg.positions is None

    def test_itermask_empty_parent_sequence_raises(self):
        with pytest.raises(ValueError, match="parent_sequence cannot be empty"):
            IterativeMaskingDMSConfig(
                parent_sequence="",
                model_name="esm2-650m",
            )


class TestGenerationAllRunsFailed:
    """When all n_runs parallel generation calls fail the error must surface."""

    PARENT = "MKTAY"

    def test_all_runs_failed_raises_not_silent(self, tmp_ds):
        """n_runs=3 all raise — first exception must propagate, not produce output_count=0."""
        instance = AsyncMock()
        instance.generate = AsyncMock(side_effect=RuntimeError("simulated auth failure"))
        instance.shutdown = AsyncMock()

        cfg = DirectGenerationConfig(
            model_name="progen2-small",
            sequence="MKTAY",
            num_sequences=2,
            n_runs=3,
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            with pytest.raises(RuntimeError, match="simulated auth failure"):
                asyncio.run(
                    stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
                )

    def test_one_run_succeeds_not_affected(self, tmp_ds):
        """One successful run out of n_runs=2 — no exception, output produced."""
        instance = AsyncMock()
        instance.generate = AsyncMock(side_effect=[
            RuntimeError("run 1 failed"),
            [{"sequence": "MKTAY"}],
        ])
        instance.shutdown = AsyncMock()

        cfg = DirectGenerationConfig(
            model_name="progen2-small",
            sequence="MKTAY",
            num_sequences=2,
            n_runs=2,
        )
        stage = GenerationStage(name="gen", config=cfg)
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            ws_out, result = asyncio.run(
                stage.process_ws(WorkingSet(frozenset()), tmp_ds, run_id="r1")
            )
        assert result.output_count >= 1


class TestSatMutPreflightFromGoodBatchOnly:
    """Preflight sample must only come from a correctly-sized batch."""

    PARENT = "MKTAY"

    def test_malformed_batch_does_not_poison_preflight(self, tmp_ds):
        """First batch returns wrong length; second returns correct length with the
        score field present. Preflight must not pass due to the malformed first batch."""
        instance = AsyncMock()
        # First batch: 1 result for 1 item (correct, if batch_size=1)
        # BUT the first element lacks the score_field — ensure preflight validates
        # using the first VALID batch only.
        call_count = 0

        async def mock_score(items, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Return one item for batch of 1 — correct length, BUT missing score_field
                return [{"wrong_key": 0.5}]
            # Subsequent batches also return wrong key
            return [{"wrong_key": 0.5} for _ in items]

        instance.score = AsyncMock(side_effect=mock_score)
        instance.shutdown = AsyncMock()

        cfg = SaturationMutagenesisConfig(
            parent_sequence=self.PARENT,
            scoring_model="thermompnn-d",
            scoring_action="score",
            score_field="ddg",
            positions=[0],
            batch_size=1,
        )
        with patch("biolmai.pipeline.generative.BioLMApiClient", mock_client_cls(instance)):
            with pytest.raises(ValueError, match="ddg"):
                asyncio.run(
                    GenerationStage(name="gen", config=cfg).process_ws(
                        WorkingSet(frozenset()), tmp_ds, run_id="r1"
                    )
                )
