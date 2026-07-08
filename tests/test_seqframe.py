"""Tests for SeqFrame v0."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biolm.seqframe import SEQFRAME_VERSION, SeqFrame, SeqFrameMetadata
from biolm.seqframe.metadata import METADATA_SCHEMA_KEY, METADATA_VERSION_KEY
from biolm.seqframe.rows import hash_sequence


_SEQFRAME_PREFIXES = ("biolm.seqframe",)


def _seqframe_module_keys() -> list[str]:
    keys: list[str] = []
    for prefix in _SEQFRAME_PREFIXES:
        for key in sys.modules:
            if key == prefix or key.startswith(prefix + "."):
                keys.append(key)
    return keys


_SENTINEL = object()


@contextmanager
def _isolated_seqframe_import(**fake_modules):
    saved = {k: v for k, v in sys.modules.items() if k in _seqframe_module_keys()}
    saved_fakes = {name: sys.modules.get(name, _SENTINEL) for name in fake_modules}
    try:
        for key in _seqframe_module_keys():
            del sys.modules[key]
        for name, val in fake_modules.items():
            sys.modules[name] = val
        err = None
        try:
            importlib.import_module("biolm.seqframe")
        except ImportError as exc:
            err = exc
        yield err
    finally:
        for key in _seqframe_module_keys():
            sys.modules.pop(key, None)
        sys.modules.update(saved)
        for name, val in saved_fakes.items():
            if val is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val


@pytest.fixture(autouse=True)
def _restore_seqframe_after_test():
    yield
    if "biolm.seqframe" not in sys.modules:
        try:
            importlib.import_module("biolm.seqframe")
        except ImportError:
            pass


SAMPLE_ROWS = [
    {"sequence": "MKLLIV", "id": "seq1"},
    {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq2"},
    {"sequence": "ATCGATCG", "id": "seq3"},
]


class TestSeqFrameMetadata:
    def test_parquet_metadata_round_trip(self):
        meta = SeqFrameMetadata(
            sequence_column="sequence",
            id_column="id",
            molecule_type="protein",
            created_by="test",
            version=SEQFRAME_VERSION,
        )
        encoded = meta.to_parquet_metadata()
        restored = SeqFrameMetadata.from_parquet_metadata(encoded)
        assert restored.sequence_column == "sequence"
        assert restored.id_column == "id"
        assert restored.molecule_type == "protein"
        assert restored.version == SEQFRAME_VERSION


class TestSeqFrameIO:
    def test_from_fasta_and_parquet_round_trip(self, tmp_path):
        fasta = tmp_path / "seqs.fasta"
        fasta.write_text(
            ">seq1\nMKLLIV\n>seq2\nACDEFGHIKLMNPQRSTVWY\n",
            encoding="utf-8",
        )
        sf = SeqFrame.from_fasta(fasta)
        assert len(sf) == 2
        assert sf.schema.molecule_type == "protein"

        out = tmp_path / "out.parquet"
        sf2 = sf.io.to_parquet(out)
        sf3 = SeqFrame.read(out)
        df = sf3.collect()
        assert list(df["id"]) == ["seq1", "seq2"]
        assert "sequence_hash" in df.columns

    def test_csv_round_trip(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,sequence\nseq1,MKLLIV\nseq2,ACDEFGHIKLMNPQRSTVWY\n",
            encoding="utf-8",
        )
        sf = SeqFrame.from_csv(csv_path)
        out_csv = tmp_path / "out.csv"
        sf.io.to_csv(out_csv)
        loaded = pd.read_csv(out_csv)
        assert len(loaded) == 2

    def test_read_rejects_plain_parquet(self, tmp_path):
        path = tmp_path / "plain.parquet"
        table = pa.table({"a": [1, 2], "b": [3, 4]})
        pq.write_table(table, path)
        with pytest.raises(ValueError, match="not a SeqFrame Parquet"):
            SeqFrame.read(path)


class TestSeqFrameQuery:
    def test_lazy_filter_and_limit(self, tmp_path):
        sf = SeqFrame.from_rows(SAMPLE_ROWS)
        original_len = len(sf)
        filtered = sf.query.filter("length < 15").query.limit(1)
        assert len(filtered) == 1
        assert len(sf) == original_len

    def test_join_on_sequence_hash(self):
        left = SeqFrame.from_rows([{"sequence": "MKLLIV", "id": "a", "score": 1.0}])
        right = SeqFrame.from_rows(
            [{"sequence": "MKLLIV", "id": "b", "annotation": "test"}]
        )
        joined = left.query.join(right, on="sequence_hash")
        df = joined.collect()
        assert len(df) == 1
        assert df.iloc[0]["sequence_hash"] == hash_sequence("MKLLIV")

    def test_select_and_sort(self):
        sf = SeqFrame.from_rows(SAMPLE_ROWS)
        result = sf.query.select(["id", "length"]).query.sort("length", ascending=False)
        df = result.collect()
        assert list(df.columns) == ["id", "length"]
        assert df.iloc[0]["length"] >= df.iloc[-1]["length"]


class TestSeqFrameProtocol:
    def test_from_protocol_run(self, tmp_path):
        mock_run = MagicMock()
        mock_run.to_dataframe.return_value = pd.DataFrame(
            {
                "id": ["p1"],
                "sequence": ["MKLLIV"],
                "stability": [0.9],
            }
        )
        sf = SeqFrame.from_protocol(mock_run, output_dir=tmp_path)
        df = sf.collect()
        assert len(df) == 1
        assert df.iloc[0]["stability"] == 0.9


class TestSeqFrameModels:
    def test_predict_merges_column(self):
        sf = SeqFrame.from_rows([{"sequence": "MKLLIV", "id": "seq1"}])
        mock_api = MagicMock()
        mock_api.predict.return_value = [{"value": 0.75}]
        mock_api.shutdown.return_value = None

        with patch("biolm.core.http.BioLMApi", return_value=mock_api):
            enriched = sf.models.predict("esm2-8m", column="score", batch_size=1)

        df = enriched.collect()
        assert "score" in df.columns
        assert df.iloc[0]["score"] == 0.75
        mock_api.shutdown.assert_called_once()


class TestSeqFrameBio:
    def test_length_column(self):
        sf = SeqFrame.from_rows([{"sequence": "MKLLIV", "id": "seq1"}])
        result = sf.bio.length(column="seq_len")
        df = result.collect()
        assert df.iloc[0]["seq_len"] == 6

    def test_lab_stub_raises(self):
        sf = SeqFrame.from_rows(SAMPLE_ROWS[:1])
        with pytest.raises(NotImplementedError, match="LLTP"):
            sf.lab.to_lltp()


class TestSeqFrameExtrasGate:
    def test_missing_duckdb_raises_helpful_error(self):
        with _isolated_seqframe_import(duckdb=None) as err:
            assert err is not None
            assert "duckdb" in str(err)
            assert "biolm-sdk[seqframe]" in str(err)

    def test_import_succeeds_when_deps_present(self):
        import biolm.seqframe as seqframe

        assert hasattr(seqframe, "SeqFrame")
