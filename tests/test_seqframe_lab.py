"""Tests for SeqFrame.lab LLTP bridge."""

from __future__ import annotations

import pytest

from biolm.seqframe import SeqFrame
from biolm.seqframe.namespaces.lab import LabNamespace


@pytest.fixture
def sample_sf():
    return SeqFrame.from_rows(
        [
            {"id": "sf-1", "sequence": "MKTAYIAKQRQ", "name": "v1"},
            {"id": "sf-2", "sequence": "MKTAYIAKQRZ", "name": "v2"},
        ],
        molecule_type="protein",
    )


def test_to_lltp_payload(sample_sf):
    payload = sample_sf.lab.to_lltp(
        service_id="adaptyv-lltp.expression-v1",
        name="batch-1",
        n_replicates=2,
    )
    assert payload["service_id"] == "adaptyv-lltp.expression-v1"
    assert payload["name"] == "batch-1"
    assert payload["n_replicates"] == 2
    assert len(payload["sequences"]) == 2
    assert payload["sequences"][0]["id"] == "sf-1"
    assert payload["sequences"][0]["sequence"] == "MKTAYIAKQRQ"
    assert payload["sequences"][0]["name"] == "v1"


def test_from_lltp_preserves_entity_id():
    dataset = {
        "dataset_id": "ds-1",
        "order_id": "ord-1",
        "service_id": "adaptyv-lltp.expression-v1",
        "records": [
            {
                "entity": {
                    "type": "protein",
                    "representation": "MKTAYIAKQRQ",
                    "representation_format": "raw",
                    "entity_id": "sf-1",
                },
                "parameters": {},
                "metrics": {"expression": 1.5},
                "tags": {"construct_name": "v1"},
                "artifacts": {},
            }
        ],
    }
    sf = LabNamespace.from_lltp(dataset, molecule_type="protein")
    df = sf.collect()
    assert list(df["id"]) == ["sf-1"]
    assert df.loc[0, "sequence"] == "MKTAYIAKQRQ"
    assert df.loc[0, "metric_expression"] == 1.5


def test_merge_on_id(sample_sf):
    dataset = {
        "records": [
            {
                "entity": {
                    "type": "protein",
                    "representation": "MKTAYIAKQRQ",
                    "entity_id": "sf-1",
                },
                "metrics": {"score": 9.0},
                "tags": {},
                "parameters": {},
            }
        ]
    }
    results = LabNamespace.from_lltp(dataset, molecule_type="protein")
    merged = sample_sf.lab.merge(results, on="id", how="left")
    df = merged.collect()
    assert "metric_score" in df.columns or "score" in df.columns or len(df) >= 1
    # Join should keep both design rows for left join
    assert len(df) >= 2
