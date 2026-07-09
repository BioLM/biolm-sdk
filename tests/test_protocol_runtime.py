"""Tests for biolm.protocols.runtime (Local Protocol Profile v1)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from biolm.protocols.runtime.profile import (
    PROFILE_DOC,
    UnsupportedProtocolFeature,
    check_supported,
)
from biolm.protocols.runtime.compile import build_execution_plan, compile_protocol
from biolm.protocols.runtime.expressions import evaluate_tree
from biolm.protocols.runtime.mapping import parse_mapping_entry
from biolm.protocols.runtime import run_local_protocol
from biolm.pipeline.data import PredictionStage

FIXTURES = Path(__file__).parent / "fixtures" / "protocols"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return yaml.safe_load(f)


class TestProfile:
    def test_profile_rejects_gather(self):
        protocol = _load_fixture("gather_protocol.yaml")
        with pytest.raises(UnsupportedProtocolFeature) as exc:
            check_supported(protocol)
        assert PROFILE_DOC in str(exc.value)
        assert "gather" in str(exc.value).lower()

    def test_profile_rejects_foreach(self):
        protocol = _load_fixture("foreach_protocol.yaml")
        with pytest.raises(UnsupportedProtocolFeature) as exc:
            check_supported(protocol)
        assert PROFILE_DOC in str(exc.value)


class TestCompile:
    def test_compile_single_predict_task(self):
        protocol = _load_fixture("esmfold_single.yaml")
        plan = build_execution_plan(protocol, {"sequence": "MKLLIV"})
        assert len(plan.tasks) == 1
        task = plan.tasks[0]
        assert task.task_id == "fold"
        assert task.slug == "esmfold"
        assert task.action == "predict"
        assert "mean_plddt" in task.stage_kwargs.get("extractions", [])
        assert task.stage_kwargs.get("structure_output") is not None

        _, pipeline = compile_protocol(protocol, {"sequence": "MKLLIV"})
        stage = pipeline.stages[0]
        assert isinstance(stage, PredictionStage)
        assert stage.model_name == "esmfold"
        assert stage.action == "predict"

    def test_compile_depends_on(self):
        protocol = _load_fixture("two_task_dag.yaml")
        plan = build_execution_plan(protocol, {"sequences": ["MKLLIV", "MKTAY"]})
        assert [t.task_id for t in plan.tasks] == ["encode", "score"]
        assert plan.tasks[1].depends_on == ["encode"]

        _, pipeline = compile_protocol(protocol, {"sequences": ["MKLLIV"]})
        score = pipeline.stages[1]
        assert score.depends_on == ["encode"]

    def test_expressions_in_params(self):
        protocol = _load_fixture("esmfold_single.yaml")
        protocol["tasks"][0]["request_body"]["params"] = {
            "num_samples": "${{ n_samples // 2 }}"
        }
        plan = build_execution_plan(protocol, {"sequence": "MKLLIV", "n_samples": 10})
        assert plan.tasks[0].params["num_samples"] == 5


class TestMapping:
    def test_mapping_jmespath_tail(self):
        spec = parse_mapping_entry("pdb", "${{ response.results[*].pdb }}")
        assert spec.response_key == "pdb"
        assert spec.kind == "structure"

    def test_mapping_literal_scalar(self):
        spec = parse_mapping_entry("plddt", "mean_plddt")
        assert spec.response_key == "mean_plddt"
        assert spec.kind == "scalar"


class TestExpressions:
    def test_evaluate_tree_inputs_only(self):
        ctx = {"n_samples": 8, "sequences": ["AAA"]}
        out = evaluate_tree({"k": "${{ n_samples // 2 }}"}, ctx)
        assert out["k"] == 4


class TestRunLocal:
    def _mock_predict_api(self, results):
        mock = AsyncMock()
        mock.predict = AsyncMock(return_value=results)
        mock.shutdown = AsyncMock()
        return mock

    def test_run_local_protocol_e2e(self, tmp_path):
        protocol = _load_fixture("esmfold_single.yaml")
        mock_api = self._mock_predict_api(
            [{"mean_plddt": 88.0, "pdb": "ATOM 1"}]
        )
        with patch("biolm.pipeline.data.BioLMApiClient", return_value=mock_api):
            result = run_local_protocol(
                protocol,
                inputs={"sequence": "MKLLIV"},
                output_dir=tmp_path,
                verbose=False,
            )
        assert len(result.records) == 1
        assert result.run_id
        row = result.records[0]
        assert row.get("mean_plddt") == 88.0 or row.get("fold") is not None
        assert "sequence" in row or "sequence_id" in row

    def test_protocol_execute(self, tmp_path):
        from biolm.protocols import Protocol

        protocol = Protocol(str(FIXTURES / "esmfold_single.yaml"))
        mock_api = self._mock_predict_api(
            [{"mean_plddt": 90.0, "pdb": "ATOM 1"}]
        )
        with patch("biolm.pipeline.data.BioLMApiClient", return_value=mock_api):
            result = protocol.execute(inputs={"sequence": "MKLLIV"})
        assert len(result.records) == 1


class TestCLI:
    def test_cli_protocol_run(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from biolm.cli.entry import cli

        protocol_path = FIXTURES / "esmfold_single.yaml"
        mock_api = AsyncMock()
        mock_api.predict = AsyncMock(
            return_value=[{"mean_plddt": 77.0, "pdb": "ATOM"}]
        )
        mock_api.shutdown = AsyncMock()

        with patch("biolm.pipeline.data.BioLMApiClient", return_value=mock_api):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "protocol",
                    "run",
                    str(protocol_path),
                    "--input",
                    "sequence=MKLLIV",
                    "--json",
                ],
            )
        assert result.exit_code == 0, result.output
        records = json.loads(result.output)
        assert len(records) == 1
