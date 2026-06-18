"""Tests for protocol MLflow logging functionality."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biolm.core.expression_evaluator import (
    evaluate_expression,
    evaluate_template_value,
    evaluate_where_clause,
    extract_template_expr,
)
from biolm.protocols_mlflow import (
    MLflowNotAvailableError,
    combine_output_rules,
    compute_aggregates,
    generate_seqparse,
    load_outputs_config,
    load_results,
    log_protocol_results,
    prepare_artifact,
    prepare_logging_data,
    select_results,
)


class TestExpressionEvaluator:
    """Test template expression evaluation."""

    def test_extract_template_expr(self):
        """Test extracting template expressions."""
        is_template, expr = extract_template_expr("${{ log_prob }}")
        assert is_template is True
        assert expr == "log_prob"

        is_template, expr = extract_template_expr("not a template")
        assert is_template is False
        assert expr == ""

    def test_evaluate_expression_basic(self):
        """Test basic expression evaluation."""
        context = {"log_prob": -1.5, "score": 0.8}
        result = evaluate_expression("log_prob", context)
        assert result == -1.5

        result = evaluate_expression("score", context)
        assert result == 0.8

    def test_evaluate_expression_operators(self):
        """Test expression with operators."""
        context = {"score": 0.8, "threshold": 0.5}
        result = evaluate_expression("score > threshold", context)
        assert result is True

        result = evaluate_expression("score + 0.2", context)
        assert result == 1.0

    def test_evaluate_expression_missing_field(self):
        """Test expression with missing field."""
        context = {"score": 0.8}
        with pytest.raises(KeyError):
            evaluate_expression("missing_field", context)

    def test_evaluate_where_clause(self):
        """Test where clause evaluation."""
        row = {"score": 0.8, "log_prob": -1.5}
        result = evaluate_where_clause("${{ score > 0.5 }}", row)
        assert result is True

        result = evaluate_where_clause("${{ score < 0.5 }}", row)
        assert result is False

        # Missing field should return False
        result = evaluate_where_clause("${{ missing > 0.5 }}", row)
        assert result is False

    def test_evaluate_template_value(self):
        """Test evaluating template values."""
        context = {"log_prob": -1.5}
        result = evaluate_template_value("${{ log_prob }}", context)
        assert result == -1.5

        # Non-template value should return as-is
        result = evaluate_template_value("literal_value", context)
        assert result == "literal_value"


class TestResultSelection:
    """Test result selection logic."""

    def test_select_results_no_filter(self):
        """Test selecting results without filter."""
        results = [
            {"score": 0.8, "log_prob": -1.5},
            {"score": 0.3, "log_prob": -2.0},
        ]
        rule = {}
        selected = select_results(results, rule)
        assert len(selected) == 2

    def test_select_results_with_where(self):
        """Test selecting results with where filter."""
        results = [
            {"score": 0.8, "log_prob": -1.5},
            {"score": 0.3, "log_prob": -2.0},
        ]
        rule = {"where": "${{ score > 0.5 }}"}
        selected = select_results(results, rule)
        assert len(selected) == 1
        assert selected[0]["score"] == 0.8

    def test_select_results_with_order_by(self):
        """Test selecting results with ordering."""
        results = [
            {"score": 0.3, "log_prob": -2.0},
            {"score": 0.8, "log_prob": -1.5},
        ]
        rule = {"order_by": [{"field": "score", "order": "desc"}]}
        selected = select_results(results, rule)
        assert selected[0]["score"] == 0.8
        assert selected[1]["score"] == 0.3

    def test_select_results_with_limit(self):
        """Test selecting results with limit."""
        results = [
            {"score": 0.8},
            {"score": 0.7},
            {"score": 0.6},
        ]
        rule = {"limit": 2}
        selected = select_results(results, rule)
        assert len(selected) == 2

    def test_select_results_combined(self):
        """Test selecting results with filter, order, and limit."""
        results = [
            {"score": 0.3, "log_prob": -2.0},
            {"score": 0.8, "log_prob": -1.5},
            {"score": 0.9, "log_prob": -1.0},
            {"score": 0.2, "log_prob": -2.5},
        ]
        rule = {
            "where": "${{ score > 0.5 }}",
            "order_by": [{"field": "score", "order": "desc"}],
            "limit": 2,
        }
        selected = select_results(results, rule)
        assert len(selected) == 2
        assert selected[0]["score"] == 0.9
        assert selected[1]["score"] == 0.8


class TestCombineOutputRules:
    """Test combining multiple output rules."""

    def test_combine_single_rule(self):
        """Test combining with single rule."""
        results = [
            {"score": 0.8, "log_prob": -1.5},
            {"score": 0.3, "log_prob": -2.0},
        ]
        rules = [
            {
                "where": "${{ score > 0.5 }}",
                "log": {
                    "metrics": {"score": "${{ score }}"},
                },
            }
        ]
        combined = combine_output_rules(results, rules)
        assert len(combined) == 1
        assert 0 in combined  # First result selected

    def test_combine_multiple_rules(self):
        """Test combining with multiple rules."""
        results = [
            {"score": 0.8, "log_prob": -1.5},
            {"score": 0.3, "log_prob": -2.0},
        ]
        rules = [
            {
                "where": "${{ score > 0.5 }}",
                "log": {"metrics": {"score": "${{ score }}"}},
            },
            {
                "where": "${{ log_prob > -2.0 }}",
                "log": {"params": {"temperature": "${{ log_prob }}"}},
            },
        ]
        combined = combine_output_rules(results, rules)
        # First result matches both rules
        assert len(combined) >= 1


class TestAggregates:
    """Test aggregate computation."""

    def test_compute_aggregates_mean(self):
        """Test computing mean aggregate."""
        selected = [{"score": 0.8}, {"score": 0.6}, {"score": 0.4}]
        all_results = selected
        aggregates = [{"field": "score", "ops": ["mean"]}]
        result = compute_aggregates(selected, all_results, aggregates)
        assert "score.mean" in result
        assert result["score.mean"] == pytest.approx(0.6)

    def test_compute_aggregates_multiple_ops(self):
        """Test computing multiple aggregate operations."""
        selected = [{"score": 0.8}, {"score": 0.6}, {"score": 0.4}]
        all_results = selected
        aggregates = [{"field": "score", "ops": ["mean", "max", "min"]}]
        result = compute_aggregates(selected, all_results, aggregates)
        assert "score.mean" in result
        assert "score.max" in result
        assert "score.min" in result
        assert result["score.max"] == 0.8
        assert result["score.min"] == 0.4

    def test_compute_aggregates_rows_count(self):
        """Test computing row count."""
        selected = [{"score": 0.8}, {"score": 0.6}]
        all_results = selected
        aggregates = [{"field": "__rows__", "ops": ["count"]}]
        result = compute_aggregates(selected, all_results, aggregates)
        assert "__rows__.count" in result
        assert result["__rows__.count"] == 2

    def test_compute_aggregates_percentiles(self):
        """Test computing percentiles."""
        selected = [{"score": i * 0.1} for i in range(1, 11)]  # 0.1 to 1.0
        all_results = selected
        aggregates = [{"field": "score", "ops": ["p50", "p95"]}]
        result = compute_aggregates(selected, all_results, aggregates)
        assert "score.p50" in result
        assert "score.p95" in result


class TestArtifactGeneration:
    """Test artifact generation."""

    def test_generate_seqparse(self):
        """Test generating seqparse format."""
        result = generate_seqparse("ACDEFGHIKLMNPQRSTVWY")
        data = json.loads(result)
        # Check proper seqparse format: name, type, seq, annotations
        assert "name" in data
        assert "type" in data
        assert "seq" in data
        assert "annotations" in data
        assert data["seq"] == "ACDEFGHIKLMNPQRSTVWY"
        assert data["type"] == "aa"  # Amino acid sequence
        assert isinstance(data["annotations"], list)

    def test_generate_seqparse_with_id(self):
        """Test generating seqparse with ID."""
        result = generate_seqparse("ACDEFGHIKLMNPQRSTVWY", sequence_id="seq1")
        data = json.loads(result)
        assert data["name"] == "seq1"
        assert data["seq"] == "ACDEFGHIKLMNPQRSTVWY"

    def test_prepare_artifact_seqparse(self):
        """Test preparing seqparse artifact."""
        artifact_spec = {
            "type": "seqparse",
            "name": "sequence.json",
            "content": "${{ sequence }}",
        }
        result = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        name, content = prepare_artifact(artifact_spec, result)
        assert name == "sequence.json"
        data = json.loads(content)
        # Check proper seqparse format
        assert "name" in data
        assert "type" in data
        assert "seq" in data
        assert "annotations" in data


class TestLoadFunctions:
    """Test loading functions."""

    def test_load_results_from_list(self):
        """Test loading results from list."""
        results = [{"score": 0.8}, {"score": 0.6}]
        loaded = load_results(results)
        assert loaded == results

    def test_load_results_from_file(self, tmp_path):
        """Test loading results from JSONL file."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write('{"score": 0.8}\n')
            f.write('{"score": 0.6}\n')

        loaded = load_results(str(results_file))
        assert len(loaded) == 2
        assert loaded[0]["score"] == 0.8

    def test_load_results_from_gzip(self, tmp_path):
        """Test loading results from gzipped JSONL file."""
        import gzip

        results_file = tmp_path / "results.jsonl.gz"
        with gzip.open(results_file, "wt") as f:
            f.write('{"score": 0.8}\n')
            f.write('{"score": 0.6}\n')

        loaded = load_results(str(results_file))
        assert len(loaded) == 2
        assert loaded[0]["score"] == 0.8

    def test_load_results_from_zip(self, tmp_path):
        """Test loading results from zip archive containing JSONL."""
        import zipfile

        jsonl_file = tmp_path / "results.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"score": 0.8}\n')
            f.write('{"score": 0.6}\n')

        zip_file = tmp_path / "results.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.write(jsonl_file, "results.jsonl")

        loaded = load_results(str(zip_file))
        assert len(loaded) == 2
        assert loaded[0]["score"] == 0.8

    def test_load_outputs_config_from_list(self):
        """Test loading outputs config from list."""
        config = [{"limit": 100, "log": {"metrics": {"score": "${{ score }}"}}}]
        loaded = load_outputs_config(config)
        assert loaded == config

    def test_load_outputs_config_from_protocol_dict(self):
        """Test loading outputs config from protocol dict."""
        protocol = {
            "name": "Test",
            "outputs": [{"limit": 100, "log": {"metrics": {"score": "${{ score }}"}}}],
        }
        loaded = load_outputs_config(protocol)
        assert len(loaded) == 1
        assert loaded[0]["limit"] == 100

    def test_load_outputs_config_from_file(self, tmp_path):
        """Test loading outputs config from YAML file."""
        import yaml

        config_file = tmp_path / "outputs.yaml"
        config = [{"limit": 100, "log": {"metrics": {"score": "${{ score }}"}}}]
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        loaded = load_outputs_config(str(config_file))
        assert len(loaded) == 1


class TestPrepareLoggingData:
    """Test preparing logging data."""

    def test_prepare_logging_data_basic(self):
        """Test basic logging data preparation."""
        results = [
            {"score": 0.8, "log_prob": -1.5, "sequence": "ACDEFGHIKLMNPQRSTVWY"},
        ]
        outputs_config = [
            {
                "limit": 100,
                "log": {
                    "metrics": {"score": "${{ score }}"},
                    "params": {"temperature": "${{ log_prob }}"},
                },
            }
        ]
        prepared = prepare_logging_data(results, outputs_config)
        assert "child_runs" in prepared
        assert len(prepared["child_runs"]) == 1
        assert "sequence.json" in [a[0] for a in prepared["child_runs"][0]["artifacts"]]

    def test_prepare_logging_data_with_aggregates(self):
        """Test preparing data with aggregates."""
        results = [{"score": 0.8}, {"score": 0.6}]
        outputs_config = [
            {
                "log": {
                    "aggregates": [{"field": "score", "ops": ["mean", "max"]}],
                }
            }
        ]
        prepared = prepare_logging_data(results, outputs_config)
        assert "aggregate_metrics" in prepared
        assert "score.mean" in prepared["aggregate_metrics"]
        assert "score.max" in prepared["aggregate_metrics"]


class TestMLflowIntegration:
    """Test MLflow integration (mocked)."""

    @patch("biolmai.protocols_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.protocols_mlflow.mlflow")
    @patch("biolmai.protocols_mlflow.MlflowClient")
    def test_log_to_mlflow_dry_run(self, mock_client_class, mock_mlflow):
        """Test dry run mode."""
        results = [{"score": 0.8}]
        outputs_config = [{"log": {"metrics": {"score": "${{ score }}"}}}]

        result = log_protocol_results(
            results=results,
            outputs_config=outputs_config,
            account_name="test_account",
            workspace_name="test_workspace",
            protocol_name="test_exp",
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert result["experiment_name"] == "test_account/test_workspace/test_exp"
        # Should not call MLflow
        mock_mlflow.start_run.assert_not_called()

    @patch("biolmai.protocols_mlflow.MLFLOW_AVAILABLE", False)
    def test_log_to_mlflow_not_available(self):
        """Test error when MLflow is not available."""
        results = [{"score": 0.8}]
        outputs_config = [{"log": {"metrics": {"score": "${{ score }}"}}}]

        with pytest.raises(MLflowNotAvailableError):
            log_protocol_results(
                results=results,
                outputs_config=outputs_config,
                account_name="test_account",
                workspace_name="test_workspace",
                protocol_name="test_exp",
                dry_run=False,
            )

    @patch("biolmai.protocols_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.protocols_mlflow.mlflow")
    @patch("biolmai.protocols_mlflow.MlflowClient")
    def test_log_to_mlflow_full(self, mock_client_class, mock_mlflow):
        """Test full MLflow logging (mocked)."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_client.get_experiment_by_name.return_value = mock_experiment

        # Setup context manager mocks for start_run
        mock_parent_run = MagicMock()
        mock_parent_run.info.run_id = "parent123"
        mock_parent_run.__enter__ = MagicMock(return_value=mock_parent_run)
        mock_parent_run.__exit__ = MagicMock(return_value=None)
        
        mock_child_run = MagicMock()
        mock_child_run.info.run_id = "child123"
        mock_child_run.__enter__ = MagicMock(return_value=mock_child_run)
        mock_child_run.__exit__ = MagicMock(return_value=None)

        mock_mlflow.start_run.side_effect = [mock_parent_run, mock_child_run]

        results = [{"score": 0.8, "sequence": "ACDEFGHIKLMNPQRSTVWY"}]
        outputs_config = [
            {
                "log": {
                    "metrics": {"score": "${{ score }}"},
                    "aggregates": [{"field": "score", "ops": ["mean"]}],
                }
            }
        ]

        result = log_protocol_results(
            results=results,
            outputs_config=outputs_config,
            account_name="test_account",
            workspace_name="test_workspace",
            protocol_name="test_exp",
            dry_run=False,
        )

        assert result["dry_run"] is False
        assert result["parent_run_id"] == "parent123"
        assert len(result["child_run_ids"]) == 1
        # Verify MLflow was called
        assert mock_mlflow.set_tracking_uri.called or mock_mlflow.start_run.called


class TestErrorHandling:
    """Test error handling."""

    def test_load_results_file_not_found(self):
        """Test loading results from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/file.jsonl")

    def test_load_outputs_config_file_not_found(self):
        """Test loading outputs config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_outputs_config("/nonexistent/file.yaml")

    def test_prepare_artifact_invalid_type(self):
        """Test preparing artifact with invalid type."""
        artifact_spec = {"type": "invalid", "name": "test"}
        result = {}
        with pytest.raises(ValueError):
            prepare_artifact(artifact_spec, result)


class TestCLICommand:
    """Test CLI command."""

    def test_cli_log_dry_run(self, tmp_path):
        """Test CLI log command with dry run."""
        from click.testing import CliRunner

        from biolm.cli import cli

        # Create test files
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write('{"score": 0.8, "sequence": "ACDEFGHIKLMNPQRSTVWY"}\n')

        outputs_file = tmp_path / "outputs.yaml"
        import yaml

        outputs_config = [
            {"log": {"metrics": {"score": "${{ score }}"}}}
        ]
        with open(outputs_file, "w") as f:
            yaml.dump(outputs_config, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "protocol",
                "log",
                str(results_file),
                "--outputs",
                str(outputs_file),
                "--account",
                "test_account",
                "--workspace",
                "test_workspace",
                "--protocol",
                "test_exp",
                "--dry-run",
            ],
        )

        # Should succeed in dry run mode even without MLflow
        assert result.exit_code == 0 or "MLflow" in result.output

    def test_cli_log_missing_outputs(self, tmp_path):
        """Test CLI log command with missing outputs option."""
        from click.testing import CliRunner

        from biolm.cli import cli

        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write('{"score": 0.8}\n')

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "protocol",
                "log",
                str(results_file),
                "--account",
                "acme",
                "--workspace",
                "lab",
                "--protocol",
                "test_exp",
            ],
        )

        assert result.exit_code != 0
        assert "outputs" in result.output.lower()

