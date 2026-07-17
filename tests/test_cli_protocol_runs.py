"""Tests for remote protocol and run CLI commands."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from biolm.cli import cli
from biolm.protocol_runs import ProtocolRunError


@pytest.fixture
def protocol_client():
    with patch("biolm.cli.ProtocolClient", create=True) as factory:
        yield factory, factory.return_value


def invoke(*args):
    return CliRunner().invoke(cli, list(args))


def test_protocol_list_table(protocol_client):
    _, client = protocol_client
    client.list.return_value = {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [
            {
                "slug": "fold-and-score",
                "version": 3,
                "name": "Fold and score",
                "description": "Predict structures and score them.",
                "is_public": True,
                "owner_type": "organization",
                "input_fields": ["sequences", "model"],
            },
            {
                "slug": "design",
                "version": 1,
                "name": "Design",
                "description": "",
                "is_public": False,
                "owner_type": "user",
                "input_fields": ["sequence"],
            },
        ],
    }

    result = invoke("protocol", "list")

    assert result.exit_code == 0, result.output
    assert "fold-and-score" in result.output
    assert "Fold and score" in result.output
    assert "organization" in result.output
    assert "public" in result.output
    assert "sequences, model" in result.output
    assert "2 protocol" in result.output
    client.list.assert_called_once_with(search=None, page=1, page_size=20)


def test_protocol_list_json_preserves_pagination(protocol_client):
    _, client = protocol_client
    payload = {
        "count": 1,
        "next": "https://biolm.ai/api/protocols/?page=2",
        "previous": None,
        "results": [{"slug": "design", "version": 1}],
    }
    client.list.return_value = payload

    result = invoke(
        "protocol",
        "list",
        "--search",
        "design",
        "--page",
        "2",
        "--page-size",
        "5",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == payload
    client.list.assert_called_once_with(search="design", page=2, page_size=5)


def test_protocol_list_empty(protocol_client):
    _, client = protocol_client
    client.list.return_value = {
        "count": 0,
        "next": None,
        "previous": None,
        "results": [],
    }

    result = invoke("protocol", "list")

    assert result.exit_code == 0, result.output
    assert "No protocols found" in result.output


def test_protocol_list_api_error_is_click_error(protocol_client):
    _, client = protocol_client
    client.list.side_effect = ProtocolRunError("GET failed")

    result = invoke("protocol", "list")

    assert result.exit_code != 0
    assert "Error: GET failed" in result.output


def test_protocol_run_submits_json_inputs_without_waiting(
    protocol_client, tmp_path
):
    _, client = protocol_client
    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text(json.dumps({"sequence": "MKTAY", "rounds": 2}))
    run = MagicMock()
    run.run_id = "ALY_123"
    run.protocol_slug = "design"
    run.protocol_version = 2
    run.status = "scheduled"
    client.submit.return_value = run

    result = invoke(
        "protocol",
        "run",
        "design",
        "--inputs",
        str(inputs_file),
        "--version",
        "2",
        "--name",
        "experiment-1",
        "--environment-id",
        "22",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "run_id": "ALY_123",
        "protocol_slug": "design",
        "protocol_version": 2,
        "status": "scheduled",
    }
    client.submit.assert_called_once_with(
        "design",
        {"sequence": "MKTAY", "rounds": 2},
        version=2,
        run_name="experiment-1",
        environment_id=22,
    )
    run.wait.assert_not_called()


def test_protocol_run_waits_and_prints_results(protocol_client, tmp_path):
    _, client = protocol_client
    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text('{"sequence": "MKTAY"}')
    run = MagicMock()
    run.run_id = "ALY_123"
    run.protocol_slug = "design"
    run.protocol_version = 1
    run.status = "scheduled"
    run.wait.return_value = run
    run.results.return_value = {
        "run_id": "ALY_123",
        "status": "succeeded",
        "results": {"sequences": ["MKLAY"]},
    }
    client.submit.return_value = run

    result = invoke(
        "protocol",
        "run",
        "design",
        "-i",
        str(inputs_file),
        "--wait",
        "--timeout",
        "10",
        "--poll-interval",
        "0.25",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"] == {"sequences": ["MKLAY"]}
    run.wait.assert_called_once_with(
        timeout=10.0,
        show_progress=False,
        poll_interval=0.25,
    )
    run.results.assert_called_once_with()


@pytest.mark.parametrize("contents", ["not json", "[1, 2, 3]"])
def test_protocol_run_rejects_invalid_input_object(
    protocol_client, tmp_path, contents
):
    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text(contents)

    result = invoke("protocol", "run", "design", "-i", str(inputs_file))

    assert result.exit_code != 0
    assert "inputs" in result.output.lower()
    protocol_client[1].submit.assert_not_called()


def test_protocol_run_api_error_is_click_error(protocol_client):
    _, client = protocol_client
    client.submit.side_effect = ProtocolRunError("Invalid inputs")

    result = invoke("protocol", "run", "design")

    assert result.exit_code != 0
    assert "Error: Invalid inputs" in result.output


def test_protocol_run_rejects_zero_poll_interval_before_submission(protocol_client):
    result = invoke(
        "protocol",
        "run",
        "design",
        "--wait",
        "--poll-interval",
        "0",
    )

    assert result.exit_code != 0
    assert "greater than zero" in result.output
    protocol_client[1].submit.assert_not_called()


def test_protocol_status_json(protocol_client):
    _, client = protocol_client
    run = MagicMock()
    run.progress.return_value = {
        "run_id": "ALY_123",
        "status": "running",
        "progress_pct": 42,
        "channel_id": "telemetry_ALY_123",
    }
    client.get_run.return_value = run

    result = invoke("protocol", "status", "ALY_123", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["progress_pct"] == 42
    client.get_run.assert_called_once_with("ALY_123")
    run.progress.assert_called_once_with()


def test_protocol_wait_reconnects_and_returns_detail(protocol_client):
    _, client = protocol_client
    run = MagicMock()
    run.wait.return_value = run
    run.results.return_value = {
        "run_id": "ALY_123",
        "status": "succeeded",
        "results": {"score": 0.9},
    }
    client.get_run.return_value = run

    result = invoke(
        "protocol",
        "wait",
        "ALY_123",
        "--timeout",
        "30",
        "--poll-interval",
        "0.5",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "succeeded"
    run.wait.assert_called_once_with(
        timeout=30.0,
        show_progress=False,
        poll_interval=0.5,
    )
    run.results.assert_called_once_with()


def test_protocol_cancel_prints_server_response(protocol_client):
    _, client = protocol_client
    run = MagicMock()
    run.cancel.return_value = {
        "run_id": "ALY_123",
        "status": "cancellation_requested",
    }
    client.get_run.return_value = run

    result = invoke("protocol", "cancel", "ALY_123", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "cancellation_requested"
    run.cancel.assert_called_once_with()


def test_protocol_results_writes_json_file(protocol_client, tmp_path):
    _, client = protocol_client
    run = MagicMock()
    detail = {
        "run_id": "ALY_123",
        "status": "succeeded",
        "results": {"sequences": ["MKTAY"]},
    }
    run.results.return_value = detail
    client.get_run.return_value = run
    output = tmp_path / "result.json"

    result = invoke(
        "protocol",
        "results",
        "ALY_123",
        "--output",
        str(output),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text()) == detail
    assert str(output) in result.output


def test_protocol_download(protocol_client, tmp_path):
    _, client = protocol_client
    run = MagicMock()
    downloaded = tmp_path / "ALY_123_results.jsonl.zip"
    run.download.return_value = downloaded
    client.get_run.return_value = run

    result = invoke(
        "protocol",
        "download",
        "ALY_123",
        "--output-dir",
        str(tmp_path),
        "--file-type",
        "jsonl",
        "--overwrite",
    )

    assert result.exit_code == 0, result.output
    assert str(downloaded) in result.output
    run.download.assert_called_once_with(
        output_dir=str(tmp_path),
        file_type="jsonl",
        overwrite=True,
    )


def test_protocol_lifecycle_api_error_is_click_error(protocol_client):
    protocol_client[1].get_run.side_effect = ProtocolRunError("Run not found")

    result = invoke("protocol", "status", "missing")

    assert result.exit_code != 0
    assert "Error: Run not found" in result.output


def test_protocol_help_does_not_classify_run_status_as_authentication():
    result = invoke("protocol", "--help")

    assert result.exit_code == 0, result.output
    assert "Authentication" not in result.output
    assert "status" in result.output
