"""Offline API tests for biolm.lab with a mocked VendorClient."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from biolm.lab import confirm, results, status, submit
from biolm.lab.config import load_config
from biolm.lab.runs import load_run
from biolm.seqframe import SeqFrame


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "version": 1,
        "default_connector": "adaptyv",
        "connectors": {
            "adaptyv": {
                "auth": {"token_env": "ADAPTYV_API_TOKEN"},
                "defaults": {"service_id": "adaptyv-lltp.expression-v1"},
            }
        },
        "experiments": {
            "express": {
                "connector": "adaptyv",
                "service_id": "adaptyv-lltp.expression-v1",
            }
        },
    }
    path = tmp_path / "lltp.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    monkeypatch.setenv("ADAPTYV_API_TOKEN", "test-token")
    return tmp_path


@pytest.fixture
def sf():
    return SeqFrame.from_rows(
        [{"id": "sf-1", "sequence": "MKTAYIAKQRQ"}],
        molecule_type="protein",
    )


def _mock_client():
    client = MagicMock()
    client.submit.return_value = {
        "vendor_order_id": "exp-1",
        "experiment_id": "exp-1",
        "status": "waiting_for_confirmation",
    }
    client.poll.return_value = {
        "experiment_id": "exp-1",
        "status": "waiting_for_confirmation",
    }
    client.map_status.return_value = [
        {
            "requirement_id": "req.quote-approval",
            "type": "APPROVAL",
            "status": "AWAITING",
        }
    ]
    client.confirm_quote.return_value = {
        "vendor_order_id": "exp-1",
        "experiment_id": "exp-1",
        "status": "in_progress",
    }
    client.fetch_results.return_value = {"items": []}
    client.to_result.return_value = {
        "dataset_id": "ds-1",
        "order_id": "exp-1",
        "service_id": "adaptyv-lltp.expression-v1",
        "records": [
            {
                "entity": {
                    "type": "protein",
                    "representation": "MKTAYIAKQRQ",
                    "entity_id": "sf-1",
                },
                "metrics": {},
                "tags": {},
                "parameters": {},
            }
        ],
    }
    return client


def test_submit_status_confirm_results(project, sf):
    client = _mock_client()
    cfg = load_config(project / "lltp.yaml")

    run = submit(sf, experiment="express", config=cfg, root=project, client=client)
    assert run.run_id
    client.submit.assert_called_once()
    payload = client.submit.call_args[0][0]
    assert payload["sequences"][0]["id"] == "sf-1"

    run2 = status(run.run_id, config=cfg, root=project, client=client)
    assert run2.status == "blocked"
    client.poll.assert_called()

    run3 = confirm(run.run_id, config=cfg, root=project, client=client)
    assert run3.status == "in_progress"
    client.confirm_quote.assert_called_once()

    sf_out = results(run.run_id, config=cfg, root=project, client=client)
    df = sf_out.collect()
    assert list(df["id"]) == ["sf-1"]

    disk = load_run(run.run_id, root=project)
    assert disk.status == "completed"
    assert disk.result["record_count"] == 1


def test_cli_lab_init_and_list(project):
    from click.testing import CliRunner

    from biolm.cli import cli

    runner = CliRunner()
    # init into a new file
    result = runner.invoke(cli, ["lab", "init", "--path", "lltp2.yaml", "--force"])
    assert result.exit_code == 0, result.output
    assert Path("lltp2.yaml").is_file()

    # list with no runs
    result = runner.invoke(cli, ["lab", "list"])
    assert result.exit_code == 0
