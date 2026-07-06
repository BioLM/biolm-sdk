"""Tests for biolm hub CLI commands."""
from unittest.mock import patch

from click.testing import CliRunner

from biolm.cli import cli


@patch("biolm.hub.discovery.fetch_hub_status")
@patch("biolm.hub.config.write_hub_api_url")
def test_hub_set_success(mock_write, mock_status, tmp_path):
    mock_status.return_value = {
        "healthy": True,
        "slug_count": 2,
        "route_count": 3,
        "message": "biolm-hub gateway is running",
    }
    mock_write.return_value = tmp_path / "config.yaml"

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "set", "http://127.0.0.1:8000"])

    assert result.exit_code == 0
    mock_write.assert_called_once_with("http://127.0.0.1:8000/api/v1")
    assert "Connected to biolm-hub" in result.output


@patch("biolm.hub.discovery.fetch_hub_status")
def test_hub_set_unreachable(mock_status):
    mock_status.return_value = {"healthy": False, "message": "connection refused"}

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "set"])

    assert result.exit_code != 0


@patch("biolm.hub.config.clear_hub_config")
def test_hub_unset(mock_clear):
    mock_clear.return_value = True

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "unset"])

    assert result.exit_code == 0
    assert "Removed hub configuration" in result.output
