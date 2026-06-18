"""Tests for biolm server CLI commands."""
from click.testing import CliRunner

from biolm.cli import cli


def test_server_status_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "status"])
    assert result.exit_code == 0


def test_server_start_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "start", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output


def test_model_catalog_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "catalog"])
    assert result.exit_code == 0
    assert "esm2-8m" in result.output or "ESM2" in result.output
