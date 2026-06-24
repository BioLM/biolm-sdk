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
    assert "--modal-env" in result.output
    assert "--detach" in result.output


def test_server_stop_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "stop", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output


def test_server_group_lists_stop():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "--help"])
    assert result.exit_code == 0
    assert "stop" in result.output


def test_server_status_help_includes_modal_env():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "status", "--help"])
    assert result.exit_code == 0
    assert "--modal-env" in result.output
    assert "--limit" in result.output


def test_model_catalog_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "catalog"])
    assert result.exit_code == 0
    assert "esm2-8m" in result.output or "ESM2" in result.output


def test_version_command():
    runner = CliRunner()
    from biolm import __version__
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_flag():
    runner = CliRunner()
    from biolm import __version__
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
