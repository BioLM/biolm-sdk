"""Tests for top-level CLI help behavior."""
from click.testing import CliRunner

from biolm.cli import cli


def test_bare_biolm_shows_help():
    """Running `biolm` with no subcommand should show the help menu."""
    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert result.exit_code == 0, result.output
    assert "BioLM CLI" in result.output
    assert "Authentication" in result.output
    assert "login" in result.output


def test_help_descriptions_are_not_truncated():
    """Command summaries should use the full first docstring line, not Click's 45-char cut."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Log out and remove saved OAuth credentials from" in result.output
    assert "Log out and remove saved OAuth credentials..." not in result.output
