"""Tests for top-level CLI help behavior."""
import click
import pytest
from click.testing import CliRunner

import biolm.cli as cli_module
from biolm.cli import cli


def test_bare_biolm_shows_help():
    """Running `biolm` with no subcommand should show the help menu."""
    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert result.exit_code == 0, result.output
    assert "BioLM CLI" in result.output


def test_hidden_commands_are_omitted_from_help(monkeypatch):
    """Commands marked hidden should remain callable without appearing in help."""
    hidden = click.Command(
        "temporary-hidden",
        callback=lambda: click.echo("hidden command called"),
        hidden=True,
    )
    monkeypatch.setitem(cli.commands, hidden.name, hidden)
    runner = CliRunner()

    help_result = runner.invoke(cli, ["--help"])
    invoke_result = runner.invoke(cli, [hidden.name])

    assert help_result.exit_code == 0, help_result.output
    assert hidden.name not in help_result.output
    assert invoke_result.exit_code == 0, invoke_result.output
    assert "hidden command called" in invoke_result.output


def test_hidden_alias_helper_copies_only_leaf_commands():
    """Hidden aliases should not mutate targets or shallow-copy command groups."""
    parent = click.Group("parent")
    target = click.Command("target", callback=lambda: None)
    helper = cli_module._hidden_leaf_alias

    alias = helper(parent, "alias", target)

    assert alias is not target
    assert alias.hidden is True
    assert target.hidden is False
    assert parent.commands["alias"] is alias
    with pytest.raises(TypeError, match="leaf"):
        helper(parent, "unsafe", click.Group("group"))


def test_top_level_help_expands_child_command_paths():
    """Top-level help lists full leaf paths, not just group names."""
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "workspace list" in result.output
    assert "workspace show" in result.output
    assert "dataset create" in result.output


def test_top_level_help_does_not_use_platform_section():
    """The generic backend-oriented Platform heading should be retired."""
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Platform" not in result.output


def test_nested_group_help_lists_visible_direct_children():
    """Group help should list its direct visible children."""
    result = CliRunner().invoke(cli, ["workspace", "--help"])

    assert result.exit_code == 0, result.output
    assert "list" in result.output
    assert "show" in result.output
    assert "switch" in result.output
    assert "create" in result.output


def test_version_command_is_callable_but_hidden_from_help():
    """The compatibility command form should work without menu membership."""
    runner = CliRunner()

    help_result = runner.invoke(cli, ["--help"])
    invoke_result = runner.invoke(cli, ["version"])

    assert help_result.exit_code == 0, help_result.output
    assert "\n│ version" not in help_result.output
    assert invoke_result.exit_code == 0, invoke_result.output
    assert "biolm" in invoke_result.output


def test_help_descriptions_are_not_truncated():
    """Command summaries should use the full first docstring line, not Click's 45-char cut."""
    runner = CliRunner()
    result = runner.invoke(cli, ["account", "--help"])

    assert result.exit_code == 0, result.output
    assert "Log out and remove saved OAuth credentials from" in result.output
    assert "Log out and remove saved OAuth credentials..." not in result.output


def test_final_top_level_help_hierarchy():
    """Top-level help lists leaf paths grouped under command-group panels."""
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    for section in (
        "Account",
        "Workspace",
        "Hub",
        "Models",
        "Protocols",
        "Datasets",
    ):
        assert section in result.output
    assert "status" in result.output
    assert "whoami" in result.output
    assert "account login" in result.output
    assert "workspace list" in result.output
    assert "dataset create" in result.output
    assert "dataset push" in result.output
    assert "Platform" not in result.output
    assert "\n│ version" not in result.output
    assert "\n│ login" not in result.output
    assert "usage show" not in result.output
    assert "org create" not in result.output
