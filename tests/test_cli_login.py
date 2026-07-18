"""Tests for biolm account login CLI command."""
from unittest.mock import patch

from click.testing import CliRunner

from biolm.cli import cli
from biolm.core.const import BIOLM_PUBLIC_CLIENT_ID


@patch("biolm.cli.oauth_login")
@patch("biolm.cli.are_credentials_valid", return_value=False)
def test_cli_account_login_uses_default_client_id(mock_valid, mock_oauth_login):
    """Canonical `biolm account login` must resolve the public OAuth client ID."""
    mock_oauth_login.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_in": 3600,
    }

    runner = CliRunner()
    result = runner.invoke(cli, ["account", "login"])

    assert result.exit_code == 0, result.output
    mock_oauth_login.assert_called_once_with(
        client_id=BIOLM_PUBLIC_CLIENT_ID,
        scope="read write",
    )
    assert "Login succeeded" in result.output


@patch("biolm.cli.oauth_login")
@patch("biolm.cli.are_credentials_valid", return_value=True)
def test_cli_account_login_skips_when_already_authenticated(
    mock_valid, mock_oauth_login
):
    """Valid existing credentials should short-circuit before OAuth starts."""
    runner = CliRunner()
    result = runner.invoke(cli, ["account", "login"])

    assert result.exit_code == 0, result.output
    mock_oauth_login.assert_not_called()
    assert "already logged in" in result.output.lower()


@patch("biolm.cli.oauth_login")
@patch("biolm.cli.are_credentials_valid", return_value=False)
def test_alias_login_still_works_without_deprecation(mock_valid, mock_oauth_login):
    """Top-level `biolm login` remains a hidden compatibility alias."""
    mock_oauth_login.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_in": 3600,
    }

    runner = CliRunner()
    result = runner.invoke(cli, ["login"])

    assert result.exit_code == 0, result.output
    mock_oauth_login.assert_called_once_with(
        client_id=BIOLM_PUBLIC_CLIENT_ID,
        scope="read write",
    )
    assert "deprecated" not in result.output.lower()


@patch("biolm.cli.os.remove")
def test_cli_account_logout(mock_remove):
    runner = CliRunner()
    result = runner.invoke(cli, ["account", "logout"])

    assert result.exit_code == 0, result.output
    mock_remove.assert_called_once()
    assert "logged out" in result.output.lower()


@patch("biolm.cli.os.remove")
def test_alias_logout_still_works_without_deprecation(mock_remove):
    runner = CliRunner()
    result = runner.invoke(cli, ["logout"])

    assert result.exit_code == 0, result.output
    mock_remove.assert_called_once()
    assert "deprecated" not in result.output.lower()
