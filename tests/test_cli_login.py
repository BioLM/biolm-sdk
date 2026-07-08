"""Tests for biolm login CLI command."""
from unittest.mock import patch

from click.testing import CliRunner

from biolm.cli import cli
from biolm.core.const import BIOLM_PUBLIC_CLIENT_ID


@patch("biolm.cli.oauth_login")
@patch("biolm.cli.are_credentials_valid", return_value=False)
def test_cli_login_uses_default_client_id(mock_valid, mock_oauth_login):
    """Default `biolm login` must resolve the public OAuth client ID in the CLI."""
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
    assert "Login succeeded" in result.output


@patch("biolm.cli.oauth_login")
@patch("biolm.cli.are_credentials_valid", return_value=True)
def test_cli_login_skips_when_already_authenticated(mock_valid, mock_oauth_login):
    """Valid existing credentials should short-circuit before OAuth starts."""
    runner = CliRunner()
    result = runner.invoke(cli, ["login"])

    assert result.exit_code == 0, result.output
    mock_oauth_login.assert_not_called()
    assert "already logged in" in result.output.lower()
