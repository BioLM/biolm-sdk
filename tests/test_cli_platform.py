"""Tests for platform management CLI commands."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from biolm.cli import cli
from biolm.platform import PlatformError, Workspace


PERSONAL = Workspace("user", 1, 11, "alice", "default")
ORG = Workspace("organization", 7, 22, "acme", "research")


@pytest.fixture
def platform_client():
    with patch("biolm.cli.PlatformClient", create=True) as factory:
        client = factory.return_value.__enter__.return_value
        yield factory, client


def invoke(*args):
    return CliRunner().invoke(cli, list(args))


def test_platform_groups_and_commands_are_registered():
    result = invoke("--help")
    assert result.exit_code == 0, result.output
    assert "workspace list" in result.output
    assert "workspace show" in result.output
    assert "workspace switch" in result.output
    assert "workspace create" in result.output
    assert "org list" in result.output
    assert "org show" in result.output
    assert "org create" in result.output
    assert "org invite" in result.output
    assert "budget show" in result.output
    assert "budget set" in result.output


def test_workspace_delete_is_not_registered():
    help_result = invoke("workspace", "--help")
    delete_result = invoke("workspace", "delete", "alice/default")

    assert help_result.exit_code == 0, help_result.output
    assert "delete " not in help_result.output
    assert delete_result.exit_code != 0
    assert "No such command" in delete_result.output


def test_workspace_list_table(platform_client):
    _, client = platform_client
    client.list_workspaces.return_value = [PERSONAL, ORG]

    result = invoke("workspace", "list")

    assert result.exit_code == 0, result.output
    assert "alice/default" in result.output
    assert "acme/research" in result.output
    assert "organization" in result.output
    assert "7" in result.output
    assert "22" in result.output


def test_workspace_list_json_is_stable_plain_json(platform_client):
    _, client = platform_client
    client.list_workspaces.return_value = [PERSONAL, ORG]

    result = invoke("workspace", "list", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {
            "path": "alice/default",
            "account_type": "user",
            "account_id": 1,
            "environment_id": 11,
        },
        {
            "path": "acme/research",
            "account_type": "organization",
            "account_id": 7,
            "environment_id": 22,
        },
    ]


def test_workspace_show_without_path_uses_current_workspace(platform_client):
    _, client = platform_client
    client.current_workspace.return_value = PERSONAL

    result = invoke("workspace", "show", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["path"] == "alice/default"
    client.current_workspace.assert_called_once_with()
    client.list_workspaces.assert_not_called()


def test_workspace_show_path_resolves_without_switching(platform_client):
    _, client = platform_client
    client.get_workspace.return_value = ORG

    result = invoke("workspace", "show", "acme/research", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["environment_id"] == 22
    client.switch_workspace.assert_not_called()
    client.get_workspace.assert_called_once_with("acme/research")


def test_workspace_show_missing_path_fails(platform_client):
    _, client = platform_client
    client.get_workspace.side_effect = PlatformError(
        "No workspace found for path 'missing/dev'"
    )

    result = invoke("workspace", "show", "missing/dev")

    assert result.exit_code != 0
    assert "No workspace found" in result.output


def test_workspace_switch_reports_active_workspace(platform_client):
    _, client = platform_client
    client.switch_workspace.return_value = ORG

    result = invoke("workspace", "switch", "acme/research", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["path"] == "acme/research"
    client.switch_workspace.assert_called_once_with("acme/research")


def test_workspace_create_for_account(platform_client):
    _, client = platform_client
    client.create_workspace.return_value = ORG

    result = invoke(
        "workspace",
        "create",
        "research",
        "--account",
        "acme",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["path"] == "acme/research"
    client.create_workspace.assert_called_once_with("research", account="acme")


def test_org_list_table_and_json(platform_client):
    _, client = platform_client
    organizations = [{"id": 7, "name": "Acme Labs", "slug": "acme"}]
    client.list_organizations.return_value = organizations

    table_result = invoke("org", "list")
    json_result = invoke("org", "list", "--format", "json")

    assert table_result.exit_code == 0, table_result.output
    assert "Acme Labs" in table_result.output
    assert "acme" in table_result.output
    assert json.loads(json_result.output) == organizations


def test_org_show_create_and_invite(platform_client):
    _, client = platform_client
    client.get_organization.return_value = {
        "id": 7,
        "name": "Acme Labs",
        "slug": "acme",
    }
    client.create_organization.return_value = {
        "id": 8,
        "name": "New Org",
        "slug": "new-org",
    }
    client.invite_to_organization.return_value = {
        "email": "person@example.com",
        "role": "admin",
        "status": "invited",
    }

    show_result = invoke("org", "show", "7", "--format", "json")
    create_result = invoke(
        "org", "create", "New Org", "--slug", "new-org", "--format", "json"
    )
    invite_result = invoke(
        "org",
        "invite",
        "7",
        "person@example.com",
        "--role",
        "admin",
        "--format",
        "json",
    )

    assert json.loads(show_result.output)["slug"] == "acme"
    assert json.loads(create_result.output)["id"] == 8
    assert json.loads(invite_result.output)["status"] == "invited"
    client.get_organization.assert_called_once_with(7)
    client.create_organization.assert_called_once_with("New Org", "new-org")
    client.invite_to_organization.assert_called_once_with(
        7, "person@example.com", role="admin"
    )


def test_org_invite_rejects_unknown_role(platform_client):
    _, client = platform_client

    result = invoke("org", "invite", "7", "person@example.com", "--role", "owner")

    assert result.exit_code != 0
    assert "Invalid value for '--role'" in result.output
    client.invite_to_organization.assert_not_called()


def test_budget_show_displays_available_response_fields(platform_client):
    _, client = platform_client
    client.get_budget.return_value = {
        "total_budget": 100.0,
        "current_usage": 25.0,
        "remaining_budget": 75.0,
        "currency": "USD",
    }

    result = invoke("budget", "show")

    assert result.exit_code == 0, result.output
    assert "total budget" in result.output.lower()
    assert "100" in result.output
    assert "USD" in result.output


def test_budget_set_and_nonnegative_validation(platform_client):
    _, client = platform_client
    client.set_budget.return_value = {"workspace_budget": 42.5, "currency": "USD"}

    valid = invoke("budget", "set", "42.5", "--format", "json")
    invalid = invoke("budget", "set", "-0.01")

    assert valid.exit_code == 0, valid.output
    assert json.loads(valid.output)["workspace_budget"] == 42.5
    client.set_budget.assert_called_once_with(42.5)
    assert invalid.exit_code != 0
    assert "not in the range" in invalid.output


def test_budget_set_rejects_typo_option(platform_client):
    _, client = platform_client

    result = invoke("budget", "set", "10", "--formt", "json")

    assert result.exit_code != 0
    assert "unexpected extra argument" in result.output.lower()
    client.set_budget.assert_not_called()


def test_platform_error_has_useful_nonzero_exit(platform_client):
    _, client = platform_client
    client.list_workspaces.side_effect = PlatformError("Authentication required")

    result = invoke("workspace", "list")

    assert result.exit_code != 0
    assert "Authentication required" in result.output


def test_each_command_uses_platform_client_context_manager(platform_client):
    factory, client = platform_client
    client.current_workspace.return_value = PERSONAL

    result = invoke("workspace", "show")

    assert result.exit_code == 0, result.output
    factory.assert_called_once_with()
    factory.return_value.__enter__.assert_called_once_with()
    factory.return_value.__exit__.assert_called_once()


def test_apikey_group_and_commands_registered():
    result = invoke("--help")
    assert result.exit_code == 0, result.output
    assert "apikey create" in result.output
    assert "apikey delete" in result.output


def test_apikey_has_no_list_command(platform_client):
    help_result = invoke("apikey", "--help")
    list_result = invoke("apikey", "list")

    assert help_result.exit_code == 0, help_result.output
    assert "list" not in help_result.output
    assert list_result.exit_code != 0
    assert "No such command" in list_result.output


def test_apikey_create_personal_default_prints_token_once(platform_client):
    _, client = platform_client
    client.create_api_key.return_value = {"token": "knox-secret-value"}

    result = invoke("apikey", "create")

    assert result.exit_code == 0, result.output
    assert "knox-secret-value" in result.output
    assert "once" in result.output.lower()
    client.create_api_key.assert_called_once_with(account=None)


def test_apikey_create_with_account_selects_owner(platform_client):
    _, client = platform_client
    client.create_api_key.return_value = {"token": "knox-secret-value"}

    result = invoke("apikey", "create", "--account", "acme")

    assert result.exit_code == 0, result.output
    client.create_api_key.assert_called_once_with(account="acme")


def test_apikey_create_json_outputs_raw_response(platform_client):
    _, client = platform_client
    payload = {"token": "knox-secret-value"}
    client.create_api_key.return_value = payload

    result = invoke("apikey", "create", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == payload


def test_apikey_create_api_error_is_click_error(platform_client):
    _, client = platform_client
    client.create_api_key.side_effect = PlatformError("Token creation failed")

    result = invoke("apikey", "create")

    assert result.exit_code != 0
    assert "Token creation failed" in result.output


def test_apikey_delete_requires_confirmation_and_can_abort(platform_client):
    _, client = platform_client

    result = CliRunner().invoke(cli, ["apikey", "delete", "knox-secret-value"], input="n\n")

    assert result.exit_code != 0
    client.delete_api_key.assert_not_called()


def test_apikey_delete_with_yes_skips_prompt_and_hides_token(platform_client):
    _, client = platform_client
    client.delete_api_key.return_value = None

    result = invoke("apikey", "delete", "knox-secret-value", "--yes")

    assert result.exit_code == 0, result.output
    assert "knox-secret-value" not in result.output
    client.delete_api_key.assert_called_once_with("knox-secret-value")


def test_apikey_delete_confirmed_via_prompt(platform_client):
    _, client = platform_client
    client.delete_api_key.return_value = None

    result = CliRunner().invoke(
        cli, ["apikey", "delete", "knoxtok01"], input="y\n"
    )

    assert result.exit_code == 0, result.output
    client.delete_api_key.assert_called_once_with("knoxtok01")


def test_apikey_delete_api_error_is_click_error(platform_client):
    _, client = platform_client
    client.delete_api_key.side_effect = PlatformError("not found")

    result = invoke("apikey", "delete", "knoxtok01", "--yes")

    assert result.exit_code != 0
    assert "not found" in result.output


def _usage_payload():
    return {
        "account_type": "organization",
        "account_id": 20,
        "institute_id": 501,
        "selected_year": 2025,
        "selected_month": 6,
        "current_year": 2026,
        "current_month": 7,
        "env_list": [{"id": 200, "slug": "prod"}],
        "filter_env_id": 200,
        "current_usage_amount": 12.5,
        "environment_usage_amount": 3.0,
        "environment_label": "prod",
        "model_charges": [
            {"model_name": "esm2-8m", "total_biolm_charge": 12.5},
            {"model_name": None, "total_biolm_charge": 0.25},
        ],
    }


def test_usage_group_and_show_command_registered():
    result = invoke("--help")

    assert result.exit_code == 0, result.output
    assert "usage show" in result.output


def test_usage_show_default_delegates_and_renders_summary(platform_client):
    _, client = platform_client
    client.get_usage_summary.return_value = _usage_payload()

    result = invoke("usage", "show")

    assert result.exit_code == 0, result.output
    client.get_usage_summary.assert_called_once_with(
        year=None,
        month=None,
        environment_id=None,
        account=None,
    )
    assert "organization" in result.output
    assert "2025-06" in result.output
    assert "12.5" in result.output
    assert "esm2-8m" in result.output
    assert "0.25" in result.output
    assert "—" in result.output


def test_usage_show_filters_and_json_passthrough(platform_client):
    _, client = platform_client
    payload = _usage_payload()
    client.get_usage_summary.return_value = payload

    result = invoke(
        "usage",
        "show",
        "--year",
        "2025",
        "--month",
        "6",
        "--environment-id",
        "200",
        "--account",
        "acme",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == payload
    client.get_usage_summary.assert_called_once_with(
        year=2025,
        month=6,
        environment_id=200,
        account="acme",
    )


@pytest.mark.parametrize(
    "args",
    [
        ("--year", "0"),
        ("--month", "0"),
        ("--month", "13"),
        ("--environment-id", "0"),
    ],
)
def test_usage_show_rejects_invalid_ranges_before_request(platform_client, args):
    _, client = platform_client

    result = invoke("usage", "show", *args)

    assert result.exit_code != 0
    client.get_usage_summary.assert_not_called()


def test_usage_show_empty_model_charges(platform_client):
    _, client = platform_client
    payload = _usage_payload()
    payload["model_charges"] = []
    client.get_usage_summary.return_value = payload

    result = invoke("usage", "show")

    assert result.exit_code == 0, result.output
    assert "No model charges" in result.output


def test_usage_show_does_not_imply_rejected_environment_filter(platform_client):
    _, client = platform_client
    payload = _usage_payload()
    payload["filter_env_id"] = None
    payload["environment_label"] = "prod"
    client.get_usage_summary.return_value = payload

    result = invoke("usage", "show")

    assert result.exit_code == 0, result.output
    assert "Environment filter" in result.output
    assert "prod" not in result.output


def test_usage_show_api_error_is_click_error(platform_client):
    _, client = platform_client
    client.get_usage_summary.side_effect = PlatformError("Usage unavailable")

    result = invoke("usage", "show")

    assert result.exit_code != 0
    assert "Usage unavailable" in result.output
