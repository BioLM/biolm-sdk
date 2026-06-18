"""Integration tests for different authentication methods.

These tests make real API requests using:
- BIOLMAI_TOKEN (API token auth) - standard CI path
- Access/refresh tokens from credentials file (Cookie auth) - obtained via BIOLM_USER/BIOLM_PASSWORD
  at /api/auth/token/ (username+password)

Note: There is no endpoint that exchanges BIOLMAI_TOKEN for access/refresh tokens. They are
separate auth mechanisms. Access/refresh tokens come from username/password or OAuth only.

OAuth (browser PKCE flow) cannot be automated in CI since it requires user interaction.
Real OAuth token testing must be done manually. Unit tests with mocks exist in test_oauth_auth.py.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from biolm.core.auth import generate_access_token


def _has_credential_env() -> bool:
    """Check if BIOLM_USER and BIOLM_PASSWORD are set (needed for access/refresh token tests)."""
    return bool(os.environ.get("BIOLM_USER") and os.environ.get("BIOLM_PASSWORD"))


@pytest.mark.asyncio
async def test_api_request_with_access_refresh_tokens():
    """Test API request using access/refresh tokens from credentials file.

    Obtains tokens via username/password (BIOLM_USER, BIOLM_PASSWORD), writes them
    to a temp credentials file, and verifies the client can make an API request
    using Cookie-based auth instead of BIOLMAI_TOKEN.

    Skips when BIOLM_USER or BIOLM_PASSWORD are not set (e.g. local dev without secrets).
    """
    if not _has_credential_env():
        pytest.skip(
            "BIOLM_USER and BIOLM_PASSWORD required for access/refresh token integration test. "
            "Set these in CI secrets or skip this test locally."
        )

    from biolm.core.http import BioLMApiClient

    # 1. Obtain access/refresh tokens via username/password
    token_response = generate_access_token(
        os.environ["BIOLM_USER"],
        os.environ["BIOLM_PASSWORD"],
    )
    if not token_response or "access" not in token_response or "refresh" not in token_response:
        pytest.skip(
            "Could not obtain access/refresh tokens. Token endpoint may have returned an error."
        )

    # 2. Write credentials to temp file (mirrors ~/.biolmai/credentials format)
    with tempfile.TemporaryDirectory() as tmpdir:
        cred_path = Path(tmpdir) / "credentials"
        creds = {
            "access": token_response["access"],
            "refresh": token_response["refresh"],
        }
        cred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cred_path, "w") as f:
            json.dump(creds, f)

        # 3. Unset BIOLMAI_TOKEN and BIOLM_TOKEN so client falls back to credentials file
        old_biomai_token = os.environ.pop("BIOLMAI_TOKEN", None)
        old_biom_token = os.environ.pop("BIOLM_TOKEN", None)
        try:
            # 4. Patch client to use our temp credentials path (patch where it's used: http)
            with patch("biolmai.core.http.ACCESS_TOK_PATH", str(cred_path)):
                client = BioLMApiClient(
                    "nanobert",
                    raise_httpx=False,
                    unwrap_single=False,
                    retry_error_batches=False,
                )
                result = await client.encode(
                    items=[{"sequence": "EVQLVESGGG"}],
                    stop_on_error=False,
                )
                await client.shutdown()

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)
            # If the API responded with a billing-cap or quota error, auth
            # actually succeeded — the request reached the model — and the
            # test goal is unverifiable through no fault of the SDK.  Skip
            # so a stuck account billing state doesn't permanently red CI.
            err_payload = result[0].get("error") if isinstance(result[0], dict) else None
            err_code = result[0].get("code") if isinstance(result[0], dict) else None
            err_status = result[0].get("status_code") if isinstance(result[0], dict) else None
            # Codes match modal_api/views.py:405-422 in biolm_web (the
            # production backend).  Adding `monthly_cap_reached` /
            # `quota_exceeded` / `rate_limited` for forward compatibility.
            if err_status in (402, 429) or err_code in (
                "lifetime_cap_reached",
                "budget_limit_reached",
                "payment_past_due",
                "monthly_cap_reached",
                "quota_exceeded",
                "rate_limited",
            ):
                pytest.skip(
                    f"BioLM account at billing/quota cap "
                    f"(status={err_status}, code={err_code}, error={err_payload!r}). "
                    "Auth succeeded — skipping content check."
                )
            assert "error" not in result[0]
        finally:
            if old_biomai_token is not None:
                os.environ["BIOLMAI_TOKEN"] = old_biomai_token
            if old_biom_token is not None:
                os.environ["BIOLM_TOKEN"] = old_biom_token


def test_credentials_file_auth_headers():
    """Test that CredentialsProvider builds Cookie headers from credentials file.

    When BIOLMAI_TOKEN and BIOLM_TOKEN are unset and ~/.biolmai/credentials exists, the client
    uses Cookie auth. This verifies the header-building logic. OAuth and legacy
    tokens both use the same Cookie format in the credentials file.
    """
    creds = {
        "access": "mock_oauth_access_token",
        "refresh": "mock_oauth_refresh_token",
        "token_url": "https://biolm.ai/o/token/",
        "client_id": "test_client",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cred_path = Path(tmpdir) / "credentials"
        with open(cred_path, "w") as f:
            json.dump(creds, f)

        old_biomai_token = os.environ.pop("BIOLMAI_TOKEN", None)
        old_biom_token = os.environ.pop("BIOLM_TOKEN", None)
        try:
            with patch("biolmai.core.http.ACCESS_TOK_PATH", str(cred_path)):
                from biolm.core.http import CredentialsProvider

                headers = CredentialsProvider.get_auth_headers(api_key=None)
                # Client uses Cookie for credentials file (legacy and OAuth alike)
                assert "Cookie" in headers
                assert "access=mock_oauth_access_token" in headers["Cookie"]
                assert "refresh=mock_oauth_refresh_token" in headers["Cookie"]
        finally:
            if old_biomai_token is not None:
                os.environ["BIOLMAI_TOKEN"] = old_biomai_token
            if old_biom_token is not None:
                os.environ["BIOLM_TOKEN"] = old_biom_token
