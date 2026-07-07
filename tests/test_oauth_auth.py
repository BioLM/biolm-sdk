"""Tests for OAuth authentication functionality."""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from biolm.core.auth import (
    _b64url,
    _gen_pkce_pair,
    are_credentials_valid,
    oauth_login,
    parse_credentials_file,
    save_access_refresh_token,
)
from biolm.core.const import ACCESS_TOK_PATH, OAUTH_REDIRECT_URI


class TestPKCEHelpers:
    """Test PKCE helper functions."""

    def test_b64url_encoding(self):
        """Test base64 URL-safe encoding without padding."""
        # Test with known input
        data = b"test data"
        encoded = _b64url(data)
        assert isinstance(encoded, str)
        assert "=" not in encoded  # No padding
        assert "/" not in encoded  # URL-safe
        assert "+" not in encoded  # URL-safe

    def test_gen_pkce_pair(self):
        """Test PKCE pair generation."""
        verifier, challenge = _gen_pkce_pair()
        
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 0
        assert len(challenge) > 0
        
        # Challenge should be SHA256 hash of verifier
        import base64
        import hashlib
        
        verifier_bytes = base64.urlsafe_b64decode(verifier + "==")
        expected_challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        assert challenge == expected_challenge

    def test_pkce_pair_uniqueness(self):
        """Test that PKCE pairs are unique."""
        pairs = [_gen_pkce_pair() for _ in range(10)]
        verifiers = [p[0] for p in pairs]
        challenges = [p[1] for p in pairs]
        
        # All verifiers should be unique
        assert len(set(verifiers)) == 10
        # All challenges should be unique
        assert len(set(challenges)) == 10


class TestCredentialPersistence:
    """Test credential file persistence and parsing."""

    def test_save_and_parse_credentials(self):
        """Test saving and parsing OAuth credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_path = Path(tmpdir) / "credentials"
            
            creds = {
                "access": "test_access_token",
                "refresh": "test_refresh_token",
                "expires_in": 3600,
                "expires_at": int(time.time()) + 3600,
                "token_url": "https://biolm.ai/o/token/",
                "client_id": "test_client_id",
            }
            
            # Save credentials
            with patch("biolmai.core.auth.ACCESS_TOK_PATH", str(cred_path)):
                with patch("biolmai.core.auth.USER_BIOLM_DIR", tmpdir):
                    with patch("biolmai.core.auth.validate_user_auth"):
                        save_access_refresh_token(creds)
            
            # Verify file exists
            assert cred_path.exists()
            
            # Parse credentials
            parsed = parse_credentials_file(str(cred_path))
            assert parsed is not None
            assert parsed["access"] == "test_access_token"
            assert parsed["refresh"] == "test_refresh_token"

    def test_parse_credentials_with_oauth_fields(self):
        """Test parsing credentials with OAuth-specific fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_path = Path(tmpdir) / "credentials"
            
            creds = {
                "access": "access_token",
                "refresh": "refresh_token",
                "expires_at": 1234567890,
                "token_url": "https://example.com/token",
                "client_id": "client123",
            }
            
            with open(cred_path, "w") as f:
                json.dump(creds, f)
            
            # Parse should work
            parsed = parse_credentials_file(str(cred_path))
            assert parsed is not None
            assert parsed["access"] == "access_token"
            assert parsed["refresh"] == "refresh_token"


class TestOAuthLogin:
    """Test OAuth login functionality with mocked HTTP requests."""

    @patch("biolmai.core.auth.webbrowser.open")
    @patch("biolmai.core.auth._start_local_callback_server")
    @patch("biolmai.core.auth.requests.post")
    @patch("biolmai.core.auth.save_access_refresh_token")
    def test_browser_login_pkce_success(
        self, mock_save, mock_post, mock_server, mock_browser
    ):
        """Test successful browser/PKCE login flow."""
        # Mock callback server
        mock_queue = MagicMock()
        mock_queue.get.side_effect = ["auth_code_123", None]  # First call returns code, second is sentinel
        mock_server.return_value = mock_queue
        
        # Mock token exchange response
        mock_token_resp = Mock()
        mock_token_resp.status_code = 200  # Set status_code to avoid error message
        mock_token_resp.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
        }
        mock_token_resp.raise_for_status = Mock()
        mock_post.return_value = mock_token_resp
        
        # Mock save function
        mock_save.return_value = None
        
        with patch("biolmai.core.auth.OAUTH_REDIRECT_URI", "http://127.0.0.1:8765/callback"):
            result = oauth_login(
                client_id="test_client",
                auth_url="https://example.com/authorize",
                token_url="https://example.com/token",
            )
        
        # Verify token exchange was called
        assert mock_post.called
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://example.com/token"
        
        # Verify credentials were saved
        assert mock_save.called
        saved_creds = mock_save.call_args[0][0]
        assert saved_creds["access"] == "new_access_token"
        assert saved_creds["refresh"] == "new_refresh_token"
        assert "expires_at" in saved_creds
        assert saved_creds["client_id"] == "test_client"


    def test_oauth_login_missing_client_id(self):
        """Test that oauth_login raises error when client_id is missing."""
        with patch("biolmai.core.const.BIOLMAI_PUBLIC_CLIENT_ID", ""):
            with pytest.raises(ValueError, match="OAuth client ID required"):
                oauth_login(client_id=None)


class TestCredentialValidation:
    """Test credential validation functionality."""

    def test_are_credentials_valid_no_file(self):
        """Test that are_credentials_valid returns False when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("biolmai.core.auth.ACCESS_TOK_PATH", str(Path(tmpdir) / "nonexistent")):
                assert are_credentials_valid() is False

    def test_are_credentials_valid_invalid_file(self):
        """Test that are_credentials_valid returns False for invalid credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_path = Path(tmpdir) / "credentials"
            # Write invalid JSON
            with open(cred_path, "w") as f:
                f.write("invalid json")
            
            with patch("biolmai.core.auth.ACCESS_TOK_PATH", str(cred_path)):
                assert are_credentials_valid() is False

    def test_are_credentials_valid_missing_tokens(self):
        """Test that are_credentials_valid returns False when tokens are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_path = Path(tmpdir) / "credentials"
            creds = {"access": "token", "refresh": None}  # Missing refresh
            
            with open(cred_path, "w") as f:
                json.dump(creds, f)
            
            with patch("biolmai.core.auth.ACCESS_TOK_PATH", str(cred_path)):
                assert are_credentials_valid() is False

    @patch("biolmai.core.auth.validate_user_auth")
    def test_are_credentials_valid_success(self, mock_validate):
        """Test that are_credentials_valid returns True for valid credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_path = Path(tmpdir) / "credentials"
            creds = {
                "access": "valid_access_token",
                "refresh": "valid_refresh_token",
            }
            
            with open(cred_path, "w") as f:
                json.dump(creds, f)
            
            # Mock successful validation
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"user": "test"}  # No "code" key
            mock_validate.return_value = mock_resp
            
            with patch("biolmai.core.auth.ACCESS_TOK_PATH", str(cred_path)):
                assert are_credentials_valid() is True
                mock_validate.assert_called_once_with(
                    access="valid_access_token",
                    refresh="valid_refresh_token"
                )

    @patch("biolmai.core.auth.validate_user_auth")
    def test_are_credentials_valid_failed_validation(self, mock_validate):
        """Test that are_credentials_valid returns False when validation fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_path = Path(tmpdir) / "credentials"
            creds = {
                "access": "invalid_token",
                "refresh": "invalid_refresh",
            }
            
            with open(cred_path, "w") as f:
                json.dump(creds, f)
            
            # Mock failed validation
            mock_resp = Mock()
            mock_resp.status_code = 401
            mock_validate.return_value = mock_resp
            
            with patch("biolmai.core.auth.ACCESS_TOK_PATH", str(cred_path)):
                assert are_credentials_valid() is False


class TestTokenRefresh:
    """Test token refresh functionality (for seqflow_auth)."""

    def test_refresh_without_secret(self):
        """Test that refresh works without client_secret (public client)."""
        from biolm.core.seqflow_auth import BiolmaiRequestHeaderProvider
        
        provider = BiolmaiRequestHeaderProvider()
        
        creds = {
            "access": "old_token",
            "refresh": "refresh_token_123",
            "expires_at": time.time() - 100,  # Expired
            "token_url": "https://example.com/token",
            "client_id": "public_client",
            # No client_secret
        }
        
        mock_response = Mock()
        mock_response.json.return_value = {"access_token": "new_token"}
        mock_response.raise_for_status = Mock()
        
        with patch("httpx.post", return_value=mock_response):
            with patch.object(provider, "_load_credentials", return_value=creds):
                with patch("builtins.open", create=True):
                    new_token = provider._refresh_token("refresh_token_123", creds)
        
        assert new_token == "new_token"

