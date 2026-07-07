"""
MLflow RequestHeaderProvider that uses biolmai credentials for authentication.

This provider reads OAuth tokens from ~/.biolmai/credentials and adds them
as Bearer tokens to MLflow requests.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

try:
    from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
except ImportError:
    # Fallback for older MLflow versions
    try:
        from mlflow.tracking.request_header.abstract import RequestHeaderProvider
    except ImportError:
        RequestHeaderProvider = object  # Type stub for IDE


class BiolmaiRequestHeaderProvider(RequestHeaderProvider):
    """
    MLflow RequestHeaderProvider that uses biolmai credentials.
    
    Reads OAuth tokens from ~/.biolmai/credentials (JSON format:
    {"access": "...", "refresh": "..."}) and adds Authorization header
    with Bearer token to all MLflow requests.
    """
    
    def __init__(self):
        self.credentials_path = Path.home() / ".biolmai" / "credentials"
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0
        self._token_url: Optional[str] = None
        self._client_id: Optional[str] = None
        self._client_secret: Optional[str] = None
    
    def in_context(self) -> bool:
        """
        Return True if provider should be used.
        
        Checks if:
        1. biolmai package is installed
        2. Credentials file exists
        3. Credentials file contains access token
        """
        try:
            import biolmai  # noqa: F401
        except ImportError:
            return False
        
        if not self.credentials_path.exists():
            return False
        
        try:
            creds = self._load_credentials()
            return bool(creds.get("access"))
        except Exception:
            return False
    
    def request_headers(self) -> dict:
        """
        Return headers to add to MLflow requests.
        
        Returns Authorization header with Bearer token from biolmai credentials.
        """
        token = self._get_valid_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def _load_credentials(self) -> dict:
        """Load credentials from ~/.biolmai/credentials file."""
        if not self.credentials_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
        
        with open(self.credentials_path, "r") as f:
            return json.load(f)
    
    def _get_valid_token(self) -> Optional[str]:
        """
        Get valid access token, refreshing if needed.
        
        Returns:
            Access token string, or None if unavailable
        """
        # Check cache first
        if self._cached_token and time.time() < self._token_expires_at:
            return self._cached_token
        
        try:
            creds = self._load_credentials()
            access_token = creds.get("access")
            refresh_token = creds.get("refresh")
            expires_at = creds.get("expires_at")  # Optional: if biolmai stores expiration
            
            # If token is expired or about to expire (within 60 seconds), try to refresh
            if refresh_token and expires_at and time.time() >= (expires_at - 60):
                # Try to refresh token
                new_token = self._refresh_token(refresh_token, creds)
                if new_token:
                    access_token = new_token
                    # Update credentials file with new token
                    creds["access"] = access_token
                    if "expires_at" in creds:
                        # Update expiration if provided
                        creds["expires_at"] = time.time() + creds.get("expires_in", 3600)
                    with open(self.credentials_path, "w") as f:
                        json.dump(creds, f, indent=2)
            
            # Cache token (assume 1 hour expiration if not specified)
            self._cached_token = access_token
            self._token_expires_at = expires_at if expires_at else (time.time() + 3600)
            
            return access_token
        except Exception as e:
            # Log error but don't raise - allow MLflow to work without auth if needed
            import warnings
            warnings.warn(f"Could not load biolmai credentials: {e}")
            return None
    
    
    def _refresh_token(self, refresh_token: str, creds: dict) -> Optional[str]:
        """
        Refresh access token using refresh token.
        
        This method attempts to refresh the token using the OAuth provider.
        Supports both public clients (PKCE, no secret) and confidential clients (with secret).
        """
        # Try to get token URL and client credentials from credentials file first,
        # then from biolm.const constants (which respect BIOLMAI_BASE_DOMAIN), then environment
        try:
            from biolm.core.const import (
                BIOLMAI_OAUTH_CLIENT_SECRET,
                BIOLMAI_PUBLIC_CLIENT_ID,
                OAUTH_TOKEN_URL,
            )
        except ImportError:
            # Fallback if const module not available
            BIOLMAI_OAUTH_CLIENT_SECRET = os.environ.get("BIOLMAI_OAUTH_CLIENT_SECRET") or os.environ.get("CLIENT_SECRET", "")
            BIOLMAI_PUBLIC_CLIENT_ID = os.environ.get("BIOLMAI_OAUTH_CLIENT_ID", "")
            # Construct OAuth token URL from BIOLMAI_BASE_DOMAIN if available, otherwise default
            base_domain = os.environ.get("BIOLMAI_BASE_DOMAIN", "https://biolm.ai")
            if not base_domain.startswith(("http://", "https://")):
                base_domain = f"http://{base_domain}"
            OAUTH_TOKEN_URL = os.environ.get("OAUTH_TOKEN_URL") or f"{base_domain}/o/token/"
        
        token_url = (
            creds.get("token_url") or
            OAUTH_TOKEN_URL
        )
        
        client_id = (
            creds.get("client_id") or
            BIOLMAI_PUBLIC_CLIENT_ID or
            os.environ.get("BIOLMAI_OAUTH_CLIENT_ID")
        )
        
        # Check both CLIENT_SECRET and BIOLMAI_OAUTH_CLIENT_SECRET for compatibility
        client_secret = (
            creds.get("client_secret") or
            BIOLMAI_OAUTH_CLIENT_SECRET or
            os.environ.get("BIOLMAI_OAUTH_CLIENT_SECRET") or
            os.environ.get("CLIENT_SECRET")
        )
        
        if not token_url or not client_id:
            # Can't refresh without token URL and client ID
            return None
        
        try:
            import httpx
            
            # Build refresh request data
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            }
            
            # Only include client_secret if provided (for confidential clients)
            # Public clients (PKCE) don't need/use client_secret
            if client_secret:
                data["client_secret"] = client_secret
            
            response = httpx.post(
                token_url,
                data=data,
                timeout=30.0
            )
            response.raise_for_status()
            token_data = response.json()
            return token_data.get("access_token") or token_data.get("access")
        except Exception as e:
            import warnings
            warnings.warn(f"Token refresh failed: {e}")
            return None


class BiolmRequestHeaderProvider(BiolmaiRequestHeaderProvider):
    """Canonical MLflow header provider name after biolm rename."""
