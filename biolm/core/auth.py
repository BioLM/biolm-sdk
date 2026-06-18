import ast
import base64
import hashlib
import http.server
import json
import os
import pprint
import queue
import secrets
import stat
import threading
import time
import urllib.parse
import webbrowser
from typing import Optional, Tuple

import click
import requests

from biolm.core.const import (
    ACCESS_TOK_PATH,
    BIOLMAI_BASE_DOMAIN,
    GEN_TOKEN_URL,
    OAUTH_AUTHORIZE_URL,
    OAUTH_INTROSPECT_URL,
    OAUTH_REDIRECT_URI,
    OAUTH_TOKEN_URL,
    USER_BIOLM_DIR,
)


def _is_debug() -> bool:
    """Check if DEBUG environment variable is enabled.
    
    Returns True if DEBUG is set to '1', 'true', 'True', etc.
    Returns False if DEBUG is unset, '0', 'false', 'False', etc.
    """
    return os.environ.get("DEBUG", "").upper().strip() in ("TRUE", "1")


def parse_credentials_file(file_path):
    """Parse credentials file, handling JSON, Python dict syntax, and mixed types.
    
    Returns a dict with all credential fields preserved, or None if parsing fails.
    Uses ast.literal_eval() which is safe and only evaluates Python literals.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # Try JSON first
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fall back to safe Python literal evaluation for dict syntax like {access: 123, refresh: 456}
            # ast.literal_eval() is safe - it only evaluates literals, no code execution
            try:
                data = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                return None
        
        # Ensure we have a dictionary
        if not isinstance(data, dict):
            return None
        
        # Preserve all fields, but ensure access and refresh are strings for backward compatibility
        result = data.copy()
        if "access" in result and result["access"] is not None:
            result["access"] = str(result["access"])
        if "refresh" in result and result["refresh"] is not None:
            result["refresh"] = str(result["refresh"])
            
        return result
        
    except Exception:
        return None


def validate_user_auth(api_token=None, access=None, refresh=None):
    """Validates an API token, to be used as 'Authorization: Token 1235abc'
    authentication method."""
    url = f"{BIOLMAI_BASE_DOMAIN}/api/v1/auth/login-check/"
    if api_token is not None:
        headers = {"Authorization": f"Token {api_token}"}
    else:
        headers = {
            "Cookie": f"access={access};refresh={refresh}",
            "Content-Type": "application/json",
        }
    try:
        r = requests.post(url=url, headers=headers)
        json_response = r.json()
        pretty_json = pprint.pformat(json_response, indent=2)
        click.echo(pretty_json)
    except Exception:
        click.echo("Token validation failed!\n")
        raise
    else:
        return r


def refresh_access_token(refresh):
    """Attempt to refresh temporary user access token, by using their refresh
    token, which has a longer TTL."""
    url = f"{BIOLMAI_BASE_DOMAIN}/api/auth/token/refresh/"
    headers = {"Cookie": f"refresh={refresh}", "Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers)
    json_response = r.json()
    if r.status_code != 200 or (r.status_code == 200 and "code" in r.json()):
        pretty_json = pprint.pformat(json_response, indent=2)
        click.echo(pretty_json)
        click.echo(
            "Token refresh failed! Please login by " "running `biolmai login`.\n"
        )
        return False
    else:
        access_refresh_dict = {"access": json_response["access"], "refresh": refresh}
        save_access_refresh_token(access_refresh_dict)
        return True


def get_auth_status():
    environ_token = os.environ.get("BIOLMAI_TOKEN") or os.environ.get("BIOLM_TOKEN")
    if environ_token:
        msg = "Environment variable BIOLMAI_TOKEN/BIOLM_TOKEN detected. Validating token..."
        click.echo(msg)
        validate_user_auth(api_token=environ_token)
    elif os.path.exists(ACCESS_TOK_PATH):
        msg = f"Credentials file found {ACCESS_TOK_PATH}. Validating token..."
        click.echo(msg)
        access_refresh_dict = parse_credentials_file(ACCESS_TOK_PATH)
        if access_refresh_dict is None:
            click.echo(f"Error reading credentials file {ACCESS_TOK_PATH}.")
            click.echo("The file may be corrupted or contain invalid data.")
            click.echo("Please login again by running `biolmai login`.")
            return
        access = access_refresh_dict.get("access")
        refresh = access_refresh_dict.get("refresh")
        
        # Check if these are OAuth tokens
        is_oauth = bool(access_refresh_dict.get("token_url") or access_refresh_dict.get("client_id"))
        
        if is_oauth:
            # Use OAuth introspection for confidential clients, expiration check for public clients
            # Note: django-oauth-toolkit introspection requires client authentication,
            # so it doesn't work for public clients (no client_secret)
            from biolm.core.const import BIOLMAI_OAUTH_CLIENT_SECRET
            client_id = access_refresh_dict.get("client_id")
            # Try credentials file first, then environment variable
            client_secret = access_refresh_dict.get("client_secret") or BIOLMAI_OAUTH_CLIENT_SECRET
            
            if _is_debug():
                click.echo(f"DEBUG: is_oauth=True, token_url={access_refresh_dict.get('token_url')}, client_id={client_id}", err=True)
                click.echo(f"DEBUG: Using client_secret: {'***' if client_secret else 'None (public client)'}", err=True)
            
            # For public clients, introspection doesn't work (requires client authentication)
            # So we validate by making an API call to a user info endpoint
            if not client_secret:
                expires_at = access_refresh_dict.get("expires_at")
                if expires_at:
                    import time
                    current_time = time.time()
                    if current_time >= expires_at:
                        click.echo("OAuth token has expired.")
                        click.echo("Please login again by running `biolmai login`.")
                        return
                    # Token not expired, validate via API call
                    if _is_debug():
                        click.echo(f"DEBUG: Public client - token not expired (expires at {expires_at}, current {current_time})", err=True)
                        click.echo("DEBUG: Validating token via user info endpoint (introspection not available for public clients)", err=True)
                
                # Validate token by calling user info endpoint
                is_valid = _validate_oauth_token_via_api(access)
                if is_valid:
                    click.echo("OAuth token is valid.")
                else:
                    click.echo("OAuth token validation failed. Token may be expired or invalid.")
                    click.echo("Please login again by running `biolmai login`.")
            else:
                # Confidential client - use introspection (requires client_secret)
                is_valid = _validate_oauth_token(access, client_id, client_secret)
                if is_valid:
                    click.echo("OAuth token is valid.")
                else:
                    click.echo("OAuth token validation failed. Token may be expired.")
                    click.echo("Please login again by running `biolmai login`.")
        else:
            # Legacy token validation
            resp = validate_user_auth(access=access, refresh=refresh)
            if resp.status_code != 200 or (
                resp.status_code == 200 and "code" in resp.json()
            ):
                click.echo("Access token validation failed. Attempting to refresh token...")
                # Attempt to use the 'refresh' token to get a new 'access' token
                if not refresh_access_token(refresh):
                    click.echo("Unexpected refresh token error.")
                else:
                    click.echo("Access token refresh was successful.")
    else:
        msg = (
            f"No {BIOLMAI_BASE_DOMAIN} credentials found. Please "
            f"set the environment variable BIOLMAI_TOKEN (or BIOLM_TOKEN) to a token from "
            f"{GEN_TOKEN_URL}, or login by running `biolmai login`."
        )
        click.echo(msg)


def generate_access_token(uname, password):
    """Generate a TTL-expiry access and refresh token, to be used as
    'Cookie: acccess=; refresh=;" headers, or the access token only as a
    'Authorization: Bearer 1235abc' token.

    The refresh token will expire in hours or days, while the access token
    will have a shorter TTL, more like hours. Meaning, this method will
    require periodically re-logging in, due to the token expiration time. For a
    more permanent auth method for the API, use an API token by setting the
    BIOLMAI_TOKEN environment variable.
    """
    url = f"{BIOLMAI_BASE_DOMAIN}/api/auth/token/"
    try:
        r = requests.post(url=url, data={"username": uname, "password": password})
        json_response = r.json()
    except Exception:
        click.echo("Login failed!\n")
        raise
    if r.status_code != 200:
        click.echo("Login failed!\n")
        resp_json = r.json()
        pretty_json = pprint.pformat(resp_json, indent=2)
        click.echo(pretty_json)
        return {}
    else:
        click.echo("Login succeeded!\n")
        return json_response


def save_access_refresh_token(access_refresh_dict):
    """Save temporary access and refresh tokens to user folder for future
    use."""
    os.makedirs(USER_BIOLM_DIR, exist_ok=True)
    # Save token
    with open(ACCESS_TOK_PATH, "w") as f:
        json.dump(access_refresh_dict, f)
    os.chmod(ACCESS_TOK_PATH, stat.S_IRUSR | stat.S_IWUSR)
    # Validate token and print user info (only for legacy tokens, not OAuth)
    access = access_refresh_dict.get("access")
    refresh = access_refresh_dict.get("refresh")
    
    # Skip validation for OAuth tokens (they use introspection, not legacy endpoint)
    is_oauth = bool(access_refresh_dict.get("token_url") or access_refresh_dict.get("client_id"))
    if not is_oauth and access and refresh:
        # Only validate legacy tokens using the old endpoint
        validate_user_auth(access=access, refresh=refresh)


def get_api_token():
    """Get a BioLM API token to use with future API requests.

    Copied from https://api.biolm.ai/#d7f87dfd-321f-45ae-99b6-eb203519ddeb.
    """
    url = f"{BIOLMAI_BASE_DOMAIN}/api/auth/token/"

    payload = json.dumps(
        {
            "username": os.environ.get("BIOLM_USER"),
            "password": os.environ.get("BIOLM_PASSWORD"),
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()

    return response_json


def get_user_auth_header():
    """Returns a dict with the appropriate Authorization header, either using
    an API token from BIOLMAI_TOKEN or BIOLM_TOKEN environment variable, or by
    reading the credentials file at ~/.biolmai/credentials next."""
    api_token = os.environ.get("BIOLMAI_TOKEN") or os.environ.get("BIOLM_TOKEN")
    if api_token:
        headers = {"Authorization": f"Token {api_token}"}
    elif os.path.exists(ACCESS_TOK_PATH):
        access_refresh_dict = parse_credentials_file(ACCESS_TOK_PATH)
        if access_refresh_dict is None:
            err = (
                f"Error reading credentials file {ACCESS_TOK_PATH}. "
                "The file may be corrupted or contain invalid data. "
                "Please run `biolmai login` to re-authenticate."
            )
            raise AssertionError(err)
        access = access_refresh_dict.get("access")
        refresh = access_refresh_dict.get("refresh")
        headers = {
            "Cookie": f"access={access};refresh={refresh}",
        }
    else:
        err = (
            f"No {BIOLMAI_BASE_DOMAIN} credentials found. Please run "
            "`biolmai status` to debug."
        )
        raise AssertionError(err)
    return headers


def _b64url(b: bytes) -> str:
    """Base64 URL-safe encoding without padding."""
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _gen_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code verifier and challenge pair."""
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _start_local_callback_server(expected_state: str, port: int = 8765, timeout: int = 180) -> queue.Queue:
    """Start a local HTTP server to receive OAuth callback.
    
    Args:
        expected_state: Expected OAuth state parameter for CSRF protection
        port: Port to bind to (default 8765, must match OAUTH_REDIRECT_URI)
        timeout: Timeout in seconds to wait for callback
    
    Returns:
        Queue that will receive the authorization code
    """
    received = queue.Queue()

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            
            params = dict(urllib.parse.parse_qsl(parsed.query))
            code = params.get("code")
            state = params.get("state")
            error = params.get("error")
            error_description = params.get("error_description")
            
            # Handle OAuth errors - check for error parameter FIRST, even if code is present
            # Some OAuth servers may return both error and code, but error takes precedence
            if error:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                error_msg = error_description or error
                error_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authorization Error - BioLM</title>
    <style>
        body {{
            font-family: system-ui, sans-serif;
            background: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }}
        .container {{
            background: white;
            border-radius: 0.75rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            padding: 3rem;
            max-width: 500px;
            text-align: center;
        }}
        h1 {{ color: #dc2626; margin-bottom: 1rem; }}
        p {{ color: #6b7280; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Authorization Error</h1>
        <p>{error_msg}</p>
        <p style="margin-top: 1rem; font-size: 0.875rem;">Please try logging in again from your terminal.</p>
    </div>
</body>
</html>"""
                self.wfile.write(error_html.encode())
                # Signal the main flow that an error occurred
                received.put(None)
                return
            
            if state != expected_state or not code:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                error_html = b"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authorization Error - BioLM</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            color: #1f2937;
        }
        .container {
            background: white;
            border-radius: 0.75rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            padding: 3rem;
            max-width: 500px;
            width: 100%;
            text-align: center;
        }
        .icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 1.5rem;
            background: #fee2e2;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
        }
        h1 {
            font-size: 1.875rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.75rem;
        }
        p {
            font-size: 1rem;
            color: #6b7280;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        .error-details {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1.5rem;
            font-size: 0.875rem;
            color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">&#9888;</div>
        <h1>Authorization Error</h1>
        <p>There was an error processing your authorization request. The state parameter is invalid or the authorization code is missing.</p>
        <div class="error-details">
            Please try logging in again from your terminal.
        </div>
    </div>
</body>
</html>"""
                self.wfile.write(error_html)
                # Signal the main flow that an error occurred
                received.put(None)
                return
            
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            success_html = b"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authorization Successful - BioLM</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            color: #1f2937;
        }
        .container {
            background: white;
            border-radius: 0.75rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
            padding: 3rem;
            max-width: 500px;
            width: 100%;
            text-align: center;
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3);
            color: white;
            font-weight: 600;
        }
        h1 {
            font-size: 1.875rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.75rem;
            letter-spacing: -0.025em;
        }
        p {
            font-size: 1rem;
            color: #6b7280;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        .success-message {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1.5rem;
            font-size: 0.875rem;
            color: #166534;
        }
        .footer {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e5e7eb;
            font-size: 0.875rem;
            color: #9ca3af;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">&#10003;</div>
        <h1>Authorization Successful</h1>
        <p>You have successfully authorized the BioLM CLI. You can now close this window and return to your terminal.</p>
        <div class="success-message">
            Your credentials have been saved securely.
        </div>
        <div class="footer">
            <p>BioLM AI</p>
        </div>
    </div>
</body>
</html>"""
            self.wfile.write(success_html)
            received.put(code)

        def log_message(self, *args, **kwargs):
            # Silence server logs
            pass

    try:
        httpd = http.server.HTTPServer(("localhost", port), CallbackHandler)
    except OSError as e:
        if "Address already in use" in str(e):
            raise RuntimeError(
                f"Port {port} is already in use. Please close any application using this port "
                f"or set a different redirect URI."
            ) from e
        raise
    
    # Start server in daemon thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    
    # Cleanup function to shutdown server after timeout
    def cleanup_after_timeout():
        time.sleep(timeout)
        httpd.shutdown()
    threading.Thread(target=cleanup_after_timeout, daemon=True).start()
    
    return received


def _browser_login_with_pkce(
    client_id: str, scope: str, auth_url: str, token_url: str, redirect_uri: str, client_secret: str = None
) -> dict:
    """Perform OAuth browser login with PKCE.
    
    Args:
        client_id: OAuth client ID
        scope: OAuth scope string
        auth_url: Authorization endpoint URL
        token_url: Token endpoint URL
        redirect_uri: Redirect URI (must match registered URI)
        client_secret: OAuth client secret (optional, for confidential clients)
    
    Returns:
        Token response dict with access_token, refresh_token, etc.
    """
    state = _b64url(secrets.token_bytes(24))
    verifier, challenge = _gen_pkce_pair()
    
    # Extract port from redirect_uri
    parsed_uri = urllib.parse.urlparse(redirect_uri)
    port = parsed_uri.port or 8765
    
    # Start callback server
    code_queue = _start_local_callback_server(state, port=port)
    
    # Build authorization URL
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    auth_url_with_params = f"{auth_url}?{urllib.parse.urlencode(params)}"
    
    # Open browser
    try:
        webbrowser.open(auth_url_with_params)
        click.echo(f"Opened browser for authorization. If it didn't open, visit:\n{auth_url_with_params}")
    except Exception as e:
        click.echo(f"Could not open browser automatically: {e}")
        click.echo(f"Please visit this URL in your browser:\n{auth_url_with_params}")
    
    # Wait for authorization code
    click.echo("Waiting for authorization...")
    click.echo(f"Make sure to complete the authorization in your browser and allow the redirect to {redirect_uri}")
    code = None
    start_time = time.time()
    timeout_seconds = 180  # Match default timeout
    
    while code is None:
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"OAuth login timed out after {timeout_seconds} seconds. "
                f"Make sure you completed the authorization in your browser and that the redirect to {redirect_uri} succeeded."
            )
        
        try:
            # Check queue with short timeout to allow checking overall timeout
            code = code_queue.get(timeout=1)
            # If we received None from the queue, it means an error occurred in the callback
            # (user denied, state mismatch, etc.) - break immediately to handle the error
            if code is None:
                break
        except queue.Empty:
            # Show progress every 10 seconds
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                remaining = timeout_seconds - int(elapsed)
                click.echo(f"Still waiting... ({remaining}s remaining)", err=True)
            continue
    
    # Check if code is None (indicates error from callback - user denied/cancelled or state mismatch)
    if code is None:
        raise RuntimeError("OAuth authorization was denied, cancelled, or failed validation.")
    
    if not code:
        raise RuntimeError("OAuth login timed out or was cancelled.")
    
    # Exchange code for tokens
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    
    # Include client_secret if provided (for confidential clients)
    # django-oauth-toolkit may require this even with PKCE for confidential clients
    if client_secret:
        data["client_secret"] = client_secret
    
    # Debug: Show what we're sending (without sensitive data)
    if os.environ.get("DEBUG"):
        click.echo(f"Exchanging code at: {token_url}", err=True)
        click.echo(f"Redirect URI: {redirect_uri}", err=True)
        click.echo(f"Client ID: {client_id}", err=True)
        click.echo(f"Code length: {len(code)}", err=True)
        click.echo(f"Verifier length: {len(verifier)}", err=True)
        click.echo(f"Client secret provided: {bool(client_secret)}", err=True)
    
    resp = requests.post(token_url, data=data, timeout=30)
    
    # Better error handling for debugging
    if resp.status_code != 200:
        try:
            error_detail = resp.json()
            click.echo(f"Token exchange failed (status {resp.status_code}): {error_detail}", err=True)
        except Exception:
            click.echo(f"Token exchange failed (status {resp.status_code}): {resp.text}", err=True)
        click.echo(f"Request was sent to: {token_url}", err=True)
        resp.raise_for_status()
    
    return resp.json()


def _validate_oauth_token_via_api(access_token: str) -> bool:
    """Validate OAuth token by making a lightweight API call to a user info endpoint.
    
    For public clients where introspection doesn't work, we can validate
    the token by making a simple API call to a user info endpoint that accepts OAuth Bearer tokens.
    
    Args:
        access_token: OAuth access token to validate
    
    Returns:
        True if token is valid, False otherwise
    """
    from biolm.core.const import BASE_API_URL, BIOLMAI_BASE_DOMAIN
    
    try:
        # Use /api/users/me/ endpoint to validate the token
        # This endpoint requires authentication and returns user info if token is valid
        url = f"{BIOLMAI_BASE_DOMAIN}/api/users/me/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        
        if os.environ.get("DEBUG"):
            click.echo(f"DEBUG: Validating OAuth token via API call to {url}", err=True)
        
        resp = requests.get(url, headers=headers, timeout=10)
        
        if os.environ.get("DEBUG"):
            click.echo(f"DEBUG: OAuth token validation response status: {resp.status_code}", err=True)
        
        if resp.status_code == 200:
            try:
                user_data = resp.json()
                if _is_debug():
                    import pprint
                    click.echo("DEBUG: Token valid, user info:", err=True)
                    click.echo(pprint.pformat(user_data, indent=2), err=True)
                else:
                    # Show selected user info fields in a clean format
                    fields_to_show = {
                        'username': user_data.get('username'),
                        'email': user_data.get('email'),
                        'first_name': user_data.get('first_name'),
                        'api_use_count': user_data.get('api_use_count'),
                        'in_trial': user_data.get('in_trial'),
                        'in_metering': user_data.get('in_metering'),
                        'in_fixed_rate': user_data.get('in_fixed_rate'),
                        'payment_past_due_or_canceled': user_data.get('payment_past_due_or_canceled'),
                        'month_dollars_billed': user_data.get('get_curr_month_dollars_billed'),
                        'usage_plan': user_data.get('usage_plan'),
                        'workspace_budget': user_data.get('workspace_budget'),
                    }
                    
                    click.echo("OAuth token is valid. User info:")
                    for key, value in fields_to_show.items():
                        if value is not None:
                            click.echo(f"  {key}: {value}")
            except Exception as e:
                if _is_debug():
                    click.echo(f"DEBUG: Failed to parse user info: {e}", err=True)
        
        # 200 means token is valid and user info was returned
        # 401/403 means token is invalid
        return resp.status_code == 200
    except Exception as e:
        if os.environ.get("DEBUG"):
            click.echo(f"DEBUG: OAuth token API validation exception: {e}", err=True)
        return False


def _validate_oauth_token(access_token: str, client_id: str = None, client_secret: str = None) -> bool:
    """Validate OAuth access token using introspection endpoint.
    
    Args:
        access_token: OAuth access token to validate
        client_id: OAuth client ID (required for django-oauth-toolkit)
        client_secret: OAuth client secret (optional, for public clients can be empty)
    
    Returns:
        True if token is valid, False otherwise
    """
    from biolm.core.const import BIOLMAI_OAUTH_CLIENT_SECRET, OAUTH_INTROSPECT_URL
    
    introspect_url = OAUTH_INTROSPECT_URL
    
    if not client_id:
        # Can't introspect without client_id for django-oauth-toolkit
        if os.environ.get("DEBUG"):
            click.echo("DEBUG: No client_id provided for introspection", err=True)
        return False
    
    if not access_token:
        if os.environ.get("DEBUG"):
            click.echo("DEBUG: No access_token provided for introspection", err=True)
        return False
    
    # Use client_secret from parameter or environment
    if client_secret is None:
        client_secret = BIOLMAI_OAUTH_CLIENT_SECRET
    
    try:
        if os.environ.get("DEBUG"):
            click.echo(f"DEBUG: Introspecting token at {introspect_url}", err=True)
        
        # For public clients (no secret), try different authentication methods
        # For confidential clients (with secret), use Basic Auth with client_id:client_secret
        methods_to_try = []
        
        if client_secret:
            # Confidential client: use Basic Auth with client_id:client_secret
            methods_to_try.append(("basic_auth_with_secret", client_secret))
        else:
            # Public client: try multiple methods
            # Method 1: Bearer token authentication (access token as Bearer)
            methods_to_try.append(("bearer_token", None))
            # Method 2: client_id in body, no auth header
            methods_to_try.append(("client_id_in_body", None))
            # Method 3: Basic Auth with client_id:"" (empty secret)
            methods_to_try.append(("basic_auth_empty_secret", ""))
        
        for method_name, secret_value in methods_to_try:
            if _is_debug():
                click.echo(f"DEBUG: Trying introspection method: {method_name}", err=True)
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }
            data = {
                "token": access_token,
            }
            
            if method_name == "basic_auth_with_secret":
                # Basic Auth with client_id:client_secret
                credentials = base64.b64encode(
                    f"{client_id}:{secret_value}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
            elif method_name == "bearer_token":
                # Bearer token authentication (access token as Bearer)
                # Some OAuth servers allow introspection using the token itself
                headers["Authorization"] = f"Bearer {access_token}"
            elif method_name == "basic_auth_empty_secret":
                # Basic Auth with client_id:"" (empty secret)
                credentials = base64.b64encode(
                    f"{client_id}:".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
            elif method_name == "client_id_in_body":
                # No auth header, client_id in body
                data["client_id"] = client_id
            
            resp = requests.post(
                introspect_url,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if _is_debug():
                click.echo(f"DEBUG: Method {method_name} - Response status: {resp.status_code}", err=True)
                if resp.status_code != 200:
                    click.echo(f"DEBUG: Method {method_name} - Response body: {resp.text}", err=True)
            
            if resp.status_code == 200:
                result = resp.json()
                if _is_debug():
                    click.echo(f"DEBUG: Introspection succeeded with method {method_name}: {result}", err=True)
                return result.get("active", False)
        
        # All methods failed
        if os.environ.get("DEBUG"):
            click.echo("DEBUG: All introspection methods failed", err=True)
        return False
    except Exception as e:
        if os.environ.get("DEBUG"):
            click.echo(f"DEBUG: Introspection exception: {e}", err=True)
        return False


def are_credentials_valid() -> bool:
    """Check if existing credentials are valid.
    
    Returns:
        True if credentials exist and are valid, False otherwise
    """
    if not os.path.exists(ACCESS_TOK_PATH):
        return False
    
    try:
        access_refresh_dict = parse_credentials_file(ACCESS_TOK_PATH)
        if access_refresh_dict is None:
            return False
        
        access = access_refresh_dict.get("access")
        refresh = access_refresh_dict.get("refresh")
        
        if not access:
            return False
        
        # Check if these are OAuth tokens (have token_url or client_id)
        is_oauth = bool(access_refresh_dict.get("token_url") or access_refresh_dict.get("client_id"))
        
        if is_oauth:
            # Use OAuth introspection endpoint
            from biolm.core.const import BIOLMAI_OAUTH_CLIENT_SECRET
            client_id = access_refresh_dict.get("client_id")
            # Try credentials file first, then environment variable
            client_secret = access_refresh_dict.get("client_secret") or BIOLMAI_OAUTH_CLIENT_SECRET
            return _validate_oauth_token(access, client_id, client_secret)
        else:
            # Legacy token validation using Cookie-based endpoint
            if not refresh:
                return False
            # Suppress output from validate_user_auth by capturing it
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                resp = validate_user_auth(access=access, refresh=refresh)
                sys.stdout = old_stdout
                return resp.status_code == 200 and "code" not in resp.json()
            except Exception:
                sys.stdout = old_stdout
                return False
    except Exception:
        return False


def oauth_login(
    *,
    client_id: Optional[str] = None,
    scope: str = "read write",
    auth_url: Optional[str] = None,
    token_url: Optional[str] = None,
    redirect_uri: Optional[str] = None,
) -> dict:
    """Perform OAuth login using PKCE and persist tokens to ~/.biolmai/credentials.
    
    Args:
        client_id: OAuth client ID (defaults to BIOLMAI_PUBLIC_CLIENT_ID from const)
        scope: OAuth scope string
        auth_url: Authorization endpoint (defaults to OAUTH_AUTHORIZE_URL)
        token_url: Token endpoint (defaults to OAUTH_TOKEN_URL)
        redirect_uri: Redirect URI (defaults to OAUTH_REDIRECT_URI)
    
    Returns:
        Token response dict
    """
    from biolm.core.const import (
        BIOLMAI_OAUTH_CLIENT_SECRET,
        BIOLMAI_PUBLIC_CLIENT_ID,
        OAUTH_AUTHORIZE_URL,
        OAUTH_REDIRECT_URI,
        OAUTH_TOKEN_URL,
    )
    
    client_id = client_id or BIOLMAI_PUBLIC_CLIENT_ID
    if not client_id:
        raise ValueError(
            "OAuth client ID required. Set BIOLMAI_OAUTH_CLIENT_ID environment variable "
            "or pass --client-id to login command."
        )
    
    # Get client_secret from environment (if provided)
    client_secret = BIOLMAI_OAUTH_CLIENT_SECRET
    
    auth_url = auth_url or OAUTH_AUTHORIZE_URL
    token_url = token_url or OAUTH_TOKEN_URL
    redirect_uri = redirect_uri or OAUTH_REDIRECT_URI
    
    # Perform browser/PKCE flow
    token_data = _browser_login_with_pkce(
        client_id, scope, auth_url, token_url, redirect_uri, client_secret=client_secret
    )
    
    # Extract tokens
    access_token = token_data.get("access_token") or token_data.get("access")
    refresh_token = token_data.get("refresh_token") or token_data.get("refresh")
    expires_in = token_data.get("expires_in", 3600)
    
    if not access_token:
        raise ValueError("No access token in OAuth response")
    
    # Prepare credentials dict
    creds = {
        "access": access_token,
        "refresh": refresh_token,
        "expires_in": expires_in,
        "expires_at": int(time.time()) + int(expires_in),
        "token_url": token_url,
        "client_id": client_id,
    }
    
    # Save client_secret if provided (for confidential clients)
    if client_secret:
        creds["client_secret"] = client_secret
    
    # Persist credentials
    save_access_refresh_token(creds)
    
    return token_data
