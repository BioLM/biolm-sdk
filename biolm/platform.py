"""Platform management client for BioLM console APIs.

Talks to ``{BIOLM_BASE_DOMAIN}/console/api/`` with a persistent ``httpx.Client``
so Django session cookies survive account-context switches. A workspace is an
account + environment pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx

from biolm.core.const import BIOLM_BASE_DOMAIN
from biolm.core.http import CredentialsProvider

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _default_base_url() -> str:
    return "{}/console/api/".format(BIOLM_BASE_DOMAIN.rstrip("/"))


class PlatformError(Exception):
    """Raised when a console API request fails or returns an error status."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class WorkspaceNotFoundError(PlatformError):
    """No workspace matched the requested path."""


class AmbiguousWorkspaceError(PlatformError):
    """More than one workspace matched the requested path."""


@dataclass(frozen=True)
class Workspace:
    """Immutable account + environment pair.

    ``account`` / ``environment`` are path segments (org slug or personal label,
    and environment slug from the API ``name`` field). IDs remain authoritative.
    """

    account_type: str
    account_id: int
    environment_id: int
    account: str
    environment: str

    @property
    def path(self) -> str:
        return "{}/{}".format(self.account, self.environment)

    def __str__(self) -> str:
        return self.path


def _personal_label(account_details: Optional[Dict[str, Any]]) -> str:
    if not account_details:
        return "personal"
    for key in ("username", "slug", "name", "email"):
        value = account_details.get(key)
        if value:
            text = str(value).strip()
            if key == "email" and "@" in text:
                text = text.split("@", 1)[0]
            if text:
                return text
    return "personal"


def _parse_cookie_header(header: str) -> Dict[str, str]:
    """Parse a Cookie request header into name/value pairs.

    Supports the credentials format ``access=...;refresh=...`` (no spaces after
    ``;``) without mangling values that contain ``=``.
    """
    cookies: Dict[str, str] = {}
    if not header:
        return cookies
    for part in header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name:
            cookies[name] = value.strip()
    return cookies


def _headers_and_cookies(
    auth_headers: Dict[str, str],
) -> Tuple[Dict[str, str], httpx.Cookies]:
    """Split CredentialsProvider headers into httpx headers + cookie jar.

    Moves a ``Cookie`` header into an :class:`httpx.Cookies` jar so Set-Cookie
    session cookies can merge with OAuth access/refresh cookies. Leaves
    ``Authorization`` Token headers untouched.

    When credentials are OAuth cookie-only (``access`` / ``refresh``), also set
    ``Authorization: Bearer <access>``. Production ``/console/api/`` ignores
    Cookie-only OAuth and requires Bearer (or Knox ``Token``).
    """
    headers = dict(auth_headers)
    cookies = httpx.Cookies()
    cookie_header = None
    for key in list(headers.keys()):
        if key.lower() == "cookie":
            cookie_header = headers.pop(key)
            break
    if cookie_header:
        for name, value in _parse_cookie_header(cookie_header).items():
            cookies.set(name, value)
    has_authorization = any(key.lower() == "authorization" for key in headers)
    access = cookies.get("access")
    if access and not has_authorization:
        headers["Authorization"] = "Bearer {}".format(access)
    return headers, cookies


class PlatformClient:
    """Sync client for BioLM platform (orgs, environments, budgets, workspaces).

    Instances retain session cookies and active account context. They are
    stateful, session-scoped, and not safe for concurrent use across threads.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = (base_url or _default_base_url()).rstrip("/") + "/"
        try:
            auth_headers = CredentialsProvider.get_auth_headers(api_key)
        except AssertionError as exc:
            raise PlatformError(
                "No BioLM credentials found. Set BIOLM_TOKEN or run `biolm login`."
            ) from exc
        self._headers, auth_cookies = _headers_and_cookies(auth_headers)
        self._owns_client = client is None
        self._csrf_ready = False
        parsed = urlparse(self._base_url)
        self._origin = "{}://{}".format(parsed.scheme, parsed.netloc)
        if client is not None:
            self._client = client
        else:
            kwargs: Dict[str, Any] = {
                "base_url": self._base_url,
                "headers": self._headers,
                "cookies": auth_cookies,
                "timeout": timeout,
            }
            if transport is not None:
                kwargs["transport"] = transport
            self._client = httpx.Client(**kwargs)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "PlatformClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _url(self, path: str) -> str:
        return path.lstrip("/")

    def _ensure_csrf(self) -> None:
        """Bootstrap Django csrftoken for unsafe console API methods.

        Hosted ``/console/api/`` enforces CSRF on POST even when Bearer auth is
        present. The login page sets the csrftoken cookie; we mirror browser
        clients by sending ``X-CSRFToken`` + ``Referer``.
        """
        if self._csrf_ready:
            return
        login_url = "{}/ui/accounts/login/".format(self._origin.rstrip("/"))
        try:
            # Login may 302; csrftoken is set on the final HTML response.
            self._client.get(login_url, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise PlatformError(
                "Platform CSRF bootstrap GET {} failed: {}".format(login_url, exc)
            ) from exc
        csrf = self._client.cookies.get("csrftoken")
        if not csrf:
            # httpx may key cookies by domain; scan the jar as a fallback.
            for cookie in self._client.cookies.jar:
                if cookie.name == "csrftoken" and cookie.value:
                    csrf = cookie.value
                    break
        if csrf:
            self._client.headers["X-CSRFToken"] = csrf
            self._client.headers["Referer"] = "{}/".format(self._origin.rstrip("/"))
        self._csrf_ready = True

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail: Any
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        message = "HTTP {}: {}".format(response.status_code, detail)
        raise PlatformError(message, status_code=response.status_code, response=detail)

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        endpoint = self._url(path)
        method_upper = method.upper()
        if method_upper in _UNSAFE_METHODS:
            self._ensure_csrf()
        try:
            response = self._client.request(method_upper, endpoint, json=json)
        except httpx.HTTPError as exc:
            raise PlatformError(
                "Platform request {} {} failed: {}".format(
                    method_upper, endpoint, exc
                )
            ) from exc
        self._raise_for_status(response)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except Exception:
            return response.text

    def get_context(self) -> Dict[str, Any]:
        return self._request("GET", "account-context/")

    def set_context(
        self,
        account_type: str,
        account_id: Optional[int] = None,
        environment_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"account_type": account_type}
        if account_id is not None:
            payload["account_id"] = int(account_id)
        if environment_id is not None:
            payload["environment_id"] = int(environment_id)
        else:
            payload["environment_id"] = None
        return self._request("POST", "account-context/", json=payload)

    def list_organizations(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "orgs/")
        return list(data or [])

    def create_organization(self, name: str, slug: str) -> Dict[str, Any]:
        return self._request("POST", "orgs/", json={"name": name, "slug": slug})

    def get_organization(self, org_id: int) -> Dict[str, Any]:
        return self._request("GET", "orgs/{}/".format(int(org_id)))

    def invite_to_organization(
        self,
        org_id: int,
        email: str,
        role: str = "member",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "orgs/{}/invite/".format(int(org_id)),
            json={"email": email, "role": role},
        )

    def list_environments(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "environments/")
        return list(data or [])

    def create_environment(self, name: str) -> Dict[str, Any]:
        return self._request("POST", "environments/", json={"name": name})

    def get_budget(self) -> Dict[str, Any]:
        return self._request("GET", "account-budget/")

    def set_budget(self, workspace_budget: float) -> Dict[str, Any]:
        return self._request(
            "POST",
            "account-budget/",
            json={"workspace_budget": float(workspace_budget)},
        )

    def _account_label(
        self,
        account_type: str,
        account_id: int,
        account_details: Optional[Dict[str, Any]],
        org_by_id: Dict[int, Dict[str, Any]],
        personal_label: str,
    ) -> str:
        if account_type == "organization":
            org = org_by_id.get(int(account_id))
            if org and org.get("slug"):
                return str(org["slug"])
            if account_details and account_details.get("slug"):
                return str(account_details["slug"])
            return str(account_id)
        return personal_label

    def _workspaces_for_account(
        self,
        account_type: str,
        account_id: int,
        account_label: str,
    ) -> List[Workspace]:
        envs = self.list_environments()
        out: List[Workspace] = []
        for env in envs:
            out.append(
                Workspace(
                    account_type=account_type,
                    account_id=int(account_id),
                    environment_id=int(env["id"]),
                    account=account_label,
                    environment=str(env["name"]),
                )
            )
        return out

    def list_workspaces(self) -> List[Workspace]:
        """List personal and organization workspaces, restoring prior context.

        The environments endpoint is session-scoped, so enumeration must switch
        the backend account context for each account. Those switches may cause
        the backend to ensure or select a default environment; there is no
        context-free environment listing endpoint available.
        """
        original = self.get_context()
        orgs = self.list_organizations()

        try:
            personal = self._discover_personal()
            personal_id = int(personal["account_id"])
            personal_label = personal["label"]

            workspaces: List[Workspace] = []

            self.set_context("user", personal_id, environment_id=None)
            workspaces.extend(
                self._workspaces_for_account("user", personal_id, personal_label)
            )

            for org in orgs:
                org_id = int(org["id"])
                self.set_context("organization", org_id, environment_id=None)
                label = str(org.get("slug") or org_id)
                workspaces.extend(
                    self._workspaces_for_account("organization", org_id, label)
                )

            return workspaces
        finally:
            self.set_context(
                original.get("account_type", "user"),
                original.get("account_id"),
                environment_id=original.get("environment_id"),
            )

    def current_workspace(self) -> Workspace:
        ctx = self.get_context()
        account_type = ctx.get("account_type", "user")
        account_id = int(ctx["account_id"])
        environment_id = ctx.get("environment_id")
        orgs = self.list_organizations()
        org_by_id = {int(o["id"]): o for o in orgs}

        if account_type == "user":
            account_label = _personal_label(ctx.get("account_details"))
        else:
            account_label = self._account_label(
                account_type,
                account_id,
                ctx.get("account_details"),
                org_by_id,
                "personal",
            )

        envs = self.list_environments()
        if environment_id is None:
            if not envs:
                raise WorkspaceNotFoundError(
                    "Current context has no environment_id and no environments"
                )
            env = next((e for e in envs if e.get("is_default")), envs[0])
        else:
            env = next((e for e in envs if int(e["id"]) == int(environment_id)), None)
            if env is None:
                raise WorkspaceNotFoundError(
                    "Environment {} not found in current account".format(environment_id)
                )

        return Workspace(
            account_type=account_type,
            account_id=account_id,
            environment_id=int(env["id"]),
            account=account_label,
            environment=str(env["name"]),
        )

    def _resolve_path(self, path: str) -> Workspace:
        matches = [ws for ws in self.list_workspaces() if ws.path == path]
        if not matches:
            raise WorkspaceNotFoundError("No workspace found for path {!r}".format(path))
        if len(matches) > 1:
            raise AmbiguousWorkspaceError(
                "Ambiguous workspace path {!r}: {} matches".format(path, len(matches))
            )
        return matches[0]

    def get_workspace(self, path: str) -> Workspace:
        """Resolve and return the workspace matching an exact path."""
        return self._resolve_path(path)

    def switch_workspace(self, workspace_or_path: Union[Workspace, str]) -> Workspace:
        if isinstance(workspace_or_path, Workspace):
            ws = workspace_or_path
        else:
            ws = self._resolve_path(str(workspace_or_path))
        self.set_context(ws.account_type, ws.account_id, environment_id=ws.environment_id)
        return ws

    def _resolve_account_slug(
        self,
        account: str,
        personal_label: str,
        personal_id: int,
        orgs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return {account_type, account_id, account_label} for an exact slug."""
        if account == personal_label:
            return {
                "account_type": "user",
                "account_id": personal_id,
                "account_label": personal_label,
            }
        org_matches = [o for o in orgs if o.get("slug") == account]
        if not org_matches:
            raise WorkspaceNotFoundError("No account found for slug {!r}".format(account))
        if len(org_matches) > 1:
            raise AmbiguousWorkspaceError(
                "Ambiguous account slug {!r}".format(account)
            )
        org = org_matches[0]
        return {
            "account_type": "organization",
            "account_id": int(org["id"]),
            "account_label": str(org["slug"]),
        }

    def _discover_personal(self) -> Dict[str, Any]:
        """Return personal account_id and label; may change session context."""
        ctx = self.get_context()
        if ctx.get("account_type") == "user":
            return {
                "account_id": int(ctx["account_id"]),
                "label": _personal_label(ctx.get("account_details")),
            }
        self.set_context("user", account_id=None, environment_id=None)
        user_ctx = self.get_context()
        return {
            "account_id": int(user_ctx["account_id"]),
            "label": _personal_label(user_ctx.get("account_details")),
        }

    def create_workspace(
        self,
        name: str,
        account: Optional[str] = None,
    ) -> Workspace:
        original = self.get_context()
        orgs = self.list_organizations()
        org_by_id = {int(o["id"]): o for o in orgs}

        try:
            personal = self._discover_personal()

            if account is None:
                # Create under the caller's original account
                self.set_context(
                    original.get("account_type", "user"),
                    original.get("account_id"),
                    environment_id=original.get("environment_id"),
                )
                target_type = original.get("account_type", "user")
                target_id = int(original["account_id"])
                if target_type == "user":
                    target_label = personal["label"]
                else:
                    target_label = self._account_label(
                        target_type,
                        target_id,
                        None,
                        org_by_id,
                        personal["label"],
                    )
            else:
                resolved = self._resolve_account_slug(
                    account,
                    personal["label"],
                    int(personal["account_id"]),
                    orgs,
                )
                target_type = resolved["account_type"]
                target_id = int(resolved["account_id"])
                target_label = resolved["account_label"]
                self.set_context(target_type, target_id, environment_id=None)

            env = self.create_environment(name)
            return Workspace(
                account_type=target_type,
                account_id=target_id,
                environment_id=int(env["id"]),
                account=target_label,
                environment=str(env["name"]),
            )
        finally:
            self.set_context(
                original.get("account_type", "user"),
                original.get("account_id"),
                environment_id=original.get("environment_id"),
            )

    def create_api_key(self, account: Optional[str] = None) -> Dict[str, str]:
        """Create an API key and return its one-time secret.

        The key is owned by the active server-side account context. Pass
        ``account`` (an org slug or the personal label) to create the key under
        a different account; the switch and creation happen in this one session
        so organization ownership cannot silently fall back to personal. The
        original context is restored afterward. The returned ``token`` is shown
        only once and is not stored by the SDK.
        """
        if account is None:
            return self._request("POST", "auth/generate_token/")

        original = self.get_context()
        orgs = self.list_organizations()
        try:
            personal = self._discover_personal()
            resolved = self._resolve_account_slug(
                account,
                personal["label"],
                int(personal["account_id"]),
                orgs,
            )
            self.set_context(
                resolved["account_type"],
                int(resolved["account_id"]),
                environment_id=None,
            )
            return self._request("POST", "auth/generate_token/")
        finally:
            self.set_context(
                original.get("account_type", "user"),
                original.get("account_id"),
                environment_id=original.get("environment_id"),
            )

    def delete_api_key(self, token_or_prefix: str) -> None:
        """Revoke an API key by full token or eight-character prefix."""
        token = (token_or_prefix or "").strip()
        if not token:
            raise ValueError("token_or_prefix must not be blank.")
        self._request("DELETE", "auth/delete_token/", json={"token": token})
        return None
