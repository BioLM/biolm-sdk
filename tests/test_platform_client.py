"""Tests for biolm.platform.PlatformClient and Workspace value object.

Uses httpx.MockTransport so behavior (auth headers, cookie-backed session
switching, workspace composition) is exercised without real network I/O.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import pytest

from biolm.platform import (
    AmbiguousOrganizationError,
    AmbiguousWorkspaceError,
    OrganizationNotFoundError,
    PlatformClient,
    PlatformError,
    Workspace,
    WorkspaceNotFoundError,
)


BASE = "https://biolm.ai/console/api"


def _path(url: str) -> str:
    return urlparse(str(url)).path.rstrip("/") + "/"


def _parse_cookie_header(header: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not header:
        return parsed
    for part in header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        parsed[name.strip()] = value.strip()
    return parsed


class FakeConsole:
    """In-memory console API with session-scoped environments."""

    def __init__(self) -> None:
        self._default_session: Dict[str, Any] = {
            "account_type": "organization",
            "account_id": 10,
            "environment_id": 100,
        }
        # Account context keyed by Django sessionid
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.user_id = 1
        self.personal_details: Dict[str, Any] = {
            "id": 1,
            "username": "astewart",
            "workspace_budget": 50.0,
        }
        self.current_user: Dict[str, Any] = {
            "id": 1,
            "username": "astewart",
            "email": "astewart@example.com",
            "first_name": "Alex",
            "last_name": "Stewart",
            "workspace_budget": 50.0,
        }
        self.orgs: List[Dict[str, Any]] = [
            {"id": 10, "name": "BioLM", "slug": "biolm"},
            {"id": 20, "name": "Acme", "slug": "acme"},
        ]
        self.envs: Dict[Tuple[str, int], List[Dict[str, Any]]] = {
            ("user", 1): [
                {"id": 1, "name": "default", "is_default": True},
                {"id": 2, "name": "sandbox", "is_default": False},
            ],
            ("organization", 10): [
                {"id": 100, "name": "example-space", "is_default": True},
                {"id": 101, "name": "staging", "is_default": False},
            ],
            ("organization", 20): [
                {"id": 200, "name": "prod", "is_default": True},
            ],
        }
        self.budget = {
            "total_budget": 100.0,
            "current_usage": 25.0,
            "remaining_budget": 75.0,
            "currency": "USD",
        }
        self.requests: List[Tuple[str, str, Optional[Dict[str, Any]], Dict[str, str]]] = []
        self._next_env_id = 1000
        self._next_session = 0
        self.fail_status: Optional[int] = None
        self.fail_body: Any = {"error": "boom"}
        # Knox API keys: each entry tracks ownership derived from session context.
        self.api_keys: List[Dict[str, Any]] = []
        self._next_token = 0
        self.usage_queries: List[Dict[str, str]] = []
        self.usage_fail_status: Optional[int] = None

    @property
    def session(self) -> Dict[str, Any]:
        """Most recently touched session (compat for existing tests)."""
        if not self.sessions:
            return dict(self._default_session)
        sid = "sess-{}".format(self._next_session)
        if sid in self.sessions:
            return self.sessions[sid]
        return next(reversed(list(self.sessions.values())))

    def _auth_ok(self, request: httpx.Request) -> Optional[httpx.Response]:
        # Match production /console/api/: Cookie-only OAuth is ignored; Token and
        # Bearer Authorization headers are accepted.
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Token ") or auth.startswith("Bearer "):
            return None
        return httpx.Response(
            403,
            json={"detail": "Authentication credentials were not provided."},
        )

    def _csrf_ok(self, request: httpx.Request) -> Optional[httpx.Response]:
        if request.method.upper() not in ("POST", "PUT", "PATCH", "DELETE"):
            return None
        cookies = _parse_cookie_header(request.headers.get("cookie", ""))
        csrf_cookie = cookies.get("csrftoken")
        header_map = {k.lower(): v for k, v in request.headers.items()}
        csrf_header = header_map.get("x-csrftoken")
        if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
            return httpx.Response(
                403,
                json={"detail": "CSRF Failed: CSRF cookie not set."},
            )
        return None

    def _session_id(self, request: httpx.Request) -> str:
        cookies = _parse_cookie_header(request.headers.get("cookie", ""))
        sid = cookies.get("sessionid")
        if sid:
            if sid not in self.sessions:
                self.sessions[sid] = dict(self._default_session)
            return sid
        self._next_session += 1
        sid = "sess-{}".format(self._next_session)
        self.sessions[sid] = dict(self._default_session)
        return sid

    def _json(
        self,
        request: httpx.Request,
        status: int,
        body: Any,
        session_id: Optional[str] = None,
    ) -> httpx.Response:
        sid = session_id or self._session_id(request)
        return httpx.Response(
            status,
            json=body,
            request=request,
            headers={"Set-Cookie": "sessionid={}; Path=/".format(sid)},
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = _path(request.url)
        method = request.method.upper()
        body: Optional[Dict[str, Any]] = None
        if request.content:
            try:
                body = json.loads(request.content.decode("utf-8"))
            except Exception:
                body = None
        self.requests.append((method, path, body, dict(request.headers)))

        # CSRF bootstrap page (outside /console/api/); no auth required.
        if path == "/ui/accounts/login/" and method == "GET":
            return httpx.Response(
                200,
                content=b"ok",
                request=request,
                headers={"Set-Cookie": "csrftoken=test-csrf; Path=/"},
            )

        auth_err = self._auth_ok(request)
        if auth_err is not None:
            return auth_err

        csrf_err = self._csrf_ok(request)
        if csrf_err is not None:
            return csrf_err

        sid = self._session_id(request)
        session = self.sessions[sid]

        if self.fail_status is not None:
            status = self.fail_status
            self.fail_status = None
            return self._json(request, status, self.fail_body, session_id=sid)

        if path == "/api/users/me/" and method == "GET":
            return self._json(request, 200, dict(self.current_user), session_id=sid)

        if path.endswith("/account-context/") and method == "GET":
            return self._json(request, 200, self._context_payload(session), session_id=sid)

        if path.endswith("/account-context/") and method == "POST":
            assert body is not None
            account_type = body.get("account_type", "user")
            account_id = body.get("account_id")
            environment_id = body.get("environment_id")
            if account_type == "user" and account_id in (None, "", "null"):
                account_id = self.user_id
            account_id = int(account_id)
            if environment_id in ("", "null"):
                environment_id = None
            if environment_id is not None:
                environment_id = int(environment_id)
            else:
                envs = self.envs.get((account_type, account_id), [])
                environment_id = envs[0]["id"] if envs else None
            session.clear()
            session.update(
                {
                    "account_type": account_type,
                    "account_id": account_id,
                    "environment_id": environment_id,
                }
            )
            return self._json(request, 200, dict(session), session_id=sid)

        if path.endswith("/orgs/") and method == "GET":
            return self._json(request, 200, list(self.orgs), session_id=sid)

        if path.endswith("/orgs/") and method == "POST":
            assert body is not None
            org = {
                "id": 99,
                "name": body["name"],
                "slug": body["slug"],
            }
            self.orgs.append(org)
            return self._json(request, 200, org, session_id=sid)

        if "/orgs/" in path and path.endswith("/invite/") and method == "POST":
            assert body is not None
            return self._json(
                request,
                200,
                {"success": True, "email": body.get("email"), "role": body.get("role", "member")},
                session_id=sid,
            )

        if "/orgs/" in path and method == "GET" and path.count("/") >= 4:
            # /console/api/orgs/{id}/
            org_id = int(path.rstrip("/").split("/")[-1])
            for org in self.orgs:
                if org["id"] == org_id:
                    return self._json(request, 200, dict(org), session_id=sid)
            return self._json(request, 404, {"error": "not_found"}, session_id=sid)

        if path.endswith("/environments/") and method == "GET":
            key = (session["account_type"], int(session["account_id"]))
            return self._json(request, 200, list(self.envs.get(key, [])), session_id=sid)

        if path.endswith("/environments/") and method == "POST":
            assert body is not None
            name = body.get("name") or body.get("slug")
            slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
            self._next_env_id += 1
            env = {"id": self._next_env_id, "name": slug, "is_default": False}
            key = (session["account_type"], int(session["account_id"]))
            self.envs.setdefault(key, []).append(env)
            return self._json(request, 200, env, session_id=sid)

        if path.endswith("/account-budget/") and method == "GET":
            return self._json(request, 200, dict(self.budget), session_id=sid)

        if path.endswith("/account-budget/") and method == "POST":
            assert body is not None
            self.budget["total_budget"] = float(body["workspace_budget"])
            response = {"workspace_budget": float(body["workspace_budget"])}
            return self._json(request, 200, response, session_id=sid)

        if path.endswith("/usage-summary/") and method == "GET":
            if self.usage_fail_status is not None:
                fail_status = self.usage_fail_status
                self.usage_fail_status = None
                return self._json(
                    request,
                    fail_status,
                    {"error": "usage unavailable"},
                    session_id=sid,
                )
            params = dict(request.url.params)
            self.usage_queries.append(params)
            return self._json(
                request,
                200,
                {
                    "account_type": session["account_type"],
                    "account_id": session["account_id"],
                    "institute_id": 501,
                    "selected_year": int(params.get("year", 2026)),
                    "selected_month": int(params.get("month", 7)),
                    "current_year": 2026,
                    "current_month": 7,
                    "env_list": [{"id": 200, "slug": "prod"}],
                    "filter_env_id": (
                        int(params["env"]) if params.get("env") else None
                    ),
                    "current_usage_amount": 12.5,
                    "environment_usage_amount": 3.0,
                    "environment_label": "prod",
                    "model_charges": [
                        {
                            "model_name": "esm2-8m",
                            "total_biolm_charge": 12.5,
                        }
                    ],
                },
                session_id=sid,
            )

        if path.endswith("/auth/generate_token/") and method == "POST":
            self._next_token += 1
            token = "knoxtok{:02d}-{}".format(self._next_token, "s" * 56)
            self.api_keys.append(
                {
                    "token": token,
                    "owner_type": session["account_type"],
                    "owner_id": int(session["account_id"]),
                }
            )
            return self._json(request, 200, {"token": token}, session_id=sid)

        if path.endswith("/auth/delete_token/") and method == "DELETE":
            assert body is not None
            supplied = str(body.get("token", ""))
            prefix = supplied[:8]
            before = len(self.api_keys)
            self.api_keys = [
                k for k in self.api_keys if not k["token"].startswith(prefix)
            ]
            if len(self.api_keys) == before:
                return self._json(request, 404, {"error": "not_found"}, session_id=sid)
            return httpx.Response(
                204,
                request=request,
                headers={"Set-Cookie": "sessionid={}; Path=/".format(sid)},
            )

        return self._json(request, 404, {"error": "not found", "path": path}, session_id=sid)

    def _context_payload(self, session: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(session)
        if payload["account_type"] == "organization":
            org = next((o for o in self.orgs if o["id"] == payload["account_id"]), None)
            payload["account_details"] = dict(org) if org else None
        else:
            payload["account_details"] = dict(self.personal_details)
        return payload


@pytest.fixture
def fake() -> FakeConsole:
    return FakeConsole()


@pytest.fixture
def client(fake: FakeConsole) -> PlatformClient:
    transport = httpx.MockTransport(fake.handler)
    return PlatformClient(api_key="test-token", base_url=BASE + "/", transport=transport)


def test_workspace_path_and_str_immutable():
    ws = Workspace(
        account_type="organization",
        account_id=10,
        environment_id=100,
        account="biolm",
        environment="example-space",
    )
    assert ws.path == "biolm/example-space"
    assert str(ws) == "biolm/example-space"
    with pytest.raises(Exception):
        ws.account = "other"  # type: ignore[misc]


def test_auth_headers_sent(client: PlatformClient, fake: FakeConsole):
    client.get_context()
    assert fake.requests
    headers = {k.lower(): v for k, v in fake.requests[0][3].items()}
    assert headers.get("authorization") == "Token test-token"


def test_missing_credentials_raise_clear_platform_error(monkeypatch):
    monkeypatch.setattr(
        "biolm.platform.CredentialsProvider.get_auth_headers",
        staticmethod(
            lambda api_key=None: (_ for _ in ()).throw(
                AssertionError("No credentials found")
            )
        ),
    )

    with pytest.raises(PlatformError) as excinfo:
        PlatformClient()

    message = str(excinfo.value).lower()
    assert "biolm_token" in message
    assert "login" in message


def test_request_wraps_http_error_with_endpoint_and_cause():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(fail)
    client = PlatformClient(
        api_key="tok",
        base_url=BASE + "/",
        transport=transport,
    )

    with pytest.raises(PlatformError) as excinfo:
        client.get_budget()

    message = str(excinfo.value)
    assert "GET" in message
    assert "account-budget/" in message
    assert "connection refused" in message
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


def test_oauth_cookie_auth_sends_bearer_and_merges_sessionid(
    fake: FakeConsole, monkeypatch
):
    """OAuth Cookie credentials must become Bearer auth and stay in the cookie jar.

    Production ``/console/api/`` ignores Cookie-only OAuth (403 credentials not
    provided) but accepts ``Authorization: Bearer <access>``. Cookies still need
    to live in the jar so Set-Cookie sessionid/csrftoken can merge in.
    """
    monkeypatch.setattr(
        "biolm.platform.CredentialsProvider.get_auth_headers",
        staticmethod(lambda api_key=None: {"Cookie": "access=a;refresh=r"}),
    )
    transport = httpx.MockTransport(fake.handler)
    client = PlatformClient(base_url=BASE + "/", transport=transport)

    client.get_context()
    client.list_environments()

    assert len(fake.requests) >= 2
    first_headers = {k.lower(): v for k, v in fake.requests[0][3].items()}
    assert first_headers.get("authorization") == "Bearer a"
    second_headers = {k.lower(): v for k, v in fake.requests[1][3].items()}
    cookie = second_headers.get("cookie", "")
    parts = _parse_cookie_header(cookie)
    assert parts.get("access") == "a"
    assert parts.get("refresh") == "r"
    assert "sessionid" in parts, "expected sessionid merged into Cookie with OAuth tokens"
    assert parts["sessionid"]  # non-empty


def test_mutating_requests_send_csrf_token(client: PlatformClient, fake: FakeConsole):
    """Unsafe console methods require csrftoken cookie + X-CSRFToken header."""
    client.set_context("organization", 10, environment_id=100)

    login_gets = [
        (method, path)
        for method, path, _body, _headers in fake.requests
        if method == "GET" and path == "/ui/accounts/login/"
    ]
    assert login_gets, "expected CSRF bootstrap GET to /ui/accounts/login/"

    post = next(
        (headers for method, path, _body, headers in fake.requests if method == "POST"),
        None,
    )
    assert post is not None
    headers = {k.lower(): v for k, v in post.items()}
    assert headers.get("x-csrftoken") == "test-csrf"
    assert "csrftoken=test-csrf" in headers.get("cookie", "")
    assert headers.get("referer", "").startswith("https://biolm.ai")


def test_context_manager_and_close(fake: FakeConsole):
    transport = httpx.MockTransport(fake.handler)
    with PlatformClient(api_key="tok", base_url=BASE + "/", transport=transport) as pc:
        assert pc.get_context()["account_type"] == "organization"
    # closed client should not accept new requests
    with pytest.raises(Exception):
        pc.get_context()


def test_get_and_set_context(client: PlatformClient, fake: FakeConsole):
    ctx = client.get_context()
    assert ctx["account_type"] == "organization"
    assert ctx["account_id"] == 10
    assert ctx["environment_id"] == 100

    updated = client.set_context("user", 1, environment_id=2)
    assert updated["account_type"] == "user"
    assert updated["account_id"] == 1
    assert updated["environment_id"] == 2
    assert fake.session["account_type"] == "user"


def test_get_current_user_uses_origin_api_and_returns_identity(
    client: PlatformClient, fake: FakeConsole
):
    user = client.get_current_user()

    assert user == fake.current_user
    assert fake.requests[-1][0:2] == ("GET", "/api/users/me/")
    headers = {key.lower(): value for key, value in fake.requests[-1][3].items()}
    assert headers["authorization"] == "Token test-token"


def test_get_current_user_maps_api_error_to_platform_error(
    client: PlatformClient, fake: FakeConsole
):
    fake.fail_status = 401
    fake.fail_body = {"detail": "invalid token"}

    with pytest.raises(PlatformError) as excinfo:
        client.get_current_user()

    assert excinfo.value.status_code == 401
    assert excinfo.value.response == {"detail": "invalid token"}


def test_persistent_cookies_across_session_switches(client: PlatformClient, fake: FakeConsole):
    client.set_context("user", 1)
    client.list_environments()
    # After the first console API response sets sessionid, later API calls must send it.
    api_requests = [
        r
        for r in fake.requests
        if r[1].startswith("/console/api/")
    ]
    assert len(api_requests) >= 2
    later = [
        r
        for r in api_requests[1:]
        if "cookie" in {k.lower() for k in r[3]}
    ]
    assert later, "expected cookie jar to persist session cookie on later requests"
    cookie_header = next(v for k, v in later[0][3].items() if k.lower() == "cookie")
    assert "sessionid=" in cookie_header


def test_list_organizations_create_get_invite(client: PlatformClient, fake: FakeConsole):
    orgs = client.list_organizations()
    assert {o["slug"] for o in orgs} == {"biolm", "acme"}

    created = client.create_organization("New Co", "newco")
    assert created["slug"] == "newco"
    assert any(o["slug"] == "newco" for o in client.list_organizations())

    detail = client.get_organization(10)
    assert detail["slug"] == "biolm"

    invited = client.invite_to_organization(10, "user@example.com", role="admin")
    assert invited["success"] is True
    invite_reqs = [r for r in fake.requests if r[0] == "POST" and r[1].endswith("/invite/")]
    assert invite_reqs[-1][2] == {"email": "user@example.com", "role": "admin"}


@pytest.mark.parametrize(
    ("identifier", "expected_id"),
    [
        ("Acme", 20),
        ("acme", 20),
        (20, 20),
        ("20", 20),
    ],
)
def test_get_organization_resolves_exact_identifier(
    client: PlatformClient,
    fake: FakeConsole,
    identifier: Any,
    expected_id: int,
):
    org = client.get_organization(identifier)

    assert org["id"] == expected_id
    assert fake.requests[-1][1].endswith("/console/api/orgs/{}/".format(expected_id))


def test_get_organization_missing_raises_specific_platform_error(
    client: PlatformClient,
):
    with pytest.raises(OrganizationNotFoundError) as excinfo:
        client.get_organization("missing")

    assert isinstance(excinfo.value, PlatformError)


def test_get_organization_rejects_name_slug_cross_collision(
    client: PlatformClient,
    fake: FakeConsole,
):
    fake.orgs[0]["name"] = "acme"

    with pytest.raises(AmbiguousOrganizationError) as excinfo:
        client.get_organization("acme")

    assert isinstance(excinfo.value, PlatformError)


def test_get_organization_deduplicates_name_and_slug_match(
    client: PlatformClient,
    fake: FakeConsole,
):
    fake.orgs[1]["name"] = "acme"

    org = client.get_organization("acme")

    assert org["id"] == 20


def test_invite_to_organization_resolves_name_to_numeric_backend_route(
    client: PlatformClient,
    fake: FakeConsole,
):
    invited = client.invite_to_organization("Acme", "user@example.com")

    assert invited["success"] is True
    invite_request = next(
        request
        for request in reversed(fake.requests)
        if request[0] == "POST" and request[1].endswith("/invite/")
    )
    assert invite_request[1].endswith("/console/api/orgs/20/invite/")


def test_list_environments_and_create_environment(client: PlatformClient, fake: FakeConsole):
    client.set_context("organization", 10)
    envs = client.list_environments()
    assert {e["name"] for e in envs} == {"example-space", "staging"}

    created = client.create_environment("new-env")
    assert created["name"] == "new-env"
    assert any(e["name"] == "new-env" for e in client.list_environments())


def test_budget_get_and_set(client: PlatformClient, fake: FakeConsole):
    budget = client.get_budget()
    assert budget == {
        "total_budget": 100.0,
        "current_usage": 25.0,
        "remaining_budget": 75.0,
        "currency": "USD",
    }

    updated = client.set_budget(250.0)
    assert updated["workspace_budget"] == 250.0
    assert fake.budget["total_budget"] == 250.0


def test_list_workspaces_composes_personal_and_orgs_and_restores_context(
    client: PlatformClient, fake: FakeConsole
):
    # Start in org context
    assert fake.session["account_type"] == "organization"
    assert fake.session["account_id"] == 10

    workspaces = client.list_workspaces()
    paths = sorted(ws.path for ws in workspaces)
    assert paths == sorted(
        [
            "astewart/default",
            "astewart/sandbox",
            "biolm/example-space",
            "biolm/staging",
            "acme/prod",
        ]
    )

    # Original context restored
    assert fake.session["account_type"] == "organization"
    assert fake.session["account_id"] == 10
    assert fake.session["environment_id"] == 100

    # Cookie jar retained across switches (multiple POSTs to account-context)
    context_posts = [r for r in fake.requests if r[0] == "POST" and r[1].endswith("/account-context/")]
    assert len(context_posts) >= 3  # personal + orgs + restore


def test_list_workspaces_personal_label_fallback(fake: FakeConsole):
    fake.personal_details = {"id": 1, "workspace_budget": 0}  # no username
    transport = httpx.MockTransport(fake.handler)
    client = PlatformClient(api_key="tok", base_url=BASE + "/", transport=transport)
    workspaces = client.list_workspaces()
    personal = [ws for ws in workspaces if ws.account_type == "user"]
    assert personal
    assert all(ws.account == "personal" for ws in personal)


def test_current_workspace(client: PlatformClient):
    ws = client.current_workspace()
    assert isinstance(ws, Workspace)
    assert ws.path == "biolm/example-space"
    assert ws.account_type == "organization"
    assert ws.account_id == 10
    assert ws.environment_id == 100


def test_current_workspace_prefers_default_when_context_has_no_environment(
    client: PlatformClient, fake: FakeConsole
):
    fake._default_session["environment_id"] = None
    fake.envs[("organization", 10)] = [
        {"id": 101, "name": "staging", "is_default": False},
        {"id": 100, "name": "example-space", "is_default": True},
    ]

    ws = client.current_workspace()

    assert ws.environment_id == 100
    assert ws.path == "biolm/example-space"


def test_get_workspace_resolves_exact_path(client: PlatformClient, fake: FakeConsole):
    before = dict(fake.session)

    workspace = client.get_workspace("acme/prod")

    assert workspace.path == "acme/prod"
    assert workspace.account_id == 20
    assert fake.session == before


def test_get_workspace_not_found_and_ambiguous(
    client: PlatformClient, fake: FakeConsole
):
    with pytest.raises(WorkspaceNotFoundError):
        client.get_workspace("missing/nope")

    fake.envs[("organization", 20)] = [
        {"id": 200, "name": "staging", "is_default": True},
    ]
    fake.orgs[1]["slug"] = "biolm"

    with pytest.raises(AmbiguousWorkspaceError):
        client.get_workspace("biolm/staging")


def test_switch_workspace_by_path_and_object(client: PlatformClient, fake: FakeConsole):
    ws = client.switch_workspace("astewart/sandbox")
    assert ws.path == "astewart/sandbox"
    assert fake.session["account_type"] == "user"
    assert fake.session["environment_id"] == 2

    target = Workspace(
        account_type="organization",
        account_id=20,
        environment_id=200,
        account="acme",
        environment="prod",
    )
    switched = client.switch_workspace(target)
    assert switched.path == "acme/prod"
    assert fake.session["account_id"] == 20
    assert fake.session["environment_id"] == 200


def test_switch_workspace_not_found_and_ambiguous(client: PlatformClient, fake: FakeConsole):
    with pytest.raises(WorkspaceNotFoundError):
        client.switch_workspace("missing/nope")

    # Force duplicate paths across accounts
    fake.envs[("organization", 20)] = [
        {"id": 200, "name": "staging", "is_default": True},
    ]
    fake.orgs[1]["slug"] = "biolm"  # collide with org 10 slug+env staging
    # Actually make personal path collide: rename acme slug to biolm won't work for same account segment.
    # Duplicate biolm/staging by giving org 20 slug biolm — list will have two biolm/staging
    with pytest.raises(AmbiguousWorkspaceError):
        client.switch_workspace("biolm/staging")


def test_create_workspace_under_current_and_named_account(
    client: PlatformClient, fake: FakeConsole
):
    # Under current (org 10)
    ws = client.create_workspace("Lab Data")
    assert ws.account == "biolm"
    assert ws.environment == "lab-data"
    assert ws.path == "biolm/lab-data"
    assert ws.account_type == "organization"
    assert fake.session["account_id"] == 10  # context unchanged

    # Under different account slug — create then restore
    before = dict(fake.session)
    ws2 = client.create_workspace("dev", account="acme")
    assert ws2.path == "acme/dev"
    assert ws2.account_type == "organization"
    assert ws2.account_id == 20
    assert fake.session == before  # restored

    # Under personal account label
    ws3 = client.create_workspace("scratch", account="astewart")
    assert ws3.path == "astewart/scratch"
    assert ws3.account_type == "user"
    assert fake.session == before


def test_http_error_raises_platform_error(client: PlatformClient, fake: FakeConsole):
    fake.fail_status = 403
    fake.fail_body = {"error": "forbidden", "message": "nope"}
    with pytest.raises(PlatformError) as excinfo:
        client.list_organizations()
    err = excinfo.value
    assert err.status_code == 403
    assert "forbidden" in str(err).lower() or "nope" in str(err).lower()


def test_usage_summary_sends_year_month_and_environment_query(
    client: PlatformClient, fake: FakeConsole
):
    result = client.get_usage_summary(
        year=2025,
        month=6,
        environment_id=200,
    )

    assert result["selected_year"] == 2025
    assert result["selected_month"] == 6
    assert result["filter_env_id"] == 200
    assert fake.usage_queries == [{"year": "2025", "month": "6", "env": "200"}]


def test_usage_summary_omits_unspecified_query_values(
    client: PlatformClient, fake: FakeConsole
):
    result = client.get_usage_summary()

    assert result["account_type"] == "organization"
    assert result["account_id"] == 10
    assert fake.usage_queries == [{}]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"year": 0}, "year"),
        ({"month": 0}, "month"),
        ({"month": 13}, "month"),
        ({"environment_id": 0}, "environment_id"),
    ],
)
def test_usage_summary_rejects_invalid_ranges_before_request(
    client: PlatformClient,
    fake: FakeConsole,
    kwargs: Dict[str, int],
    message: str,
):
    before = len(fake.requests)

    with pytest.raises(ValueError, match=message):
        client.get_usage_summary(**kwargs)

    assert len(fake.requests) == before


def test_usage_summary_for_named_org_switches_and_restores_context(
    client: PlatformClient, fake: FakeConsole
):
    before = dict(fake.session)

    result = client.get_usage_summary(account="acme")

    assert result["account_type"] == "organization"
    assert result["account_id"] == 20
    assert fake.session == before


def test_usage_summary_for_personal_account_switches_and_restores_context(
    client: PlatformClient, fake: FakeConsole
):
    before = dict(fake.session)

    result = client.get_usage_summary(account="astewart")

    assert result["account_type"] == "user"
    assert result["account_id"] == 1
    assert fake.session == before


def test_usage_summary_restores_context_after_request_error(
    client: PlatformClient, fake: FakeConsole
):
    before = dict(fake.session)
    fake.usage_fail_status = 503

    with pytest.raises(PlatformError):
        client.get_usage_summary(account="acme")

    assert fake.session == before


def test_usage_summary_unknown_account_raises_and_restores_context(
    client: PlatformClient, fake: FakeConsole
):
    before = dict(fake.session)

    with pytest.raises(WorkspaceNotFoundError):
        client.get_usage_summary(account="does-not-exist")

    assert fake.session == before


def test_create_api_key_uses_current_context_and_returns_secret(
    client: PlatformClient, fake: FakeConsole
):
    # Default fake context is org 10.
    result = client.create_api_key()

    assert set(result) == {"token"}
    assert result["token"]
    assert len(fake.api_keys) == 1
    assert fake.api_keys[0]["owner_type"] == "organization"
    assert fake.api_keys[0]["owner_id"] == 10
    assert fake.api_keys[0]["token"] == result["token"]


def test_create_api_key_for_personal_account_switches_and_restores(
    client: PlatformClient, fake: FakeConsole
):
    before = dict(fake.session)

    result = client.create_api_key(account="astewart")

    assert result["token"]
    assert fake.api_keys[0]["owner_type"] == "user"
    assert fake.api_keys[0]["owner_id"] == 1
    assert fake.session == before  # context restored


def test_create_api_key_for_org_account_scopes_ownership_and_restores(
    client: PlatformClient, fake: FakeConsole
):
    before = dict(fake.session)

    result = client.create_api_key(account="acme")

    assert result["token"]
    assert fake.api_keys[0]["owner_type"] == "organization"
    assert fake.api_keys[0]["owner_id"] == 20
    assert fake.session == before  # context restored


def test_create_api_key_unknown_account_raises(client: PlatformClient):
    with pytest.raises(WorkspaceNotFoundError):
        client.create_api_key(account="does-not-exist")


def test_delete_api_key_accepts_full_token(client: PlatformClient, fake: FakeConsole):
    created = client.create_api_key()
    token = created["token"]

    result = client.delete_api_key(token)

    assert result is None
    assert fake.api_keys == []


def test_delete_api_key_accepts_eight_char_prefix(
    client: PlatformClient, fake: FakeConsole
):
    created = client.create_api_key()
    prefix = created["token"][:8]

    client.delete_api_key(prefix)

    assert fake.api_keys == []


def test_delete_api_key_blank_fails_before_request(
    client: PlatformClient, fake: FakeConsole
):
    with pytest.raises(ValueError):
        client.delete_api_key("   ")

    delete_reqs = [r for r in fake.requests if r[0] == "DELETE"]
    assert not delete_reqs


def test_delete_api_key_missing_raises_platform_error(client: PlatformClient):
    with pytest.raises(PlatformError):
        client.delete_api_key("nomatch0")


def test_exports_from_package_and_workspaces_compat():
    import biolm
    from biolm import AmbiguousWorkspaceError as AWE
    from biolm import PlatformClient as PC
    from biolm import PlatformError as PE
    from biolm import Workspace as WS
    from biolm import WorkspaceNotFoundError as WNFE
    from biolm.workspaces import Workspace as WS2
    from biolm.workspaces import PlatformClient as PC2

    assert PC is PlatformClient
    assert PE is PlatformError
    assert WNFE is WorkspaceNotFoundError
    assert AWE is AmbiguousWorkspaceError
    assert WS is Workspace
    assert WS2 is Workspace
    assert PC2 is PlatformClient
    assert hasattr(biolm, "Volume")
