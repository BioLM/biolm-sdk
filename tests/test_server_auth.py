"""Tests for biolm server authentication."""
import pytest
from starlette.requests import Request

from biolm.server.auth import NoneAuth, TokenAuth


def _make_request(auth_header: str = "") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"authorization", auth_header.encode())] if auth_header else [],
    }
    return Request(scope)


def test_none_auth_allows():
    assert NoneAuth().validate(_make_request()) is None


def test_token_auth_valid():
    backend = TokenAuth("secret")
    assert backend.validate(_make_request("Token secret")) is None


def test_token_auth_invalid():
    backend = TokenAuth("secret")
    resp = backend.validate(_make_request("Token wrong"))
    assert resp is not None
    assert resp.status_code == 401
