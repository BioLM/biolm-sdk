"""Authentication backends for biolm server."""
from __future__ import annotations

from typing import Optional, Protocol

from starlette.requests import Request
from starlette.responses import JSONResponse


class AuthBackend(Protocol):
    def validate(self, request: Request) -> Optional[JSONResponse]:
        """Return a JSONResponse on auth failure, or None if authorized."""


class NoneAuth:
    def validate(self, request: Request) -> Optional[JSONResponse]:
        return None


class TokenAuth:
    def __init__(self, token: str):
        self.token = token

    def validate(self, request: Request) -> Optional[JSONResponse]:
        auth = request.headers.get("Authorization", "")
        expected = f"Token {self.token}"
        if auth != expected:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing server token.", "status_code": 401},
            )
        return None


def build_auth_backend(mode: str, token: str) -> AuthBackend:
    if mode == "token":
        return TokenAuth(token)
    return NoneAuth()
