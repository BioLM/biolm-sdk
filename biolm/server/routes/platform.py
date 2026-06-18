"""501 stubs for hosted-platform-only routes."""
from __future__ import annotations

from starlette.responses import JSONResponse

PLATFORM_MSG = "Not supported by biolm server. Use the hosted BioLM platform."


def not_supported(status_code: int = 501) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": PLATFORM_MSG, "status_code": status_code},
    )
