"""Orchestration entrypoint for local protocol execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from biolm.protocols.runtime.backends.local import LocalRunResult, run_local
from biolm.protocols.runtime.compile import compile_protocol, compile_to_pipeline


def run_local_protocol(
    protocol: dict,
    inputs: Optional[dict[str, Any]] = None,
    *,
    output_dir: Optional[str | Path] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> LocalRunResult:
    """Run a protocol dict locally (main public API)."""
    return run_local(
        protocol,
        inputs or {},
        output_dir=output_dir,
        verbose=verbose,
        **kwargs,
    )


__all__ = [
    "LocalRunResult",
    "compile_protocol",
    "compile_to_pipeline",
    "run_local_protocol",
]
