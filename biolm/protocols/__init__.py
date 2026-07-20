"""BioLM Protocols — validation, hosted runs, and local execution."""

from biolm.protocols.model import Protocol
from biolm.protocols.runs import (
    ProtocolClient,
    ProtocolNotFoundError,
    ProtocolRun,
    ProtocolRunError,
)
from biolm.protocols.validation import (
    PROTOCOL_SCHEMA_PATH,
    ProtocolValidationResult,
    ValidationError,
    validate_protocol_file,
)

__all__ = [
    "PROTOCOL_SCHEMA_PATH",
    "Protocol",
    "ProtocolClient",
    "ProtocolNotFoundError",
    "ProtocolRun",
    "ProtocolRunError",
    "ProtocolValidationResult",
    "ValidationError",
    "validate_protocol_file",
]


def __getattr__(name: str):
    """Lazy export of pipeline-gated local runtime symbols."""
    if name in (
        "LocalRunResult",
        "UnsupportedProtocolFeature",
        "compile_protocol",
        "compile_to_pipeline",
        "run_local_protocol",
    ):
        from biolm.protocols import runtime as _runtime

        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
