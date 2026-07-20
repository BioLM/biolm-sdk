"""
Local Protocol Runtime — compile and execute Protocol YAML via biolm.pipeline.

Requires optional dependencies::

    pip install "biolm[pipeline]"
"""

from __future__ import annotations

_MISSING = []
for _name in ("duckdb", "pandas", "jmespath"):
    try:
        __import__(_name)
    except ImportError:
        _MISSING.append(_name)

if _MISSING:
    raise ImportError(
        "biolm.protocols.runtime requires optional dependencies that are not installed: "
        f"{', '.join(_MISSING)}.\n\n"
        "Install with:\n\n"
        "    pip install 'biolm[pipeline]'\n"
    )
del _MISSING
try:
    del _name
except NameError:
    pass

from biolm.protocols.runtime.backends.local import LocalRunResult
from biolm.protocols.runtime.compile import compile_protocol, compile_to_pipeline
from biolm.protocols.runtime.execute import run_local_protocol
from biolm.protocols.runtime.profile import UnsupportedProtocolFeature

__all__ = [
    "LocalRunResult",
    "UnsupportedProtocolFeature",
    "compile_protocol",
    "compile_to_pipeline",
    "run_local_protocol",
]
