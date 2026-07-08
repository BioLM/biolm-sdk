"""SeqFrame — sequence-centric dataframe abstraction for the BioLM SDK.

SeqFrame provides a high-level abstraction over Parquet for representing and
operating on collections of biological sequences. It is complementary to
``biolm dataset`` (upload/download/versioning): datasets may contain SeqFrames
as artifacts, while SeqFrame handles querying, enrichment, and conversion.

Requires optional dependencies::

    pip install 'biolm-sdk[seqframe]'

Also available when ``biolm-sdk[pipeline]`` is installed (same data stack deps).
"""

from __future__ import annotations

_MISSING = []
for _name in ("duckdb", "pandas", "pyarrow"):
    try:
        __import__(_name)
    except ImportError:
        _MISSING.append(_name)

if _MISSING:
    raise ImportError(
        "biolm.seqframe requires optional dependencies that are not installed: "
        f"{', '.join(_MISSING)}.\n\n"
        "Install with:\n\n"
        "    pip install 'biolm-sdk[seqframe]'\n\n"
        "Or install the pipeline extra (includes the same data dependencies):\n\n"
        "    pip install 'biolm-sdk[pipeline]'"
    )
del _MISSING
try:
    del _name
except NameError:
    pass

from biolm.seqframe.core import SeqFrame
from biolm.seqframe.metadata import SEQFRAME_VERSION, SeqFrameMetadata

__all__ = ["SeqFrame", "SeqFrameMetadata", "SEQFRAME_VERSION"]
