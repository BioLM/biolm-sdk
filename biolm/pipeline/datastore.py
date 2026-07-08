"""
Backward-compatibility shim.

Old code imported from ``biolm.pipeline.datastore``; the implementation has
moved to ``biolm.pipeline.datastore_duckdb``.  Import ``DataStore`` from
either location — they resolve to the same class.
"""

from biolm.pipeline.datastore_duckdb import DuckDBDataStore as DataStore  # noqa: F401

__all__ = ["DataStore"]
