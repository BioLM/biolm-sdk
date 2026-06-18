"""
Tests for the missing-extras ImportError gate in biolmai/pipeline/__init__.py.

Covers D4:
- The gate fires when optional deps are absent.
- The error message names the missing package(s).
- The error message includes the pip install command.
- Normal import succeeds when all deps are present.

IMPORTANT: Each test meticulously cleans sys.modules before and after so the
deliberately-broken import cannot poison the real biolmai.pipeline namespace
used by other tests in the session.
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _isolated_pipeline_import(**fake_modules):
    """
    Context manager that:
    1. Removes biolmai.pipeline (and sub-modules) from sys.modules.
    2. Injects ``None`` sentinel(s) for the named packages (simulating missing).
    3. Attempts to import biolm.pipeline — yields the ImportError if raised,
       or yields None if the import succeeded.
    4. Always restores sys.modules to its pre-test state.
    """
    # Snapshot all current biolmai.pipeline entries
    saved = {k: v for k, v in sys.modules.items()
             if k == "biolmai.pipeline" or k.startswith("biolmai.pipeline.")}

    # Also snapshot the packages we're going to fake-absent
    saved_fakes = {name: sys.modules.get(name, _SENTINEL) for name in fake_modules}

    try:
        # Evict all biolmai.pipeline sub-modules so Python re-executes __init__.py
        for k in list(sys.modules):
            if k == "biolmai.pipeline" or k.startswith("biolmai.pipeline."):
                del sys.modules[k]

        # Inject None so `import <name>` raises ImportError
        for name, val in fake_modules.items():
            sys.modules[name] = val  # None → ImportError on `import name`

        err = None
        try:
            importlib.import_module("biolmai.pipeline")
        except ImportError as exc:
            err = exc
        yield err

    finally:
        # --- Restore ---
        # Remove anything that was imported during the test
        for k in list(sys.modules):
            if k == "biolmai.pipeline" or k.startswith("biolmai.pipeline."):
                del sys.modules[k]

        # Restore original biolmai.pipeline modules (including the real one)
        sys.modules.update(saved)

        # Restore faked packages
        for name, val in saved_fakes.items():
            if val is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val


_SENTINEL = object()  # marker for "was not in sys.modules"


# ---------------------------------------------------------------------------
# Fixture: ensure biolmai.pipeline is available after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_pipeline_after_test():
    """Always re-import biolm.pipeline after each test so the module is valid."""
    yield
    # If biolmai.pipeline was evicted (e.g. by an isolation context) and not yet
    # restored, import it fresh so the next test in the session can use it.
    if "biolmai.pipeline" not in sys.modules:
        try:
            importlib.import_module("biolmai.pipeline")
        except ImportError:
            pass  # If deps truly missing, nothing we can do


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMissingExtrasGate:
    """biolmai/pipeline/__init__.py raises a helpful ImportError when deps missing."""

    def test_import_error_message_names_missing_packages(self):
        """Error message includes the name of each missing package."""
        with _isolated_pipeline_import(duckdb=None) as err:
            assert err is not None, "Expected ImportError when duckdb is missing"
            assert "duckdb" in str(err), (
                f"Error message should name 'duckdb' but got: {err}"
            )

    def test_import_error_includes_install_command(self):
        """Error message includes the pip install command for biolm[pipeline]."""
        with _isolated_pipeline_import(duckdb=None) as err:
            assert err is not None, "Expected ImportError when duckdb is missing"
            msg = str(err)
            assert "pip install" in msg, f"Expected pip install hint in: {msg}"
            assert "biolm[pipeline]" in msg, (
                f"Expected 'biolm[pipeline]' in error message but got: {msg}"
            )

    def test_import_error_lists_multiple_missing_packages(self):
        """When multiple deps are missing, all are named in the error message."""
        with _isolated_pipeline_import(duckdb=None, pandas=None) as err:
            assert err is not None, "Expected ImportError when duckdb and pandas are missing"
            msg = str(err)
            assert "duckdb" in msg, f"Expected 'duckdb' in: {msg}"
            assert "pandas" in msg, f"Expected 'pandas' in: {msg}"

    def test_import_succeeds_when_deps_present(self):
        """Smoke test: normal import without any mocking should succeed."""
        # We don't isolate here — just verify the real import still works
        # (this will fail the test suite immediately if deps are broken).
        if "biolmai.pipeline" not in sys.modules:
            importlib.import_module("biolmai.pipeline")
        import biolm.pipeline as pipeline
        assert pipeline is not None
        # Check key symbols are present
        assert hasattr(pipeline, "DataPipeline")
        assert hasattr(pipeline, "ThresholdFilter")
        assert hasattr(pipeline, "StructureSpec")
        assert hasattr(pipeline, "MatrixExtractionSpec")

    def test_gate_fires_before_submodule_import_errors(self):
        """
        The gate should raise a clean ImportError (not AttributeError / ModuleNotFoundError
        from a partially-imported submodule).
        """
        with _isolated_pipeline_import(duckdb=None) as err:
            assert err is not None
            # Must be ImportError (not something else like AttributeError)
            assert isinstance(err, ImportError)
            # Must not be a traceback from a half-imported module — the message
            # should be the human-readable gate message.
            assert "biolmai.pipeline requires optional dependencies" in str(err), (
                f"Gate message not found. Got: {err}"
            )

    def test_error_message_mentions_opt_in(self):
        """Error message explains the pipeline package is opt-in."""
        with _isolated_pipeline_import(numpy=None) as err:
            assert err is not None
            assert "numpy" in str(err)
            # The message should be helpful, not just a raw ImportError from numpy
            assert "biolmai.pipeline requires" in str(err)
