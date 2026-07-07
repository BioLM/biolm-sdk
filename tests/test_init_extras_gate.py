"""
Tests for the missing-extras ImportError gate in biolm/pipeline/__init__.py.

Covers D4:
- The gate fires when optional deps are absent.
- The error message names the missing package(s).
- The error message includes the pip install command.
- Normal import succeeds when all deps are present.

IMPORTANT: Each test meticulously cleans sys.modules before and after so the
deliberately-broken import cannot poison the real biolm.pipeline namespace
used by other tests in the session.
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager

import pytest


_PIPELINE_PREFIXES = ("biolm.pipeline", "biolmai.pipeline")


def _pipeline_module_keys() -> list[str]:
    keys: list[str] = []
    for prefix in _PIPELINE_PREFIXES:
        for key in sys.modules:
            if key == prefix or key.startswith(prefix + "."):
                keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _isolated_pipeline_import(**fake_modules):
    """
    Context manager that:
    1. Removes biolm.pipeline (and sub-modules) from sys.modules.
    2. Injects ``None`` sentinel(s) for the named packages (simulating missing).
    3. Attempts to import biolm.pipeline — yields the ImportError if raised,
       or yields None if the import succeeded.
    4. Always restores sys.modules to its pre-test state.
    """
    saved = {k: v for k, v in sys.modules.items() if k in _pipeline_module_keys()}
    saved_fakes = {name: sys.modules.get(name, _SENTINEL) for name in fake_modules}

    try:
        for key in _pipeline_module_keys():
            del sys.modules[key]

        for name, val in fake_modules.items():
            sys.modules[name] = val

        err = None
        try:
            importlib.import_module("biolm.pipeline")
        except ImportError as exc:
            err = exc
        yield err

    finally:
        for key in _pipeline_module_keys():
            sys.modules.pop(key, None)

        sys.modules.update(saved)

        for name, val in saved_fakes.items():
            if val is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val


_SENTINEL = object()


# ---------------------------------------------------------------------------
# Fixture: ensure biolm.pipeline is available after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_pipeline_after_test():
    """Always re-import biolm.pipeline after each test so the module is valid."""
    yield
    if "biolm.pipeline" not in sys.modules:
        try:
            importlib.import_module("biolm.pipeline")
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMissingExtrasGate:
    """biolm/pipeline/__init__.py raises a helpful ImportError when deps missing."""

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
        if "biolm.pipeline" not in sys.modules:
            importlib.import_module("biolm.pipeline")
        import biolm.pipeline as pipeline

        assert pipeline is not None
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
            assert isinstance(err, ImportError)
            assert "biolm.pipeline requires optional dependencies" in str(err), (
                f"Gate message not found. Got: {err}"
            )

    def test_error_message_mentions_opt_in(self):
        """Error message explains the pipeline package is opt-in."""
        with _isolated_pipeline_import(numpy=None) as err:
            assert err is not None
            assert "numpy" in str(err)
            assert "biolm.pipeline requires" in str(err)
