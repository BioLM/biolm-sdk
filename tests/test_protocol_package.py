"""Tests for biolm.protocols package structure and compatibility shims."""

from __future__ import annotations


class TestProtocolPackageExports:
    def test_validation_exports(self):
        from biolm.protocols import (
            PROTOCOL_SCHEMA_PATH,
            ProtocolValidationResult,
            ValidationError,
            validate_protocol_file,
        )

        assert PROTOCOL_SCHEMA_PATH.endswith("protocol_schema.json")
        assert callable(validate_protocol_file)

    def test_hosted_exports(self):
        from biolm.protocols import (
            ProtocolClient,
            ProtocolNotFoundError,
            ProtocolRun,
            ProtocolRunError,
        )

        assert ProtocolClient is not None
        assert issubclass(ProtocolRunError, Exception)

    def test_model_export(self):
        from biolm.protocols import Protocol

        assert hasattr(Protocol, "validate")
        assert hasattr(Protocol, "execute")

    def test_runtime_lazy_export(self):
        from biolm.protocols import run_local_protocol

        assert callable(run_local_protocol)


class TestProtocolCompatShims:
    def test_protocol_runs_shim(self):
        from biolm.protocol_runs import ProtocolClient as shim_client
        from biolm.protocols.runs import ProtocolClient as canonical_client

        assert shim_client is canonical_client

    def test_protocol_runtime_shim(self):
        from biolm.protocol_runtime import (
            LocalRunResult as shim_result,
            run_local_protocol as shim_run,
        )
        from biolm.protocols.runtime import (
            LocalRunResult as canonical_result,
            run_local_protocol as canonical_run,
        )

        assert shim_run is canonical_run
        assert shim_result is canonical_result

    def test_top_level_run_local_protocol_when_pipeline_installed(self):
        import biolm

        if not getattr(biolm, "_HAS_PIPELINE", False):
            return
        from biolm import run_local_protocol as top_level
        from biolm.protocols.runtime import run_local_protocol as canonical

        assert top_level is canonical


class TestIntegrationTestSelection:
    def test_protocol_runtime_change_selects_runtime_tests(self):
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "select_integration_tests.py"
        out = subprocess.check_output(
            [sys.executable, str(script), "biolm/protocols/runtime/compile.py"],
            text=True,
        ).strip()
        assert "tests/test_protocol_runtime.py" in out.split()

    def test_protocol_validation_change_selects_protocol_tests(self):
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "select_integration_tests.py"
        out = subprocess.check_output(
            [sys.executable, str(script), "biolm/protocols/validation.py"],
            text=True,
        ).strip()
        assert "tests/test_protocols.py" in out.split()
