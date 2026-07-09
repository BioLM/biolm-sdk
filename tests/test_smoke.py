import pytest


pytestmark = pytest.mark.smoke


def test_import_biolm():
    import biolm  # noqa: F401


def test_import_cli_entrypoint():
    from biolm.cli.entry import cli  # noqa: F401


def test_import_pipeline_optional_modules_do_not_crash_import():
    # Pipeline subpackage should import even if some optional deps are missing.
    import biolm.pipeline  # noqa: F401

