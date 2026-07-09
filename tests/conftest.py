import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.live (network / live server required).",
    )


def pytest_configure(config):
    # Offline tiers construct BioLMApiClient in mocked tests; avoid requiring real tokens.
    if not config.getoption("--run-live"):
        os.environ.setdefault("BIOLM_TOKEN", "unit-test-token")

    # Ensure markers are registered even if pyproject isn't consulted in some contexts.
    config.addinivalue_line("markers", "smoke: fast sanity checks suitable for PRs")
    config.addinivalue_line("markers", "unit: offline unit tests (default PR tier)")
    config.addinivalue_line("markers", "integration: heavier offline integration/E2E tests")
    config.addinivalue_line("markers", "live: requires network and a live BioLM server/token")
    config.addinivalue_line("markers", "slow: long-running tests (benchmarks, large datasets)")


def pytest_collection_modifyitems(config, items):
    run_live = bool(config.getoption("--run-live"))
    # Back-compat: if CI exports a token, that is *necessary* but not *sufficient* to
    # run live tests. We still require an explicit opt-in to avoid accidental slow PR CI.
    has_token = bool(os.environ.get("BIOLM_TOKEN") or os.environ.get("BIOLMAI_TOKEN"))

    if run_live:
        if not has_token:
            raise pytest.UsageError(
                "--run-live was passed but BIOLM_TOKEN/BIOLMAI_TOKEN is not set."
            )
        return

    skip_live = pytest.mark.skip(
        reason="Skipped live tests (pass --run-live and set BIOLM_TOKEN/BIOLMAI_TOKEN)."
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)

