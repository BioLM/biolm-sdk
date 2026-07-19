"""Lab-in-the-Loop (LLTP) orchestration for biolm-sdk.

Install::

    pip install 'biolm-sdk[lltp]'

Connectors are separate packages (not yet on PyPI). Install from GitHub::

    pip install "adaptyv-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=adaptyv-lltp/src/py"
    pip install "twist-lltp @ git+https://github.com/BioLM/lltp-connectors.git#subdirectory=twist-lltp/src/py"

Project config lives in ``lltp.yaml``; run state under ``.biolm/lltp/<run_id>.json``.
SeqFrame conversion helpers live on ``sf.lab`` (``to_lltp`` / ``from_lltp`` / ``merge``).
"""

from __future__ import annotations

from biolm.lab.api import confirm, list_runs, results, status, submit
from biolm.lab.config import LabConfig, load_config
from biolm.lab.runs import LabRun, list_run_ids, load_run

__all__ = [
    "LabConfig",
    "LabRun",
    "confirm",
    "list_run_ids",
    "list_runs",
    "load_config",
    "load_run",
    "results",
    "status",
    "submit",
]
