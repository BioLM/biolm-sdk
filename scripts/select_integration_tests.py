#!/usr/bin/env python3
"""Select integration test files based on changed source paths.

Usage:
  python scripts/select_integration_tests.py biolm/pipeline/data.py tests/foo.py
  git diff --name-only origin/main...HEAD | xargs python scripts/select_integration_tests.py

Prints a space-separated list of test file paths. If no area matches, prints
nothing (caller should fall back to the full integration tier).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AREAS_PATH = ROOT / ".github" / "test-areas.yml"


def load_areas() -> dict:
    data = yaml.safe_load(AREAS_PATH.read_text(encoding="utf-8"))
    return data.get("areas", {})


def area_for_path(path: str, areas: dict) -> str | None:
    normalized = path.replace("\\", "/")
    for area_name, spec in areas.items():
        for prefix in spec.get("sources", []):
            prefix = prefix.rstrip("/")
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return area_name
    return None


def main(argv: list[str]) -> int:
    changed = [p for p in argv if p.strip()]
    areas = load_areas()

    if not changed:
        return 0

    matched_areas: set[str] = set()
    for path in changed:
        area = area_for_path(path, areas)
        if area:
            matched_areas.add(area)

    # Config / global test changes → run full integration marker suite.
    global_triggers = {
        "tox.ini",
        "pyproject.toml",
        "setup.cfg",
        "requirements_dev.txt",
        ".github/workflows/ci.yml",
        ".github/workflows/nightly.yml",
    }
    if any(p.replace("\\", "/") in global_triggers or p.startswith("tests/") for p in changed):
        print("FULL")
        return 0

    if not matched_areas:
        return 0

    selected: list[str] = []
    for area in sorted(matched_areas):
        selected.extend(areas[area].get("tests", []))

    # Stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for path in selected:
        if path not in seen:
            seen.add(path)
            ordered.append(path)

    print(" ".join(ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
