"""Unit tests for docs manifest kind inference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing the script as a module without packaging it.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_docs_manifest import collect_item_slugs, infer_kind  # noqa: E402


class TestInferKind:
    def test_guide(self):
        assert infer_kind(["guide/quickstart", "guide/concepts"]) == "guide"

    def test_reference_sdk_cli_yaml(self):
        assert infer_kind(["sdk/models", "cli/login", "yaml/protocol-schema"]) == "reference"

    def test_notes_changelog_and_notes_path(self):
        assert infer_kind(["changelog", "notes/migration-1.0"]) == "notes"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no slugs"):
            infer_kind([])

    def test_unknown_slug_raises(self):
        with pytest.raises(ValueError, match="Unclassified"):
            infer_kind(["orphan/page"])

    def test_mixed_kinds_raises(self):
        with pytest.raises(ValueError, match="mixes kinds"):
            infer_kind(["guide/quickstart", "sdk/models"])


class TestCollectItemSlugs:
    def test_nested_children(self):
        items = [
            {
                "slug": "sdk/models",
                "title": "Models",
                "children": [{"slug": "sdk/models/examples", "title": "Examples"}],
            },
            {"slug": "cli/login", "title": "Login"},
        ]
        assert collect_item_slugs(items) == [
            "sdk/models",
            "sdk/models/examples",
            "cli/login",
        ]
