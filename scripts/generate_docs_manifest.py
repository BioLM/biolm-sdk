#!/usr/bin/env python3
"""Generate manifest.json from a Sphinx ``-b json`` build.

The manifest captures navigation (from docs toctrees), per-page metadata,
and a flat slug index so a Next.js site can fetch and render ``.fjson`` pages
without Sphinx HTML/CSS.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SKIP_SLUGS = frozenset(
    {
        "search",
        "genindex",
        "py-modindex",
        "ORPHANS_AND_DUPLICATES",
    }
)
SKIP_SLUG_PREFIXES = ("_modules/",)

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def parse_toctrees(content: str) -> list[dict[str, Any]]:
    """Parse ``.. toctree::`` blocks from RST/MD source."""
    lines = content.splitlines()
    blocks: list[dict[str, Any]] = []
    index = 0

    while index < len(lines):
        if lines[index].strip() != ".. toctree::":
            index += 1
            continue

        index += 1
        options: dict[str, str] = {}
        entries: list[str] = []

        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith(":") and ":" in stripped[1:]:
                key, value = stripped[1:].split(":", 1)
                options[key.strip()] = value.strip()
                index += 1
                continue
            if lines[index].startswith("   ") and stripped and not stripped.startswith(":"):
                entries.append(stripped)
                index += 1
                continue
            break

        blocks.append({"options": options, "entries": entries})

    return blocks


def resolve_doc_slug(entry: str, source_dir: Path, docs_root: Path) -> str:
    """Resolve a toctree entry to a docs slug (path without suffix)."""
    normalized = entry.strip()
    for suffix in (".rst", ".md"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    candidate = (source_dir / normalized).resolve()
    try:
        relative = candidate.relative_to(docs_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Could not resolve toctree entry {entry!r} from {source_dir}") from exc

    # Avoid Path.with_suffix(): it treats ``migration-1.0`` as ``migration-1`` + ``.0``.
    return str(relative).replace("\\", "/")


def load_page_meta(build_dir: Path, slug: str) -> dict[str, Any] | None:
    path = build_dir / f"{slug}.fjson"
    if not path.is_file():
        return None

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    prev = data.get("prev")
    next_page = data.get("next")

    return {
        "title": strip_html(data.get("title") or slug.rsplit("/", 1)[-1]),
        "prev": _nav_link(prev, slug),
        "next": _nav_link(next_page, slug),
    }


def _resolve_relative_slug(href: str, current_slug: str) -> str:
    path = href.strip().strip("/")
    if path.endswith(".html"):
        path = path[: -len(".html")]

    if not path.startswith("..") and not path.startswith("."):
        return path or "index"

    base_parts = current_slug.split("/")
    for part in path.split("/"):
        if part == "..":
            if base_parts:
                base_parts.pop()
        elif part and part != ".":
            base_parts.append(part)

    return "/".join(base_parts) or "index"


def _nav_link(link: dict[str, str] | None, current_slug: str) -> dict[str, str] | None:
    if not link:
        return None

    href = link.get("link", "").strip()
    if not href or href.startswith("http://") or href.startswith("https://"):
        return None

    slug = _resolve_relative_slug(href.split("#")[0], current_slug)
    if not slug:
        return None

    return {"slug": slug, "title": link.get("title", slug)}


def should_include_slug(slug: str) -> bool:
    if slug in SKIP_SLUGS:
        return False
    return not any(slug.startswith(prefix) for prefix in SKIP_SLUG_PREFIXES)


def build_nav_item(
    slug: str,
    build_dir: Path,
    docs_root: Path,
    depth: int,
    max_depth: int,
) -> dict[str, Any] | None:
    meta = load_page_meta(build_dir, slug)
    if meta is None:
        return None

    item: dict[str, Any] = {
        "slug": slug,
        "title": meta["title"],
    }

    if depth >= max_depth:
        return item

    source = _find_source_file(docs_root, slug)
    if source is None:
        return item

    children: list[dict[str, Any]] = []
    for block in parse_toctrees(source.read_text(encoding="utf-8")):
        child_depth = int(block["options"].get("maxdepth", max_depth))
        for entry in block["entries"]:
            child_slug = resolve_doc_slug(entry, source.parent, docs_root)
            child = build_nav_item(
                child_slug,
                build_dir,
                docs_root,
                depth + 1,
                child_depth,
            )
            if child is not None:
                children.append(child)

    if children:
        item["children"] = children

    return item


def _find_source_file(docs_root: Path, slug: str) -> Path | None:
    for suffix in (".rst", ".md"):
        candidate = docs_root / f"{slug}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_navigation(docs_root: Path, build_dir: Path) -> list[dict[str, Any]]:
    index = docs_root / "index.rst"
    if not index.is_file():
        raise FileNotFoundError(f"Missing root toctree file: {index}")

    navigation: list[dict[str, Any]] = []

    for block in parse_toctrees(index.read_text(encoding="utf-8")):
        caption = block["options"].get("caption", "").rstrip(":").strip()
        max_depth = int(block["options"].get("maxdepth", 1))
        items: list[dict[str, Any]] = []

        for entry in block["entries"]:
            slug = resolve_doc_slug(entry, docs_root, docs_root)
            item = build_nav_item(slug, build_dir, docs_root, depth=1, max_depth=max_depth)
            if item is not None:
                items.append(item)

        if items:
            section: dict[str, Any] = {"items": items}
            if caption:
                section["caption"] = caption
            navigation.append(section)

    return navigation


def collect_pages(build_dir: Path) -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}

    for path in sorted(build_dir.rglob("*.fjson")):
        slug = str(path.relative_to(build_dir).with_suffix(""))
        if not should_include_slug(slug):
            continue

        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        prev = _nav_link(data.get("prev"), slug)
        next_page = _nav_link(data.get("next"), slug)

        pages[slug] = {
            "title": strip_html(data.get("title") or slug.rsplit("/", 1)[-1]),
            "prev": prev,
            "next": next_page,
        }

    return pages


def load_global_context(build_dir: Path) -> dict[str, Any]:
    path = build_dir / "globalcontext.json"
    if not path.is_file():
        return {}

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        "project": data.get("project"),
        "version": data.get("version"),
        "release": data.get("release"),
        "copyright": data.get("copyright"),
    }


def generate_manifest(build_dir: Path, docs_root: Path) -> dict[str, Any]:
    if not build_dir.is_dir():
        raise FileNotFoundError(f"JSON build directory does not exist: {build_dir}")

    pages = collect_pages(build_dir)
    navigation = build_navigation(docs_root, build_dir)
    context = load_global_context(build_dir)

    return {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": context.get("project"),
        "package_version": context.get("version"),
        "release": context.get("release"),
        "copyright": context.get("copyright"),
        "nav": navigation,
        "pages": pages,
        "slugs": sorted(pages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("docs/_build/json"),
        help="Sphinx JSON build output directory",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Sphinx documentation source directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest output path (default: <build-dir>/manifest.json)",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    docs_root = args.docs_dir.resolve()
    output = args.output or (build_dir / "manifest.json")

    manifest = generate_manifest(build_dir, docs_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote {output} ({len(manifest['slugs'])} pages)")


if __name__ == "__main__":
    main()
