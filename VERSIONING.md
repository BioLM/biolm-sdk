# Version Management Guide

## Overview

This project uses **[python-semantic-release](https://python-semantic-release.readthedocs.io/)** to manage versions from [Conventional Commits](https://www.conventionalcommits.org/) on the `main` branch.

**Key principles:**

- **Production versions**: CI bumps `biolm/__version__` and creates a Git tag (e.g. `v0.5.0`) on `main`
- **PyPI publish**: Triggered when a GitHub Release is published (see `.github/workflows/publish.yml`)
- **Release token**: The CI `version` job must use `SEMANTIC_RELEASE_TOKEN` (PAT or GitHub App token with Contents write), not `GITHUB_TOKEN`. Releases created with `GITHUB_TOKEN` do not trigger other workflows, so `publish.yml` never runs.
- **Non-conventional commits**: Ignored for versioning (no CI failure)
- **TestPyPI**: Use `-rc` suffixes manually when needed (e.g. `0.5.0-rc.1`), or `workflow_dispatch` on the Publish workflow

## How It Works

1. **Commit with conventional format** (e.g. `feat:`, `fix:`, `BREAKING:`) or squash-merge a PR whose title includes them (e.g. `PD-60 feat: ...`)
2. **Push to `main`**
3. **CI runs python-semantic-release** which:
   - Analyzes commit messages since the last tag
   - Determines version bump (major/minor/patch)
   - Updates `biolm/__init__.py` and `pyproject.toml` (`project.version`)
   - Creates git tag (e.g. `v0.5.0`) via the deploy key (`SEMANTIC_RELEASE_SSH_KEY`)
   - Opens a GitHub Release via `SEMANTIC_RELEASE_TOKEN` (so `release: published` can trigger publish)
   - Commits the version bump with `[skip ci]` to avoid loops
4. **Publish workflow** uploads to PyPI when the GitHub Release is published

### Required secrets

| Secret | Purpose |
|--------|---------|
| `SEMANTIC_RELEASE_SSH_KEY` | Deploy key for pushing the version-bump commit + tag to `main` |
| `SEMANTIC_RELEASE_TOKEN` | PAT/App token for the GitHub Releases API (must not be `GITHUB_TOKEN`) |

If a Release already exists on GitHub but is missing from PyPI, backfill with **Actions → Publish to PyPI → Run workflow**, choose the tag ref (e.g. `v1.5.0`) and repository `pypi`.

## Commit Message Format

| Commit Type | Version Bump | Example |
|------------|--------------|---------|
| `feat: ...` | **Minor** (0.4.0 → 0.5.0) | `feat: add protocol batch API` |
| `fix: ...` | **Patch** (0.4.0 → 0.4.1) | `fix: resolve auth timeout` |
| `BREAKING: ...` | **Major** (0.4.0 → 1.0.0) | `BREAKING: rename package to biolm-sdk` |
| Other / no prefix | **None** | `update readme`, `PD-60 wip` |

Ticket prefixes are fine when combined with a conventional type: `PD-60 feat: add versioning` or `feat(PD-60): add versioning`.

## What NOT to Do

**Do not manually edit `__version__` in `biolm/__init__.py` for production releases.** That causes mismatches with git tags and PyPI.

```bash
# BAD — do not do this for production
vim biolm/__init__.py  # hand-edit __version__
```

## Correct Workflow

### Regular releases

1. Make changes and merge to `main` with a conventional commit (or squash PR title).
2. Wait for CI: tests, then the `version` job on `main`.
3. Confirm the new tag and GitHub Release (e.g. `v0.5.0`).
4. The **Publish to PyPI** workflow runs on release publish and uploads the package.

### Docs on GitHub Pages

Docs deploy from **`main`** on push (`.github/workflows/docs.yml`).

### Local testing

You do not need to bump the version for local testing. Build and test at the current version:

```bash
pip install -e .
pytest -q tests/
```

### TestPyPI (release candidates)

For pre-release uploads without conflicting with the next production version:

- Use **workflow_dispatch** on the Publish workflow (TestPyPI) after configuring trusted publishing on TestPyPI, or
- Publish an `-rc` GitHub Release (routes to TestPyPI per `publish.yml`).

## Changelog

Release notes are maintained in **[CHANGELOG.md](CHANGELOG.md)** at the repo root.
python-semantic-release updates this file on each version bump (``feat``/``fix``/``BREAKING`` commits).

Configuration: ``[tool.semantic_release.changelog]`` in ``pyproject.toml``. The
insertion flag ``<!-- version list -->`` must remain in CHANGELOG.md for update mode.

Docs mirror: ``docs/reference/changelog.rst`` includes the same file.

Dry-run changelog output:

```bash
semantic-release changelog --print
```

## Configuration

See `[tool.semantic_release]` in `pyproject.toml`. The canonical version lives in `biolm/__init__.py` (also written to `pyproject.toml` by CI). `setup.py` reads `__init__.py`; `docs/conf.py` imports `biolm`.

## Troubleshooting

```bash
# Current package version
python -c "import biolm; print(biolm.__version__)"

# Latest tag
git describe --tags --abbrev=0

# Dry-run (requires python-semantic-release installed)
semantic-release version --print
```

If `package` version and git tags disagree, align with the latest tag and let the next conventional commit on `main` produce a correct release.
