# Orphaned and Duplicate Doc Pages

Summary of pages that are not in the sidebar (orphaned) or duplicated under different paths.

## Orphaned – not in any toctree

These files are never listed in a toctree in `index.rst`, so they do not appear in the sidebar. They may still be reachable via `:doc:` links.

| File | Notes |
|------|--------|
| `api-reference/index.rst` | Removed from sidebar; still linked as `:doc:\`api-reference/index\`` from sdk pages. Builds a separate entry point to the same API reference as `sdk/api-reference/index`. |
| `resources/guides/notebooks.rst` | Removed from toctree; same content as `tutorials_use_cases/notebooks.rst`. |
| `cli/overview.rst` | Not in index toctree. Referenced in authoring-guide. |
| `cli/usage/authenticating.rst` | Not in index toctree. |
| `cli/usage/workspaces.rst` | Not in index toctree. |
| `cli/usage/models.rst` | Not in index toctree. |
| `cli/usage/protocols.rst` | Not in index toctree. |
| `cli/usage/datasets.rst` | Not in index toctree. |
| `authoring-guide.rst` | Not in toctree; linked from index body only (intentional). |

## Duplicate content under different paths

| Path A | Path B | Notes |
|--------|--------|--------|
| `api-reference/index.rst` | `sdk/api-reference/index.rst` | Both are entry points to the same API reference (both include `api-reference/modules`). Only the SDK one is in the sidebar now. |
| `tutorials_use_cases/notebooks.rst` | `resources/guides/notebooks.rst` | Same body text (link to jupyter.biolm.ai). First is in toctree; second is orphaned. |
| `biolm.io.rst` (root) | `api-reference/biolm.io.rst` | Identical autodoc content. Only `api-reference/*` is used by the API reference toctree. |
| `biolm.core.rst` (root) | `api-reference/biolm.core.rst` | Same. |
| `biolm.core.legacy.rst` (root) | `api-reference/biolm.core.legacy.rst` | Same. |

## Recommendation

- **Safe to delete** (orphaned duplicates, unused by any toctree):  
  `docs/biolm.io.rst`, `docs/biolm.core.rst`, `docs/biolm.core.legacy.rst`, `docs/resources/guides/notebooks.rst`.
- **Keep but be aware**: `api-reference/index.rst` is still the target of `:doc:\`api-reference/index\`` from several sdk docs; leave it in place or update those links to `sdk/api-reference/index` and then you could remove `api-reference/index.rst` if you want a single entry point.
- **cli/usage and cli/overview**: Add them to a toctree (e.g. under CLI with `:maxdepth: 2`) if you want them in the sidebar; otherwise they remain authoring/usage-only.
