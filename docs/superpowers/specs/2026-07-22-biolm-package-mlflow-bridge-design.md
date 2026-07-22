# BioLM Package ↔ MLflow Bridge Design

Addendum to [2026-07-21-biolm-definition-design.md](2026-07-21-biolm-definition-design.md).

## Purpose

Connect locked BioLM packages to MLflow serving and Modal deploy without collapsing BioLM’s multi-action API into a single anonymous `predict`. The locked `BioLM` manifest remains the source of truth for **actions** and **weight load policy**; `mlflow-biolm` owns the MLflow flavor, pyfunc, and deploy plugin.

## Ownership

| Package | Owns |
|---------|------|
| **biolm-sdk** | Recipe → `build` / `--bundle` → `BioLM` package |
| **mlflow-biolm** | `BioLMPyfuncModel`, canonical schemas, export to `MLmodel`, `get_deploy_client("biolm")` |
| **biolm-modal** | Modal runtime / deployment API; **re-exports** pyfunc from `mlflow-biolm` (does not own the plugin long-term) |

## Weight load modes

| Mode | Meaning | Default |
|------|---------|---------|
| `lazy` | Resolve at runtime (catalog slug, remote URI) | `from.load` |
| `preload` | Materialize into package `artifacts/` at bundle/export | head `artifact.load` |

```yaml
from:
  slug: esm2-8m
  load: lazy
layers:
  - type: embedding_head
    artifact:
      load: preload
      uri: https://…/model.joblib   # optional until resolved
      path: ./artifacts/model.joblib  # set after --bundle
```

Foundation bases are never bundled in v0. Heads are preloaded when `biolm model build --bundle` (or export materializes a URI).

## Actions as translation

```text
BioLM.actions  →  BioLMPyfuncModel.predict({"action","data"})  →  Modal POST /{action}
```

Package `actions` are authoritative for deploy. Recipe `actions` (if present) must include at least `encode` and `predict`; build merges them and adds schema refs:

- `biolm.encode.v1`
- `biolm.predict.v1`

## Export / deploy flow

```text
~/.biolm/models/<name>/<tag>/BioLM
        │ export (mlflow-biolm)
        ▼
output/
  BioLM          # copy
  MLmodel
  artifacts/
  python_model / code
        │ get_deploy_client("biolm")
        ▼
Modal: /encode, /predict  (actions from BioLM if config.actions omitted)
```

## Non-goals

Hub custom deploy product; preloading foundation weights; DSM/LoRA layers.
