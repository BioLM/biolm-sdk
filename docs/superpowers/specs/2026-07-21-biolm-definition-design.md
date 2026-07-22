# BioLM Definition Design

## Purpose

Introduce a first-class **BioLM definition**: a declarative way to describe how a
biological language model is produced from a base model plus adaptation layers,
and a compiled **package** that Hub, deploy, and (later) MLflow can consume.

Today BioLM has catalog models as opaque slugs and finetuning as imperative
``Finetune.*`` API calls. Protocol YAML composes *inference workflows* over
slugs. There is no artifact that says “this model *is* a base plus these
adaptations,” in a form that is versionable, buildable, and shippable.

This design fills that gap with a Dockerfile-like recipe and an MLflow-like
package, without inventing a Docker daemon or merging model identity into
Protocol YAML.

## Analogies (mental model)

| Concept | BioLM | Docker | dbt | MLflow |
|---------|-------|--------|-----|--------|
| Author source | Recipe ``*.yaml`` | Dockerfile | Model ``.sql`` (templated) | — |
| Build / compile | ``biolm model build`` | ``docker build`` | ``dbt run`` (compiles) | ``mlflow.pyfunc.log_model`` |
| Artifact | Package dir + ``BioLM`` | Image | Compiled SQL in ``target/`` | Model dir + ``MLmodel`` |
| Local store | ``~/.biolm/models/<name>/<tag>/`` | Local image store | ``target/`` | Local tracking / artifact store |
| Publish | ``push`` / ``deploy`` (later) | ``docker push`` | Deploy to warehouse | Registry / UC |

The Docker analogy teaches lifecycle (don’t push the Dockerfile; push what build
produced). The dbt analogy teaches the relationship between the two documents
(source vs compiled). The MLflow analogy teaches on-disk layout and a future
flavor plugin.

Authors declare **intent**; the runner owns **mechanics** (XGBoost vs future
LoRA vs DSM). The package records what actually ran.

## Decisions

- **Two lifecycle artifacts, one schema family:** recipe (editable) and package
  (compiled / locked). Same field vocabulary; a locked package **requires**
  concrete ``actions`` and a successful ``built`` block (including ``run_id``
  on each completed layer). Digests and artifact URIs are **best-effort in
  v0**: write them when the Finetune run payload exposes them; do not fail
  ``build`` solely because a digest is missing.
- **Recipe lives in the project** as a ``.yaml`` file (git-tracked blueprint).
  ``build`` does not rewrite the recipe by default.
- **Package lives in a local registry** (flat files, no daemon):
  ``~/.biolm/models/<name>/<tag>/`` with shouty manifest ``BioLM`` (YAML content,
  **no** ``.yaml`` suffix — tool artifact, not casually edited).
- **CLI verb:** ``biolm model build``.
- **Schema spine:** ``from`` / ``layers`` / ``actions`` ≈ Docker ``FROM`` /
  ``RUN`` / expose-or-entrypoint.
- **Layers are a single list**, enriched on compile into the package — never a
  duplicate ``build.layers`` fork inside one file.
- **Protocols stay separate.** A BioLM definition is not a workflow; protocols
  keep calling model slugs. This document does not change Protocol YAML.
- **v0 vertical:** XGBoost-on-embeddings via one abstract layer kind
  ``embedding_head``, compiling to today’s ``Finetune.xgboost``, emitting a
  package with multi-action ``encode`` + ``predict``.
- **v0 layer cardinality:** exactly one ``layers`` entry, and it must be
  ``type: embedding_head``. Ordered multi-layer stacks are a later vertical.
- **v0 tag default:** ``latest`` when ``--tag`` is omitted. Collision policy
  (overwrite vs error) remains deferred; plan may overwrite ``latest``.
- **v0 training knobs:** only fields needed to call ``Finetune.xgboost`` with
  sensible defaults. Unlisted API knobs (``n_estimators``, ``hyperopt``, …)
  use the Finetune client defaults unless the recipe later grows optional
  pass-through — do not expand scope to mirror the full API in v0.

## Recipe vs package

```text
project/
  models/
    antibody-binder-clf.yaml     # recipe (author)

~/.biolm/models/
  antibody-binder-clf/
    v1/
      BioLM                      # package manifest (compiled)
      # later optional: MLmodel, conda.yaml, weights pointers only
```

| | Recipe | Package (``BioLM``) |
|--|--------|---------------------|
| Who writes it | Human | ``build`` |
| Extension | ``.yaml`` | none (``BioLM``) |
| Location | Project tree | ``~/.biolm/models/<name>/<tag>/`` |
| Mutable | Yes | Rebuild / new tag to change |
| Data refs | Local path (v0) | Resolved path (+ digest later) |
| Layer fields | Intent (type, task, knobs) | Intent + required ``run_id``; digests/artifacts optional in v0 |
| ``actions`` | May be partial / omitted | Required, concrete I/O contract |
| Consumers | Author, ``build`` | Hub, push/deploy, MLflow flavor |

Optional later: ``build --write-package ./dist/...`` to also emit a package
directory next to the project without using the home-dir cache.

## On-disk package layout

```text
~/.biolm/models/<name>/<tag>/
└── BioLM
```

- ``<name>`` — model identity slug (from recipe ``name``).
- ``<tag>`` — build tag; **v0 default ``latest``** when ``--tag`` omitted.
  Collision policy deferred (plan may overwrite ``latest``).
- ``BioLM`` — compiled YAML manifest.

Heavy weights stay in object storage / training backend URIs referenced *from*
``BioLM``. The local registry stores metadata, not full weight blobs.

A future MLflow plugin can add ``MLmodel`` beside ``BioLM`` in the same
directory (``biolm`` flavor reading ``BioLM``, or thin ``python_function``
dispatch). That is deliberately natural given this layout; it is not required
for v0.

## Schema spine

### Recipe (author-facing example)

```yaml
schema_version: 1
name: antibody-binder-clf
description: Binder classifier on ESM2 embeddings

from: esm2-8m

layers:
  - type: embedding_head
    task: classification          # classification | regression
    data: ./data/binders.csv      # v0: local CSV path (or equivalent file Finetune accepts)
    # optional:
    # target_column: label        # default label
    # text_column: sequence       # default sequence
    # embedding_models: [esm2-8m] # default: [from]

# optional in recipe; build infers actions from the layer if omitted
```

### Package (``BioLM`` — compiled example)

```yaml
schema_version: 1
name: antibody-binder-clf
description: Binder classifier on ESM2 embeddings
tag: latest

from:
  slug: esm2-8m

layers:
  - type: embedding_head
    task: classification
    data:
      path: ./data/binders.csv    # as resolved at build time (absolute path OK)
    embedding_models: [esm2-8m]
    target_column: label
    text_column: sequence
    run_id: ft_abc123             # required on locked package
    # artifact / digest / metrics: optional in v0 when Finetune returns them

actions:
  encode:
    input: sequence               # delegates to base (from)
  predict:
    input: sequence
    task: classification

built:
  at: 2026-07-21T20:00:00Z
  status: locked                  # required
  recipe_path: /path/to/models/antibody-binder-clf.yaml
```

**Locked package minimum (v0 validation):** ``name``, ``from.slug``, exactly one
layer with ``type``, ``task``, ``run_id``, and ``data.path`` (or equivalent
resolved pointer); ``actions.encode`` and ``actions.predict`` each with at
least ``input``; ``built.status: locked`` and ``built.at``. Digests/artifacts
optional.

### Field notes

| Field | Role |
|-------|------|
| ``from`` | Base BioLM catalog slug (v0). Prior package identity as ``from`` is later. |
| ``layers`` | Ordered adaptation stack; abstract ``type``, not backend internals |
| ``actions`` | Serving contract (BioLM multi-action surface) |
| ``built`` | Package-only run-level metadata (not a second layer list) |

**Layer kinds (v0):** only ``embedding_head``.

**Layer kinds (later, not v0):** e.g. full finetune, LoRA/adapter, DSM stage,
RL — each maps to a runner backend. DSM stage stacks are the intended second
vertical once the package shape is stable.

### Dataset references

- **v0:** ``layers[].data`` is a **local filesystem path** to a CSV (or other
  file contents ``Finetune.xgboost`` already accepts when read into a CSV
  string / row list). No Hub dataset id resolution in v0.
- **Later:** Hub dataset id, SeqFrame, URI schemes, etc. Do not block v0 on a
  full dataset URI RFC.

## Build flow (v0 vertical)

```text
antibody-binder-clf.yaml
        │
        ▼
 biolm model build [path] [--tag TAG]
        │
        ├─ validate recipe
        ├─ resolve data (minimal in v0)
        ├─ Finetune.xgboost(...)
        ├─ Finetune.wait(run_id)
        └─ write ~/.biolm/models/<name>/<tag>/BioLM
```

Happy path:

1. Author recipe: ``from: esm2-8m`` + one ``embedding_head`` layer + data ref.
2. ``biolm model build`` compiles to ``Finetune.xgboost``, waits for success.
3. Emit package with lineage + ``actions.encode`` / ``actions.predict``.
4. (Stub / docs only in early impl) show how Hub or an MLflow flavor would read
   ``BioLM`` — shipping deploy is out of v0.

Why this vertical:

- ``Finetune.xgboost`` already exists.
- Forces the multi-action package story (frozen encoder + head).
- One abstract layer kind is enough to prove recipe → package.
- Exercises the gap between BioLM’s action surface and MLflow’s single
  ``predict`` without requiring a full flavor yet.

## CLI (sketch)

```text
biolm model build PATH [--tag TAG] [--name NAME]
biolm model package show NAME[:TAG]     # later
biolm model push NAME[:TAG]             # later — Hub
biolm model deploy NAME[:TAG]           # later — hub / Modal
```

v0 needs ``build`` at minimum. Show / push / deploy are reserved for later
plans; their existence justifies why the package is separate from the recipe.

Python sketch (non-binding):

```python
from biolm.models.definition import build_model

pkg = build_model("models/antibody-binder-clf.yaml", tag="v1")
pkg.path  # ~/.biolm/models/antibody-binder-clf/v1
pkg.manifest  # parsed BioLM
```

## Relationship to other constructs

| Construct | Role vs BioLM definition |
|-----------|---------------------------|
| Catalog ``Model`` / slug | Runtime client; package may *become* a slug after publish |
| ``Finetune`` | Imperative executor; ``build`` compiles recipe → these calls |
| Protocol YAML | Workflows over slugs; does not define models |
| Local dataset | Future ``layers[].data`` source; v0 uses a local CSV path only |
| SeqFrame / LLTP | Data / lab inputs upstream of training data; not the definition |
| MLflow plugin (today) | Protocol logging + dataset push/pull |
| MLflow flavor (future) | Read ``BioLM`` (+ optional ``MLmodel``) in the package dir |
| biolm-hub | Future consumer of packages for custom deploy |

## Non-goals (v0)

- New training algorithms beyond wrapping existing ``Finetune``
- Full layer taxonomy (LoRA, DSM multi-stage, RL)
- Hub custom-deploy product or ``push`` / ``deploy`` implementation
- Shipping an MLflow ``biolm`` flavor (layout must not block it)
- Recipe templating engine (Jinja / profiles); compile = resolve + lock
- Local weight blob storage / content-addressed layer FS / any daemon
- Changing Protocol YAML or protocol runners
- Exhaustive dataset URI resolution RFC

## Open questions (deferred, non-blocking)

- Collision policy when ``latest`` (or a given tag) already exists.
- Formal JSON Schema for recipe vs package (same ``schema_version``, different
  required sets).
- Naming of the Python module (``biolm.models.definition`` vs ``biolm.definition``).
- How published Hub slugs map back to local ``name``/``tag``.
- Richer ``actions`` JSON Schema (beyond ``input`` / ``task``) for Hub/MLflow.

## Success criteria for an implementation plan

1. Author a minimal ``embedding_head`` recipe YAML and validate it.
2. ``build`` launches and waits on ``Finetune.xgboost`` (or a test double).
3. Package directory appears at ``~/.biolm/models/<name>/<tag>/BioLM`` with
   resolved lineage fields and ``encode`` + ``predict`` actions.
4. Recipe file on disk is unchanged after a successful build.
5. Docs explain recipe vs ``BioLM`` package using the Docker/dbt analogies above.

## Related

See [2026-07-22-biolm-package-mlflow-bridge-design.md](2026-07-22-biolm-package-mlflow-bridge-design.md)
for weight load modes, actions translation, ``mlflow-biolm`` ownership, and export/deploy.
