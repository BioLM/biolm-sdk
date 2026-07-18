# Local Datasets Design

## Purpose

Prototype a first-class **dataset** construct in the BioLM SDK before a platform
dataset API exists. Local datasets are the default product concept; optional
backends (starting with MLflow) only extend verbs such as `push` / `pull`.

Datasets are a common venue for protocol results, finetuning inputs, sequence
designs, and similar artifacts — an inventory of addressable bags of files with
lightweight metadata, not a query engine (SeqFrame) or a pipeline cache
(DuckDBDataStore).

## Decisions

- Self-describing directories with `dataset.yaml` (no separate central registry file).
- Discovery under designated roots only (recursive within those roots); explicit
  paths may point anywhere.
- Core API: `DatasetClient` + `Dataset` — inventory plus addressable IDs.
- Create via `create` (greenfield) and `init` (adopt an existing directory).
- Remote verbs: `push` / `pull --backend …`; MLflow is a plugin adapter only.
- `pull` defaults to `~/.biolm/datasets/<id>/`; optional path override.
- Duplicate IDs across roots are a hard error (no silent shadowing).
- Typed openers (`SeqFrame`, etc.) are deferred; `type` is a soft label for now.

## On-disk layout

```text
~/.biolm/datasets/
└── finetuning-v1/
    ├── dataset.yaml
    └── data/
        ├── train.csv
        └── validation.csv
```

Existing trees can be adopted in place:

```text
project/training-data/
├── dataset.yaml
├── train.csv
└── validation.csv
```

### Discovery roots (priority order)

1. Explicit roots passed to `DatasetClient`
2. Project-local `./.biolm/datasets/`
3. User-global `~/.biolm/datasets/`
4. Extra roots from `~/.biolm/config.yaml` key `dataset_roots`

Automatic recursion is limited to these roots. Paths passed explicitly to
`get` / `init` / `show` may lie anywhere; datasets outside roots are reachable
by path but not by id-based `list` / `get` until their parent is a root.

### Identity

- `id` is a stable slug stored in `dataset.yaml` (not inferred from directory name).
- APIs accept a `Dataset`, dataset id, or filesystem path.
- Paths resolve directly; ids resolve across discovery roots.
- Duplicate ids → error listing every matching path.

## `dataset.yaml` schema

```yaml
schema_version: 1
id: finetuning-v1
description: Paired sequences for ESM2 LoRA fine-tune
created_at: 2026-07-17T18:00:00Z
type: files   # files | seqframe | protocol_results | ...
tags:
  - finetune
  - esm2
attrs:
  modality: protein
  source_protocol_run: run_abc123
```

| Field | Required | Notes |
|-------|----------|--------|
| `schema_version` | yes | Start at `1` |
| `id` | yes | Stable slug used as `dataset_id` |
| `description` | no | Human context |
| `created_at` | no | ISO-8601; stamped by create/init |
| `type` | no | Default `files` |
| `tags` | no | List of strings |
| `attrs` | no | Arbitrary map |

Not in v0 schema: file manifests, checksums, version history, publish state.

## Core API

```python
from biolm.datasets import DatasetClient, Dataset

client = DatasetClient()
ds = client.create("finetuning-v1", type="files", tags=["finetune"])
ds = client.init(Path("./training-data"), id="finetuning-v1")
datasets = client.list(type="files", tag="finetune")
ds = client.get("finetuning-v1")
ds = client.get(Path("./training-data"))

ds.id, ds.path, ds.type, ds.tags, ds.attrs
ds.files()
ds.add("train.csv")
ds.refresh()
ds.push(backend="mlflow")
client.pull("finetuning-v1", backend="mlflow")
```

### Creation

- **`create`**: `<primary_root>/<id>/dataset.yaml` + empty `data/`.
- **`init`**: write `dataset.yaml` into an existing directory; do not move files;
  warn if outside discovery roots; refuse overwrite unless `force=True`.

### `add`

Copy into `data/` when that directory exists, otherwise into the dataset root.
Supports files and recursive directories.

## CLI

```text
biolm dataset create ID [--type] [--tag] [--root] [--force]
biolm dataset init PATH --id ID [--type] [--tag] [--force]
biolm dataset list [--type] [--tag] [--format table|json]
biolm dataset show ID|PATH
biolm dataset add ID|PATH FILE [FILE ...]
biolm dataset push ID|PATH --backend mlflow [--mlflow-uri ...]
biolm dataset pull ID --backend mlflow [--path DIR] [--mlflow-uri ...]
```

Local is the default concept. There is no top-level MLflow dataset CLI.

## Plugin seam: push / pull

Core owns the verbs; backends implement:

```python
class DatasetPushPullBackend(Protocol):
    name: str
    def push(self, dataset: Dataset, **opts) -> dict: ...
    def pull(self, dataset_id: str, dest: Path, **opts) -> Dataset: ...
```

MLflow adapter wraps existing upload/download helpers, tags runs with
`type=dataset` / `dataset_id`, uploads `dataset.yaml` for round-trip, and
synthesizes metadata on pull when needed.

Missing backend → clear install hint (`pip install biolm-sdk[mlflow]`).

## Errors

| Situation | Behavior |
|-----------|----------|
| Invalid / missing `dataset.yaml` | Skip on discovery; hard error on get/init path |
| Duplicate `id` across roots | Error listing all matching paths |
| create/init when yaml exists | Error unless `force` |
| Unknown / missing backend | Error + install hint |
| pull into occupied dest with different id | Error unless `force` |
| add missing source | Error |

## Non-goals (v0)

- SeqFrame / typed loaders
- Platform backend
- Manifests, checksums, versions
- Delete / rename
- Wiring `ProtocolRun.save_to(dataset=…)`
- Multi-backend remotes / delta sync / auto-push

## Relationship to other constructs

| Construct | Role |
|-----------|------|
| Local dataset | Named, shareable bag of files + metadata |
| SeqFrame (future) | Biological tabular query / enrich over local data |
| DuckDBDataStore | Pipeline execution / cache persistence |
| MLflow plugin | Optional push/pull backend only |
