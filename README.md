# biolm-sdk

[![PyPI](https://img.shields.io/pypi/v/biolm-sdk.svg)](https://pypi.org/project/biolm-sdk/)
[![CI](https://github.com/BioLM/biolm-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/BioLM/biolm-sdk/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-biolm.ai%2Fdocs-blue)](https://biolm.ai/docs)

Call biological language models from Python or the terminal.

Encode protein sequences, predict structures, generate variants, score antibodies, run DNA models — through one client against [BioLM](https://biolm.ai)'s hosted API or a local [biolm-hub](https://github.com/BioLM/biolm-hub) gateway.

```python
from biolm import Model

# Predict a structure
result = Model("esmfold").predict(type="sequence", items="MKTAYIAKQRQGHQAMAEIKQ")
print(result["mean_plddt"])

# Embed a library
embeddings = Model("esm2-8m").encode(
    type="sequence",
    items=["MKTAYIAKQRQ", "MKLAVIDSAQRQ", "MENDELMENDEL"],
)
```

Install: `pip install biolm-sdk` · Import: `import biolm` · CLI: `biolm`

---

## Setup

**Python 3.8+**

```bash
pip install biolm-sdk
```

**Credentials** — get a token at [biolm.ai](https://biolm.ai/ui/accounts/user-api-tokens/), then:

```bash
export BIOLM_TOKEN=<token>
# or
biolm login
```

Check everything is wired up:

```bash
biolm status
```

---

## Run models

### From Python

Bind a model, pass sequences (or PDBs, or other typed inputs), get results back.

```python
from biolm import Model

# Embeddings
esm = Model("esm2-8m")
vecs = esm.encode(type="sequence", items=["MKTAYIAKQRQ", "MDNELE"])

# Generation
progen = Model("progen2-oas")
seqs = progen.generate(
    type="context",
    items="M",
    params={"temperature": 0.7, "num_samples": 5, "max_length": 50},
)

# Structure from sequence
fold = Model("esmfold")
pdb = fold.predict(type="sequence", items="MKTAYIAKQRQ")
```

Load inputs from disk:

```python
from biolm.io import load_fasta

sequences = load_fasta("library.fasta")
Model("esm2-8m").encode(type="sequence", items=sequences)
```

Large jobs can stream to JSONL instead of memory:

```python
Model("esmfold").predict(
    type="sequence",
    items=sequences,
    output="disk",
    file_path="structures.jsonl",
)
```

### From the terminal

```bash
biolm model list
biolm model show esmfold
biolm model run esmfold predict -i sequences.fasta -o results.json
biolm model example esm2-8m encode   # prints a Python snippet you can paste
```

The CLI accepts FASTA, CSV, PDB, and JSON. It talks to the same API the SDK does.

---

## Build workflows

For jobs that are more than a single model call.

### Protocols

Multi-step jobs defined in YAML — validate locally, submit to the platform, poll until done.

```bash
biolm protocol validate design.yaml
biolm protocol run design.yaml -i inputs.json
```

```python
from biolm import run_protocol

results = run_protocol(
    "my-protocol-slug",
    inputs={"sequence": "MKTAYIAKQRQ"},
)
```

### Pipelines

For protein design at scale: generate variants, score them, filter, cluster — with DuckDB caching so re-runs skip work already done.

```bash
pip install "biolm-sdk[pipeline]"
```

**Saturation mutagenesis** — enumerate single mutants, score, keep the top N:

```python
from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

pipeline = GenerativePipeline(configs=[
    SaturationMutagenesisConfig(
        parent_sequence="MKTAYIAKQRQ",
        scoring_model="esm2-650m",
        score_field="logits",
        top_n=20,
    )
])
df = pipeline.run()
```

**Custom stages** — predict → filter → rank, composed explicitly:

```python
from biolm.pipeline import DataPipeline
from biolm.pipeline.filters import ThresholdFilter, RankingFilter

pipeline = DataPipeline(sequences=my_sequences)
# The model slug is intentionally spelled "temberture-regression" in the API.
pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
pipeline.add_filter(ThresholdFilter("tm", min_value=48.0))
pipeline.add_filter(RankingFilter("tm", top_n=10))
df = pipeline.run()
```

Or use the shorthand:

```python
from biolm.pipeline import Predict

df = Predict("temberture-regression", sequences=my_sequences, extractions="prediction", columns="tm")
```

See `scripts/` in this repo for antibody design, stability engineering, and multi-model examples.

---

## Run models locally with biolm-hub

Point the SDK at a [biolm-hub](https://github.com/BioLM/biolm-hub) gateway to run open-source models on your own hardware:

```bash
bh serve                              # in the biolm-hub repo
biolm hub set http://127.0.0.1:8000   # redirect SDK + CLI
biolm model list                      # discovers models from hub OpenAPI
```

`biolm hub unset` returns to the hosted API.

---

## How it works

You write synchronous Python. Under the hood the client is async: it reads each model's schema to pick batch sizes, sends batches in parallel (up to 16 concurrent by default), rate-limits to the API's throttle, retries transient network errors, and gzip-compresses large payloads.

| You need… | Use |
|-----------|-----|
| A notebook or script | `Model` |
| A one-liner | `biolm(entity=..., action=..., items=...)` |
| An async app or custom concurrency | `BioLMApiClient` from `biolm.core.http` |
| Full control over retries, schema, batching | `BioLMApi` |

Generators work as `items` — the client consumes them batch-by-batch without loading everything into memory.

---

## What's in the box

| | Python | CLI |
|---|--------|-----|
| Model inference | `Model`, `biolm()` | `biolm model` |
| YAML workflows | `run_protocol()`, `ProtocolClient` | `biolm protocol` |
| Design pipelines | `biolm.pipeline` *(optional extra)* | — |
| Local model gateway | `biolm.hub` | `biolm hub` |
| Platform accounts & environments | `PlatformClient`, `Workspace` | `biolm workspace`, `biolm org`, `biolm budget` |
| MLflow-backed datasets | `biolm.plugins.mlflow` *(optional extra)* | `biolm dataset` |
| Finetuning (XGBoost, DSM) | `Finetune` | — |
| File I/O | `biolm.io` (FASTA, CSV, PDB, JSON) | built into `biolm model run` |

Models include ESM2, ESMFold, ESM-1v, ProteinMPNN, ProGen2, AntiFold, IgBERT, DNABERT2, ABodyBuilder3, and more. Browse with `biolm model list` or at [biolm.ai](https://biolm.ai).

---

## Documentation

Full guides, API reference, and tutorials: **[biolm.ai/docs](https://biolm.ai/docs)**

| Task | Link |
|------|------|
| First run | [Quickstart](https://biolm.ai/docs/guide/quickstart.html) |
| Batching, errors, rate limits | [Core concepts](https://biolm.ai/docs/guide/concepts.html) |
| Pipeline design primitives | [Pipeline](https://biolm.ai/docs/sdk/pipeline.html) |
| Protocol YAML schema | [Protocol schema](https://biolm.ai/docs/yaml/protocol-schema.html) |
| CLI reference | [CLI](https://biolm.ai/docs/cli/index.html) |

---

## Development

```bash
git clone git@github.com:BioLM/biolm-sdk.git && cd biolm-sdk
pip install -r requirements_dev.txt
make install
RS=118 make test
```

See [CONTRIBUTING.rst](CONTRIBUTING.rst).

---

## License

Apache 2.0

---

<sub>Previously published as <code>biolmai</code>. Migration: <a href="https://biolm.ai/docs/notes/migration-1.0.html">biolm.ai/docs/notes/migration-1.0</a>.</sub>
