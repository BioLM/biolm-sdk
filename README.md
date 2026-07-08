# biolm-sdk

[![PyPI](https://img.shields.io/pypi/v/biolm-sdk.svg)](https://pypi.org/project/biolm-sdk/)
[![CI](https://github.com/BioLM/biolm-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/BioLM/biolm-sdk/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-docs.biolm.ai-blue)](https://docs.biolm.ai)

Python SDK and CLI for the [BioLM](https://biolm.ai) platform and [biolm-hub](https://github.com/BioLM/biolm-hub) gateways.

- **PyPI:** [`biolm-sdk`](https://pypi.org/project/biolm-sdk/)
- **Repository:** [BioLM/biolm-sdk](https://github.com/BioLM/biolm-sdk)
- **Documentation:** [docs.biolm.ai](https://docs.biolm.ai)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

## Install

```shell
pip install biolm-sdk
pip install "biolm-sdk[pipeline]"  # optional pipeline framework
```

Requires Python 3.8+. Import in Python as `biolm`:

```python
from biolm import biolm
from biolm.core.http import BioLMApiClient
```

The legacy `biolmai` import and CLI alias remain available but are deprecated. See the [migration guide](https://docs.biolm.ai/notes/migration-1.0.html).

## Authentication

```shell
export BIOLM_TOKEN=<your_token>   # from https://biolm.ai/ui/accounts/user-api-tokens/
biolm login                        # OAuth; credentials saved to ~/.biolm/credentials
biolm status
```

## Quick start

**CLI**

```shell
biolm model list
biolm model run esm2-8m encode -i sequences.json
biolm protocol validate my-protocol.yaml
```

**Python**

```python
from biolm import biolm

result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")
```

**biolm-hub** (open-source models locally):

```shell
# In biolm-hub repo: bh serve
biolm hub set
biolm model list
```

## Concepts

| Area | SDK | CLI |
|------|-----|-----|
| Models | `biolm()`, `Model`, `BioLMApiClient` | `biolm model` |
| Protocols | `Protocol`, `ProtocolClient`, `run_protocol()` | `biolm protocol` |
| Pipelines | `biolm.pipeline` (optional extra) | — |
| Hub | `biolm.hub` | `biolm hub` |
| Workspaces / volumes | `Workspace`, `Volume` | `biolm workspace`, `biolm dataset` |
| Finetuning | `Finetune` | — |

Full guides: [docs.biolm.ai](https://docs.biolm.ai).

## Pipeline framework (optional)

Declarative multi-stage workflows with DuckDB-backed caching, resumability, and DAG execution:

```python
from biolm.pipeline import DataPipeline, ThresholdFilter

pipeline = DataPipeline(sequences=my_sequences)
pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
pipeline.add_filter(ThresholdFilter("tm", min_value=48.0))
pipeline.run()
df = pipeline.get_final_data()
```

See the [pipeline docs](https://docs.biolm.ai/sdk/pipeline.html).

## Migration from biolmai

| Before | After |
|--------|-------|
| `pip install biolmai` | `pip install biolm-sdk` |
| `import biolmai` | `import biolm` |
| `biolmai` CLI | `biolm` CLI |

Details: [migration guide](https://docs.biolm.ai/notes/migration-1.0.html).

## Development

```shell
git clone git@github.com:BioLM/biolm-sdk.git
cd biolm-sdk
pip install -r requirements_dev.txt
make install
RS=118 make test
```

See [CONTRIBUTING.rst](CONTRIBUTING.rst) for pull request and commit guidelines.

## License

Apache Software License 2.0
