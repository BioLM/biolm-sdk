# biolm-sdk

Python SDK and CLI for the [BioLM](https://biolm.ai) platform and [biolm-hub](https://github.com/BioLM/biolm-hub) gateways.

- **PyPI:** [`biolm-sdk`](https://pypi.org/project/biolm-sdk/)
- **Repository:** [BioLM/biolm-sdk](https://github.com/BioLM/biolm-sdk)
- **Docs:** [docs.biolm.ai](https://docs.biolm.ai) · [GitHub Pages JSON export](https://biolm.github.io/biolm-sdk/)

## Install

```shell
pip install biolm-sdk
pip install "biolm-sdk[pipeline]"  # optional pipeline framework
```

Import in Python as `biolm`:

```python
from biolm import biolm
from biolm.core.http import BioLMApiClient
```

The legacy `biolmai` PyPI package and CLI alias remain available but are deprecated. See [migration guide](docs/getting-started/migration-1.0.rst).

## Quick start

```shell
biolm login
biolm model list
biolm model run esm2-8m encode -i sequences.json
```

Open-source models via biolm-hub:

```shell
# In biolm-hub repo: bh serve
biolm hub set
biolm model list
```

## Pipeline framework (optional)

`biolm-sdk` includes an optional declarative pipeline system for protein-engineering workflows: chain `add_prediction` / `add_filter` / `add_clustering` calls, get DuckDB-backed caching, resumability, branched DAG execution, and a `WorkingSet` transport that scales to large sequence sets.

```python
from biolm.pipeline import DataPipeline, ThresholdFilter

pipeline = DataPipeline(sequences=my_sequences)
pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
pipeline.add_filter(ThresholdFilter("tm", min_value=48.0))
pipeline.run()
df = pipeline.get_final_data()
```

Full docs: [`biolm/pipeline/README.md`](biolm/pipeline/README.md) · architecture: [`biolm/pipeline/PIPELINE_VISION_AND_ARCHITECTURE.md`](biolm/pipeline/PIPELINE_VISION_AND_ARCHITECTURE.md).

## Development

Clone the repo and install in editable mode:

```shell
git clone git@github.com:BioLM/biolm-sdk.git
cd biolm-sdk
pyenv install 3.11.5
pyenv virtualenv 3.11.5 biolm-sdk
pyenv activate biolm-sdk
pip install -r requirements_dev.txt
make install
```

Set your API token (canonical name `BIOLM_TOKEN`; `BIOLMAI_TOKEN` still works):

```shell
export BIOLM_TOKEN=<your_token>
biolm status
```

Run tests:

```shell
RS=118 make test
```

See [CONTRIBUTING.rst](CONTRIBUTING.rst) for pull request guidelines.
