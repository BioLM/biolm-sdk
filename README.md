# Python SDK for BioLM API

## Pipeline framework (optional)

`biolmai` ships an optional declarative pipeline system for protein-engineering
workflows: chain `add_prediction` / `add_filter` / `add_clustering` calls,
get DuckDB-backed caching, resumability, branched DAG execution, and a
`WorkingSet` transport that scales to ~1M sequences as 28 MB of integers.

```shell
pip install 'biolmai[pipeline]'
```

```python
from biolm.pipeline import DataPipeline, ThresholdFilter

pipeline = DataPipeline(sequences=my_sequences)
pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
pipeline.add_filter(ThresholdFilter("tm", min_value=48.0))
pipeline.run()
df = pipeline.get_final_data()
```

Full docs: [`biolmai/pipeline/README.md`](biolmai/pipeline/README.md) · architecture: [`biolmai/pipeline/PIPELINE_VISION_AND_ARCHITECTURE.md`](biolmai/pipeline/PIPELINE_VISION_AND_ARCHITECTURE.md).

## Development

First, set up the Python environment using `pyenv`:
```shell
pyenv install 3.11.5
pyenv virtualenv 3.11.5 py-biolm
pyenv activate py-biolm
```

Then, install the development dependencies:
```shell
pip install -r requirements_dev.txt
```

Finally, run the Makefile to complete the setup:
```shell
make install
```

This installs the biolmai package itself in editable mode. This allows you to make changes to the package and see those changes reflected without needing to reinstall the package:
It runs the command: `pip install -e .`. The -e flag stands for "editable mode". This is a useful setup for development as it means any changes you make to the package code will be immediately available in your environment, without needing to reinstall the package each time you make a change.


## Set the `BIOLMAI_TOKEN` environment variable
Set the token in your shell to authenticate your requests to the BioLM API:
```shell
export BIOLMAI_TOKEN=<your_token>
```
The token can be obtained from your BioLM account: [Get your BioLM API token](https://biolm.ai/ui/accounts/user-api-tokens/).
Run `biolmai status` to verify that the token is set correctly.

## Run tests with a specific seed
Use this command to run the automated tests:
```shell
RS=118 make test
```
The `RS` environment variable sets the seed for any random operations in the tests, ensuring consistent results.
