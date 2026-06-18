========
BioLM AI
========


.. image:: https://img.shields.io/pypi/v/biolm.svg
        :target: https://pypi.python.org/pypi/biolm

.. image:: https://api.travis-ci.com/BioLM/py-biolm.svg?branch=production
        :target: https://travis-ci.org/github/BioLM/py-biolm

.. image:: https://readthedocs.org/projects/biolm-ai/badge/?version=latest
        :target: https://biolm-ai.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Python client and SDK for `BioLM <https://biolm.ai>`_

Install the package:

.. code-block:: bash

    pip install biolm

Local server (proxy for Modal-deployed models):

.. code-block:: bash

    pip install biolm[server]
    biolm server start

See ``docs/cli/server.rst`` and ``docs/server/oss-integration.md``.

Basic usage:

.. code-block:: python

    from biolm import biolm

    # Encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # Predict a batch of sequences
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

    # Write results to disk
    biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"], output='disk', file_path="results.jsonl")

Asynchronous usage:

.. code-block:: python

    from biolm.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

Overview
========

The BioLM Python client provides a high-level, user-friendly interface for interacting with the BioLM API. It supports both synchronous and asynchronous usage, automatic batching, flexible error handling, and efficient processing of biological data.

Main features:

- High-level BioLM constructor for quick requests
- Sync and async interfaces
- Automatic or custom rate limiting/throttling
- Schema-based batch size detection
- Flexible input formats (single key + list, or list of dicts)
- Low memory usage via generators
- Flexible error handling (raise, continue, or stop on error)
- Universal HTTP client for both sync and async

Features
========

- **High-level constructor**: Instantly run an API call with a single line.
- **Sync and async**: Use `BioLM` for sync, or `BioLMApiClient` for async.
- **Flexible rate limiting**: Use API throttle, disable, or set your own (e.g., '1000/second').
- **Schema-based batching**: Automatically queries API for max batch size.
- **Flexible input**: Accepts a single key and list, or list of dicts, or list of lists for advanced batching.
- **Low memory**: Uses generators for validation and batching.
- **Error handling**: Raise HTTPX errors, continue on error, or stop on first error.
- **Disk output**: Write results as JSONL to disk.
- **Universal HTTP client**: Efficient for both sync and async.
- **Direct access to schema and batching**: Use `BioLMApi` for advanced workflows, including `.schema()`, `.call()`, and `._batch_call_autoschema_or_manual()`.

**Example endpoints and actions:**

- `esm2-8m/encode`: Embedding for protein sequences.
- `esmfold/predict`: Structure prediction for protein sequences.
- `progen2-oas/generate`: Sequence generation from a context string.
- `dnabert2/predict`: Masked prediction for protein sequences.
- `ablang2/encode`: Embeddings for paired-chain antibodies.

* Free software: Apache Software License 2.0
* Documentation: https://docs.biolm.ai