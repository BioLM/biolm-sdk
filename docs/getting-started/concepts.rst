.. _getting-started-concepts:

=========
Concepts
=========

This page explains core concepts for using the BioLM Python client: the different client interfaces, sync vs async, batching, error handling, disk output, and rate limiting.

------------------------
Feature summary
------------------------

- **Simple one-off calls:** Use the high-level function or the class-based ``Model``; no setup required.
- **Sync and async:** Use the sync interface for scripts and notebooks, or the async client for high throughput and async apps.
- **Auto-batching:** Items are split by the API’s maximum batch size and sent in parallel. See :doc:`../sdk/usage/batching`.
- **Flexible input:** Single value, list of values, list of dicts, or a generator. For lists of plain strings you pass a type (e.g. sequence). See :doc:`../sdk/usage/batching`.
- **Error handling:** Raise exceptions, continue and collect errors, or stop on first error; optional retry of failed batches. See :doc:`../sdk/usage/error-handling`.
- **Disk output:** Write results as JSONL for very large jobs. See :ref:`disk-output` in :doc:`../sdk/usage/usage`.
- **Rate limiting:** Default throttle from the API schema; you can set a custom rate or concurrency. See :doc:`../sdk/usage/rate_limiting`.
- **Advanced control:** The sync and async API clients expose schema access, manual batching, and more. See :doc:`../sdk/usage/usage`.

------------------------
BioLM (simple sync)
------------------------

**``BioLM``** is the simplest synchronous interface: call a function with entity, action, and items. Single-item calls return a single result (dict); batch calls return a list. No event loop or asyncio required. Under the hood it uses the same asynchronous backend as ``BioLMApiClient``, so you get the performance of concurrent batch requests in a blocking call—no need to write async code.

**Example:**

.. code-block:: python

    from biolm import biolm

    # Single item: returns a dict
    result = biolm(entity="esmfold", action="predict", items="MDNELE")
    print(result["mean_plddt"])

    # Batch: returns a list of dicts
    result = biolm(entity="esmfold", action="predict", items=["MDNELE", "MENDEL"])

Internally, ``BioLM`` is a thin sync wrapper around the async client (via the ``synchronicity`` package), so batches are still sent concurrently on the backend; only the Python API is blocking.

------------------------
BioLMApi and BioLMApiClient
------------------------

For more control or high throughput you can use:

- **``BioLMApi``** — Synchronous interface to the same async backend; same style as ``BioLM`` but with more options (rate limits, retry, schema access). Blocking calls, concurrent batches under the hood.
- **``BioLMApiClient``** — The async client: its methods are coroutines and must be awaited. Use in async code (e.g. ``await model.encode(...)``) or with ``asyncio.run(...)``.

**When to use which:**

- **``BioLM``** — Simplest interface; best for scripts, Jupyter, and one-off or small batches.
- **``BioLMApiClient``** — Many requests in parallel, web servers, or any async app; you control concurrency and rate limiting.
- **``BioLMApi``** — Sync interface with more control than ``BioLM`` (e.g. custom rate limits, retry).

``BioLMApiClient`` always returns a list (even for one item) unless you set ``unwrap_single``. ``BioLM`` and ``BioLMApi`` return a single dict when you pass a single item.

**Sync example:**

.. code-block:: python

    from biolm import biolm
    result = biolm(entity="esmfold", action="predict", items="MDNELE")

**Async example:**

.. code-block:: python

    from biolm.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

**Using the async client from sync code:** wrap the call in :code:`asyncio.run(...)`, or use ``BioLMApi`` for a blocking interface.

------------------------
Batching and input formats
------------------------

The client supports single items, lists, and generators. It batches automatically using the model’s maximum batch size from the API schema. You do not need to split input yourself.

**Single item or list of values (e.g. sequences):** pass a type so the client knows how to interpret strings:

.. code-block:: python

    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILV")
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=["MSILV", "MDNELE"])

**List of dicts:** type is inferred from the keys, so you usually don’t pass type:

.. code-block:: python

    result = biolm(entity="esmfold", action="predict", items=[{"sequence": "SEQ1"}, {"sequence": "SEQ2"}])

**Generator (memory-efficient):** the client consumes it batch-by-batch:

.. code-block:: python

    def sequences(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield {"sequence": line.strip()}

    result = biolm(entity="esm2-8m", action="encode", items=sequences("sequences.txt"))

For full details and manual batching (list of lists), see :doc:`../sdk/usage/batching`.

------------------------
Error handling
------------------------

You can raise HTTP errors as exceptions or get them as dicts in the results. You can also stop on the first error batch or process all items and collect errors; with the API clients you can optionally retry failed batches as single items.

**Fail fast (exceptions):**

.. code-block:: python

    from biolm import biolm
    try:
        result = biolm(entity="esmfold", action="predict", type="sequence", items="BADSEQ", raise_httpx=True)
    except Exception as e:
        print("Caught:", e)

**Continue on error (errors as dicts in results):**

.. code-block:: python

    result = biolm(
        entity="esmfold", action="predict", type="sequence",
        items=["GOODSEQ", "BADSEQ"],
        raise_httpx=False, stop_on_error=False
    )
    for r in result:
        if "error" in r:
            print("Error:", r["error"])
        else:
            print("OK:", r.get("mean_plddt"))

See :doc:`../sdk/usage/error-handling` for the full behavior matrix and retry options.

------------------------
Disk output
------------------------

For very large jobs you can write results to a JSONL file instead of holding them in memory. Set output to disk and provide a file path. Results are one JSON object per line, in input order. Batch error behavior (stop on first error vs continue, retry) is the same as in-memory. See the :ref:`disk-output` section in :doc:`../sdk/usage/usage`.

.. code-block:: python

    biolm(
        entity="esmfold", action="predict", type="sequence",
        items=["SEQ1", "SEQ2"],
        output="disk", file_path="results.jsonl",
        stop_on_error=False
    )

------------------------
Rate limiting
------------------------

By default the client uses the API schema’s recommended throttle and a concurrency limit (semaphore) so you get good throughput without overloading. You can disable throttling, set a custom rate (e.g. requests per second or minute), or set a custom concurrency limit. See :doc:`../sdk/usage/rate_limiting` for details and examples.

.. code-block:: python

    from biolm.core.http import BioLMApi

    # Default (recommended): uses API throttle + semaphore
    model = BioLMApi("esmfold")

    # Custom: 1000 requests per second, max 5 concurrent
    model = BioLMApi("esmfold", rate_limit="1000/second", semaphore=5)
