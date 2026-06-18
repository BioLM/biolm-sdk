=====
Usage
=====

You can call the API in three ways: the one-off function, the class-based Model (one model, multiple calls), or the API client for advanced control. Examples below.

**Sync vs async:** The function ``biolm()``, ``Model``, and ``BioLMApi`` are **synchronous** (blocking) interfaces: you call them without ``await`` and get the result when the call returns. Under the hood they use the same asynchronous backend, so batches are still sent concurrently and you get high throughput without writing async code. For async code (e.g. FastAPI, Jupyter with top-level ``await``), use ``BioLMApiClient`` and ``await`` its methods. See :doc:`async-sync` for details and which methods can be awaited.

One-off calls (function)
------------------------

.. code-block:: python

    from biolm import biolm

    # ESM2-8M: encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # ESM2-8M: encode a batch of sequences
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=["MSILV", "MDNELE"])

    # ESMFold: predict structure for a batch
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["MDNELE", "MENDEL"])

    # ProGen2-OAS: generate new sequences from a context
    result = biolm(
        entity="progen2-oas",
        action="generate",
        type="context",
        items="M",
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}
    )
    # result is a list of dicts with "sequence" keys

    # Write results to disk
    biolm(entity="esmfold", action="predict", type="sequence", items=["MSILV", "MDNELE"], output='disk', file_path="results.jsonl")

Class-based (one model, multiple calls)
---------------------------------------

Bind to a model once, then call encode, predict, or generate as needed.

.. code-block:: python

    from biolm import Model

    # One model, multiple operations
    model = Model("esm2-8m")
    result = model.encode(type="sequence", items=["MSILV", "MDNELE"])

    model = Model("esmfold")
    result = model.predict(type="sequence", items=["MDNELE", "MENDEL"])

    model = Model("progen2-oas")
    result = model.generate(type="context", items="M", params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17})

API client (sync, more control)
-------------------------------

For schema access, custom error handling, and manual batching:

.. code-block:: python

    from biolm.core.http import BioLMApi

    # Use BioLMApi for more control, e.g. batching, error handling, schema access
    model = BioLMApi("esm2-8m", raise_httpx=False)

    # Encode a batch
    result = model.encode(items=[{"sequence": "MSILV"}, {"sequence": "MDNELE"}])

    # Generate with ProGen2-OAS
    model = BioLMApi("progen2-oas")
    result = model.generate(
        items=[{"context": "M"}],
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}
    )

    # Access the schema for a model/action
    schema = model.schema("esm2-8m", "encode")
    max_batch = model.extract_max_items(schema)

    # Call the API directly (rarely needed)
    resp = model.call("encode", [{"sequence": "MSILV"}])

    # Advanced: manual batching (prefer .encode(items=...) for normal use)
    batches = [[{"sequence": "MSILV"}, {"sequence": "MDNELE"}], [{"sequence": "MENDEL"}]]
    result = model._batch_call_autoschema_or_manual("encode", batches)

.. note::

   **Normal use:** Prefer :meth:`~biolmai.core.http.BioLMApi.encode` (e.g. ``model.encode(items=...)``) for encoding; ``_batch_call_autoschema_or_manual`` is for advanced cases when you need to control batches explicitly.

.. tip::

   **Large datasets?** Pass a generator instead of a list so items are consumed batch-by-batch—you never load everything into memory. See :doc:`batching`. For concurrency and rate limits, see :doc:`rate_limiting`.

Async usage (BioLMApiClient — await these methods)
--------------------------------------------------

Only **BioLMApiClient** exposes async methods; you must await them (e.g. ``await model.encode(...)``, ``await model.predict(...)``, ``await model.schema(...)``). The function ``biolm()``, ``Model``, and ``BioLMApi`` are synchronous and must not be awaited.

.. code-block:: python

    from biolm.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

.. _disk-output:

Disk output
-----------

For large jobs you can write results to a JSONL file instead of returning them in memory. Set ``output`` to disk and pass a ``file_path``. One line per input item, in input order. If a batch fails, an error dict is written for each item in that batch. You can stop on first error or process all items; with the API client you can also retry failed batches as single items. See :doc:`error-handling` for the options.

**Examples:**

.. code-block:: python

    # Write to disk, continue on errors
    biolm(entity="esmfold", action="predict", type="sequence", items=["MSILV", "BADSEQ"],
          output='disk', file_path="results.jsonl", stop_on_error=False)

    # Write to disk, stop on first error
    biolm(entity="esmfold", action="predict", type="sequence", items=["MSILV", "BADSEQ"],
          output='disk', file_path="results.jsonl", stop_on_error=True)

**When to use which:** One-off or quick scripts → use the function. One model and several operations → use Model. More control (batching, errors, schema, reuse) → use ``BioLMApi`` or ``BioLMApiClient``. See :doc:`../models` and :doc:`../../getting-started/concepts`.
