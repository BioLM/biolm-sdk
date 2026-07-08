==================
Client interfaces
==================

**Recommended:** :class:`~biolm.models.Model` — bind to a model slug, then call
``encode``, ``predict``, ``generate``, or ``lookup``. See :doc:`../sdk/models`.

For legacy one-shot calls (``biolm()``) or direct HTTP clients (``BioLMApi``,
``BioLMApiClient``), see :doc:`../sdk/core` and the sections below.

**Sync vs async:** ``Model``, ``biolm()``, and ``BioLMApi`` are **synchronous**
(blocking). For async code (e.g. FastAPI, Jupyter with top-level ``await``), use
``BioLMApiClient`` and ``await`` its methods. See :doc:`concurrency` for details.

Model (recommended)
-------------------

Bind to a model once, then call encode, predict, or generate as needed.

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")
    result = model.encode(type="sequence", items=["MSILV", "MDNELE"])

    model = Model("esmfold")
    result = model.predict(type="sequence", items=["MDNELE", "MENDEL"])

    model = Model("progen2-oas")
    result = model.generate(
        type="context",
        items="M",
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17},
    )

One-off calls (``biolm()`` — legacy)
------------------------------------

Still common in quickstarts and older examples. Prefer :class:`~biolm.models.Model`
for new code.

.. code-block:: python

    from biolm import biolm

    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["MDNELE", "MENDEL"])
    result = biolm(
        entity="progen2-oas",
        action="generate",
        type="context",
        items="M",
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17},
    )

    biolm(
        entity="esmfold",
        action="predict",
        type="sequence",
        items=["MSILV", "MDNELE"],
        output="disk",
        file_path="results.jsonl",
    )

HTTP clients (advanced)
-----------------------

For schema access, custom error handling, and manual batching:

.. code-block:: python

    from biolm.core.http import BioLMApi

    model = BioLMApi("esm2-8m", raise_httpx=False)
    result = model.encode(items=[{"sequence": "MSILV"}, {"sequence": "MDNELE"}])

    model = BioLMApi("progen2-oas")
    result = model.generate(
        items=[{"context": "M"}],
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17},
    )

    schema = model.schema("esm2-8m", "encode")
    max_batch = model.extract_max_items(schema)

    batches = [[{"sequence": "MSILV"}, {"sequence": "MDNELE"}], [{"sequence": "MENDEL"}]]
    result = model._batch_call_autoschema_or_manual("encode", batches)

.. note::

   Prefer :meth:`~biolm.core.http.BioLMApi.encode` for normal use;
   ``_batch_call_autoschema_or_manual`` is for explicit batch control.

.. tip::

   **Large datasets?** Pass a generator instead of a list so items are consumed
   batch-by-batch. See :doc:`batching`. For concurrency and rate limits, see
   :doc:`rate-limiting`.

Async usage (``BioLMApiClient``)
--------------------------------

Only **BioLMApiClient** exposes async methods; you must await them. ``Model``,
``biolm()``, and ``BioLMApi`` are synchronous and must not be awaited.

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

For large jobs you can write results to a JSONL file instead of returning them in
memory. Set ``output`` to disk and pass a ``file_path``. See :doc:`error-handling`
for stop-on-error and retry options.

**Examples:**

.. code-block:: python

    from biolm import Model, biolm

    model = Model("esmfold")
    model.predict(
        type="sequence",
        items=["MSILV", "BADSEQ"],
        output="disk",
        file_path="results.jsonl",
        stop_on_error=False,
    )

    biolm(
        entity="esmfold",
        action="predict",
        type="sequence",
        items=["MSILV", "BADSEQ"],
        output="disk",
        file_path="results.jsonl",
        stop_on_error=True,
    )

**When to use which:** New code → :class:`~biolm.models.Model`. Legacy one-shots →
``biolm()``. Direct HTTP control or async → ``BioLMApi`` / ``BioLMApiClient``. See
:doc:`../sdk/models`, :doc:`../sdk/core`, and :doc:`concepts`.
