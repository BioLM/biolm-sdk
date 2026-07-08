``biolm.core``
==============

Low-level HTTP clients and legacy helpers. Product APIs such as :class:`~biolm.models.Model`
and :class:`~biolm.protocols.Protocol` wrap this layer — prefer those unless you need
direct control over batching, concurrency, schema access, or async execution.

When to use
-----------

- :class:`~biolm.core.http.BioLMApi` — sync client with schema helpers and manual batching
- :class:`~biolm.core.http.BioLMApiClient` — async client (``await`` its methods)
- ``biolm()`` — legacy one-shot sync wrapper (re-exported from ``biolm``); still common in examples
- :doc:`../intro/client-interfaces` — extended examples for batching, disk output, and errors
- :doc:`../intro/concurrency` — async patterns and throughput tuning

.. warning::

   ``biolm.core.legacy`` is deprecated and warns on import. Do not start new code on it.
   See :doc:`../api-reference/biolm.core.legacy` and :doc:`../notes/migration-1.0`.

Examples
--------

Legacy one-shot call:

.. code-block:: python

    from biolm import biolm

    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

Sync HTTP client:

.. code-block:: python

    from biolm.core.http import BioLMApi

    model = BioLMApi("esm2-8m", raise_httpx=False)
    result = model.encode(items=[{"sequence": "MSILV"}, {"sequence": "MDNELE"}])
    schema = model.schema("esm2-8m", "encode")

Async HTTP client:

.. code-block:: python

    from biolm.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        await model.shutdown()

    asyncio.run(main())

API
---

.. autofunction:: biolm.biolm

.. autoclass:: biolm.core.http.BioLMApi
   :members: encode, predict, generate, lookup, schema, call, shutdown
   :undoc-members:

.. autoclass:: biolm.core.http.BioLMApiClient
   :members: encode, predict, generate, lookup, schema, call, shutdown
   :undoc-members:

See also
--------

- :doc:`models` — :class:`~biolm.models.Model` (recommended inference interface)
- :doc:`../intro/batching` — generators, batch sizes, and manual batching
- :doc:`../intro/error-handling` — stop-on-error, retries, disk output
- :doc:`../intro/rate-limiting` — throttling and concurrency
- :doc:`../api-reference/biolm.core` — full ``biolm.core`` module tree
