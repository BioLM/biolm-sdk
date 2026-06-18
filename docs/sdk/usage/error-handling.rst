========================
Error Handling
========================

You can either raise HTTP errors as exceptions (fail fast) or get them as dicts in the results and choose whether to stop on the first error batch or process all items. With the API client you can also retry failed batches as single items.

**Fail fast (exceptions):**

.. code-block:: python

    from biolm import biolm
    try:
        result = biolm(entity="esmfold", action="predict", type="sequence", items="BADSEQ", raise_httpx=True)
    except Exception as e:
        print("Caught:", e)

**Continue and collect errors:**

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
            print("OK")

------------------------
Options in brief
------------------------

- **raise_httpx** — If True, HTTP errors raise an exception immediately. If False, errors are returned as dicts (with "error" and "status_code" keys). Default: False for the high-level function/BioLM, True for BioLMApi/BioLMApiClient.
- **stop_on_error** — When raise_httpx is False: True = stop after the first error batch; False = process all items and include error dicts in results.
- **retry_error_batches** — (BioLMApi/BioLMApiClient only.) When True, failed batches are retried as single items so you can get partial results.

------------------------
Behavior Matrix
------------------------

.. list-table:: Error Handling Behavior Matrix
   :widths: 20 20 20 40
   :header-rows: 1

   * - ``raise_httpx``
     - ``stop_on_error``
     - ``retry_error_batches``
     - Behavior
   * - True
     - (any)
     - (any)
     - Exception raised on first HTTP error (no results returned)
   * - False
     - True
     - False
     - Stop after first error batch; errors returned as dicts
   * - False
     - False
     - False
     - Continue on errors; errors returned as dicts in results
   * - False
     - True/False
     - True
     - Failed batches retried as single items; errors as dicts

------------------------
More examples
------------------------

**Stop on first error:**

.. code-block:: python

    result = biolm(
        entity="esmfold", action="predict", type="sequence",
        items=["GOODSEQ", "BADSEQ", "ANOTHER"],
        raise_httpx=False, stop_on_error=True
    )
    # Only results up to and including the first error are returned

**Retry failed batches as single items (BioLMApi/BioLMApiClient only):**

Sync (BioLMApi):

.. code-block:: python

    from biolm.core.http import BioLMApi

    model = BioLMApi("esm2-8m", raise_httpx=False, retry_error_batches=True)
    result = model.encode(items=[{"sequence": "GOOD"}, {"sequence": "BAD"}])
    # If a batch fails, each item is retried individually

Async (BioLMApiClient):

.. code-block:: python

    from biolm.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esm2-8m", raise_httpx=False, retry_error_batches=True)
        result = await model.encode(items=[{"sequence": "GOOD"}, {"sequence": "BAD"}])
        return result

    asyncio.run(main())

------------------------
Error result shape
------------------------

When raise_httpx is False, failed items appear as dicts in the result list with an "error" string and a "status_code" (e.g. 422). The examples above show how to detect and handle them.

------------------------
Parameter Availability Summary
------------------------

.. list-table:: Parameter Availability by Client
   :widths: 25 20 20 20
   :header-rows: 1

   * - Client
     - raise_httpx
     - stop_on_error
     - retry_error_batches
   * - ``biolm()``
     - ✅ (kwarg)
     - ✅ (kwarg)
     - ❌
   * - ``BioLM``
     - ✅ (kwarg)
     - ✅ (kwarg)
     - ❌
   * - ``BioLMApi``
     - ✅ (constructor)
     - ✅ (method param)
     - ✅ (constructor)
   * - ``BioLMApiClient``
     - ✅ (constructor)
     - ✅ (method param)
     - ✅ (constructor)

------------------------
See Also
------------------------

- :doc:`batching`
- :doc:`usage` (includes Disk output)
- :doc:`../faq`
