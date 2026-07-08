===
FAQ
===

**Q:** Can I use generators or iterators for ``items``?

A: Yes. Pass a generator (or any iterable) instead of a list. The client consumes it batch-by-batch, so you never hold all items in memory. Ideal for large files or streams. The generator is fully consumed during the call. See :doc:`batching`.

**Q: When do I need to specify** ``type`` (e.g. ``type="sequence"``)?

A: When ``items`` is a string or a list of non-dict values (e.g. a list of sequence strings). If ``items`` is a list or generator of dicts like ``{"sequence": "..."}``, the client infers the type and you don't need it.

**Q: What characters are valid in protein sequences?**

A: Use standard amino acid letters: ``ACDEFGHIKLMNPQRSTVWYBXZUO``. Example: ``random.choices('ACDEFGHIKLMNPQRSTVWY', k=6)`` for random valid sequences.

**Q: How do I process a large batch of sequences?**

A: Provide a list of dicts or a list of values; batching is automatic. For **very large** datasets, use a generator so items are streamed batch-by-batch. For huge result sets, use ``output='disk'`` to write JSONL to a file.

**Q: How do I handle errors gracefully?**

A: Set ``raise_httpx=False`` and choose ``stop_on_error=True`` or ``False``. With ``BioLMApi``, you can also set ``retry_error_batches=True`` to retry failed batches as single items. Example:

.. code-block:: python

    result = biolm(..., raise_httpx=False)
    for r in result:
        if isinstance(r, dict) and "error" in r:
            print("Error:", r["error"])

**Q: How do I write results to disk?**

A: Set ``output='disk'`` and provide ``file_path`` in either ``BioLM`` or ``BioLMApi``. Example:

.. code-block:: python

    biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"],
          output='disk', file_path="results.jsonl")

**Q: How do I use the async client?**

A: Use ``BioLMApiClient``; its methods are coroutines and must be awaited (e.g. ``await model.encode(...)``, ``await model.predict(...)``). Do not await ``biolm()``, ``Model``, or ``BioLMApi``—those are synchronous. See :doc:`async-sync` for which methods can be awaited.

**Q: How does the client achieve high throughput?**

A: By default, the client batches your items (schema-based size), sends batch requests in parallel (up to 16 concurrent), and applies API-recommended rate limiting. No configuration needed. See :doc:`rate-limiting`.

**Q: How do I set a custom rate limit?**

A: Use ``rate_limit="1000/second"`` or provide your own semaphore to ``BioLMApi`` or ``BioLMApiClient``.

**Q: When should I use BioLMApi instead of BioLM?**

A: Use ``BioLMApi`` if you need:
    - To reuse a client for multiple calls (avoids re-auth)
    - To access the schema or batch size programmatically
    - To call lower-level methods like ``.call()`` or ``.schema()``
    - To do advanced batching or error handling

**Q: What are** :code:`.schema()` **,** :code:`.call()` **, and** :code:`._batch_call_autoschema_or_manual()` **for?**

A: These are lower-level methods on ``BioLMApi``/``BioLMApiClient``:

- ``.schema(model, action)``: Fetches the API schema for a model/action, useful for inspecting input/output formats and max batch size.
- ``.call(func, items, ...)``: Makes a direct API call for a given function (e.g., "encode"), bypassing batching logic. Useful for custom workflows or debugging.
- ``._batch_call_autoschema_or_manual(func, items, ...)``: Internal batching logic that splits items into batches based on schema, handles errors, and can write to disk. Advanced users may use this for custom batching or error handling.
