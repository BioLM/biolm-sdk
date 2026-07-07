.. _sdk-overview:

Overview
========

The BioLM Python SDK lets you call BioLM models from Python with minimal setup: encode sequences, predict structures, and generate sequences. Use a one-off function, a class-based Model, or the API client for more control.

**Quick example**

.. code-block:: python

    from biolm import biolm

    # Encode a sequence (e.g. ESM2-8M)
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # Predict structure (e.g. ESMFold)
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["MDNELE", "MENDEL"])

    # Generate sequences (e.g. ProGen2-OAS)
    result = biolm(
        entity="progen2-oas",
        action="generate",
        type="context",
        items="M",
        params={"temperature": 0.7, "num_samples": 2, "max_length": 17}
    )

**What you can do**

- **Encode** sequences to get embeddings (e.g. ESM2-8M).
- **Predict** protein structures from sequences (e.g. ESMFold).
- **Generate** new sequences from context (e.g. ProGen2-OAS).

**Ways to use the SDK**

- **One-off or script:** Use the function or the class-based Model. See :doc:`usage/usage`.
- **Sync vs async:** Scripts and notebooks → function or Model. High throughput or async apps → ``BioLMApi`` / ``BioLMApiClient``. See :doc:`usage/async-sync`.
- **Batching, errors, rate limits, disk output:** All supported; see :doc:`usage/batching`, :doc:`usage/rate_limiting`, :doc:`usage/error-handling`.

**Next steps**

- :doc:`models` — Available models and examples.
- :doc:`pipeline` — Pipeline design primitives: ``SaturationMutagenesisConfig``,
  ``IterativeMaskingDMSConfig``, ``DirectGenerationConfig``, and the full
  ``GenerativePipeline`` execution pattern.
- :doc:`usage/usage` — Usage patterns and when to use which interface.
- :doc:`faq` — Common questions.
- :doc:`api-reference/index` — Full API reference.
