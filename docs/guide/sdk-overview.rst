.. _intro-sdk-overview:

Python SDK overview
===================

The BioLM Python SDK lets you call BioLM models from Python with minimal setup: encode sequences, predict structures, and generate sequences. The recommended interface is :class:`~biolm.models.Model`; lower-level HTTP clients live in ``biolm.core``.

**Quick example**

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")
    result = model.encode(type="sequence", items="MSILVTRPSPAGEEL")

    model = Model("esmfold")
    result = model.predict(type="sequence", items=["MDNELE", "MENDEL"])

    model = Model("progen2-oas")
    result = model.generate(
        type="context",
        items="M",
        params={"temperature": 0.7, "num_samples": 2, "max_length": 17},
    )

**What you can do**

- **Encode** sequences to get embeddings (e.g. ESM2-8M).
- **Predict** protein structures from sequences (e.g. ESMFold).
- **Generate** new sequences from context (e.g. ProGen2-OAS).

**Ways to use the SDK**

- **Model inference:** Use :class:`~biolm.models.Model`. See :doc:`client-interfaces` and :doc:`../sdk/models`.
- **Protocols, pipelines, platform:** See :sdklink:`SDK reference <../../sdk/index.html>`.
- **Sync vs async:** ``Model`` is sync; async apps use ``BioLMApiClient``. See :doc:`concurrency`.
- **Batching, errors, rate limits, disk output:** See :doc:`batching`, :doc:`rate-limiting`, :doc:`error-handling`.
- **Advanced / legacy HTTP clients:** See :doc:`../sdk/core`.

**Next steps**

- :doc:`../sdk/models` — ``Model`` interface and examples.
- :doc:`../sdk/core` — ``biolm()``, ``BioLMApi``, and ``BioLMApiClient``.
- :doc:`../sdk/pipeline` — Pipeline config types and ``GenerativePipeline``.
- :doc:`client-interfaces` — When to use which client interface.
- :doc:`faq` — Common questions.
- :sdklink:`Full SDK module reference <../../sdk/index.html>`
