``biolm.models``
================

High-level interface for calling BioLM models: one-off requests, a bound
:class:`~biolm.models.Model`, and module-level helpers.

When to use
-----------

- ``biolm()`` — scripts and notebooks; one import, one call.
- ``Model`` — same model, many calls (``.encode()``, ``.predict()``, ``.generate()``).
- **Lower-level clients** — more control over batching, concurrency, and errors; see
  :doc:`../intro/client-interfaces` and :doc:`../intro/concurrency`.

Examples
--------

One-off call:

.. code-block:: python

    from biolm import biolm

    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

Bound model:

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")
    embeddings = model.encode(type="sequence", items=["SEQ1", "SEQ2"])

    model = Model("esmfold")
    structures = model.predict(type="sequence", items=["SEQ1", "SEQ2"])

API
---

.. autofunction:: biolm.biolm

.. autoclass:: biolm.models.Model
   :members: encode, predict, generate, lookup
   :undoc-members:

See also
--------

- :doc:`../intro/client-interfaces` — when to use ``biolm()`` vs ``Model`` vs ``BioLMApiClient``
- :doc:`../intro/quickstart` — minimal install and auth
- :doc:`../api-reference/biolm` — ``biolm.models`` module reference
