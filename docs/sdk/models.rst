``biolm.models``
================

Model
-----

Recommended interface for model inference: bind to a model slug once, then call
``encode``, ``predict``, ``generate``, or ``lookup``. Wraps the HTTP client in
``biolm.core`` with a simpler, product-focused API.

Catalog and example helpers live in ``biolm.models.examples`` (also re-exported
from ``biolm`` as ``get_example`` and ``list_models``).

When to use
-----------

- **Model** — preferred for scripts, notebooks, and apps that call the same model repeatedly.
- **Lower-level clients** — ``biolm()``, ``BioLMApi``, and ``BioLMApiClient`` for advanced
  batching, async, and schema control; see :doc:`core`.

Examples
--------

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")
    embeddings = model.encode(type="sequence", items=["SEQ1", "SEQ2"])

    model = Model("esmfold")
    structures = model.predict(type="sequence", items=["SEQ1", "SEQ2"])

    model = Model("progen2-oas")
    sequences = model.generate(
        type="context",
        items="M",
        params={"temperature": 0.7, "num_samples": 2, "max_length": 17},
    )

Example generation
----------------

Generate copy-pasteable Python for a model action, or browse the model catalog:

.. code-block:: python

    from biolm import Model
    from biolm.models.examples import get_example, list_models

    model = Model("esm2-8m")
    print(model.get_example("encode"))
    print(get_example("esm2-8m", "encode"))

    for entry in list_models():
        print(entry.get("model_slug") or entry.get("name"))

From the terminal: ``biolm model example esm2-8m encode``. See :doc:`../cli/model`.

API
---

.. autoclass:: biolm.models.Model
   :members: encode, predict, generate, lookup, get_example, get_examples
   :undoc-members:

.. autofunction:: biolm.models.examples.get_example

.. autofunction:: biolm.models.examples.list_models

See also
--------

- :doc:`core` — ``biolm()``, ``BioLMApi``, and ``BioLMApiClient``
- :doc:`../intro/client-interfaces` — sync vs async and disk output
- :doc:`../intro/quickstart` — minimal install and auth
- :doc:`../cli/model` — ``biolm model example`` and catalog commands
- :doc:`../api-reference/biolm.models` — full ``biolm.models`` package reference
