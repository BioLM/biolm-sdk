.. _how-biolms-work:

=========================
How BioLMs work
=========================

The :doc:`what-are-biolms` guide showed the *four inference actions* and the
single ``items`` + ``type`` + ``params`` pattern that every model call follows.
This page explains *why* that pattern works for hundreds of different models: the
BioLM platform standardizes how every model is named, called, described, and
batched, so the same code runs any model in the `catalog
<https://biolm.ai/models>`_ without special-casing. Understanding this
standardization makes it easy to discover a model's inputs, validate a request
before you send it, and reason about what the SDK does on your behalf.

Model slugs are stable identifiers
==================================

Every model is addressed by a short **slug** — ``esm2-8m``, ``esmfold``,
``progen2-oas`` — and that slug is the one identifier you need everywhere. You
pass it to :class:`~biolm.models.Model`, to the ``biolm model`` CLI commands, and
it is what the catalog lists. Slugs are stable: the same slug maps to the same
model and weights over time, so a script written today keeps working, and a slug
you find in the catalog is exactly what you paste into your code.

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")   # the slug is the only per-model detail you supply

The four actions, per model
===========================

Every call maps to one of four actions — ``encode``, ``predict``, ``generate``,
and ``lookup`` — described in :doc:`what-are-biolms`. The key standardization is
that **not every model supports every action**, but the ones it does support all
behave identically across models. ESMFold exposes ``predict`` but not
``generate``; ProGen2 exposes ``generate`` but not ``predict``. A model
advertises its supported actions (and their schemas) so you never have to guess —
and when you call an unsupported action you get a clear error rather than
undefined behavior.

The ``items`` + ``type`` pattern
================================

Under the hood, the API expects each item as a JSON object (a dict). The SDK
normalizes your input into that shape with ``prepare_items_for_api``, and the
rule is simple:

- **Plain values need a** ``type``. If you pass strings (or a list of strings),
  the SDK wraps each one as ``{type: value}``. That is why ``type="sequence"`` is
  required for ``model.encode(items="MSILVTRP...")`` — it becomes
  ``{"sequence": "MSILVTRP..."}``.
- **Dicts are used as-is**. If your items are already dicts, the SDK infers the
  keys from them and does not add a ``type``. Passing ``type`` alongside dict
  items has no effect — the keys are already explicit, so ``type`` is ignored.
  (Passing ``type`` together with a *list of lists* of dicts is rejected
  outright.)

.. code-block:: python

    # These two calls send an identical request body:
    model.encode(type="sequence", items="MSILVTRPSPAGEEL")
    model.encode(items={"sequence": "MSILVTRPSPAGEEL"})

This is why single items, lists, generators, and lists-of-dicts all "just work":
they funnel through one normalization step into the same JSON the API expects.

``params`` for model-specific options
=====================================

``items`` carries *what* you want run; ``params`` carries *how*. Sampling
temperature, number of samples, output normalization, and similar knobs are
model- and action-specific, so they live in a separate ``params`` dict rather
than being mixed into each item. Valid keys vary — ``progen2-oas`` accepts
``temperature`` and ``num_samples`` for ``generate``, while an ``encode`` model
may accept none — which raises the obvious question: how do you know what a model
accepts?

JSON schemas describe every model + action
==========================================

The answer is that the platform publishes a **JSON schema for each model and
action pair**. The schema is the source of truth for the accepted input
``type``, the shape of each item, the valid ``params``, and the response format.

:class:`~biolm.models.Model` deliberately does not expose the raw schema on its
surface — the point of ``Model`` is to run inferences, not to introspect. When
you need the schema, reach for one of two tools:

.. code-block:: python

    from biolm.core.http import BioLMApiClient

    client = BioLMApiClient("esm2-8m")
    schema = await client.schema("esm2-8m", "encode")  # native async

Or, from the command line, print schemas alongside a model's actions:

.. code-block:: bash

    biolm model show esm2-8m --include-schemas

Both read the same published schema, so the CLI and SDK never disagree about what
a model accepts.

Batch size comes from the schema too
====================================

Because the schema declares each model's limits, the SDK can size batches for
you. When you pass a list to ``items``, ``BioLMApiClient`` calls
``_get_max_batch_size`` (which reads the schema's ``maxItems`` limit) and splits
your input into requests no larger than the model allows — you send one list, the
SDK sends the right number of correctly sized requests. You do not hard-code a
batch size, and if a model's limit changes, your code adapts automatically. See
:doc:`batching` for how concurrency and batching fit together.

Why standardization matters
===========================

Taken together — stable slugs, a fixed set of actions, one ``items`` + ``type``
pattern, a ``params`` dict, and a published schema per model and action — these
conventions mean the platform can add models without changing how you call them.
The schema drives validation, batching, and documentation from a single source,
so the catalog, the CLI, and the SDK all describe each model the same way.

For the per-model specifics — which actions each model supports, its input
``type``, accepted ``params``, and request/response schemas — use the
`BioLM model catalog <https://biolm.ai/models>`_ or ``biolm model show``. This
guide intentionally does not duplicate that catalog; it explains the rules the
catalog's entries follow.

Where to go next
================

- :doc:`what-are-biolms` — the four actions and the call pattern, with examples.
- :doc:`choosing-models` — find the right slug for your task.
- :doc:`batching` — run many items efficiently with automatic concurrency.
- :doc:`../sdk/models` — the full ``Model`` API and more examples.
- :doc:`../cli/model` — complete ``biolm model`` command reference.
- `BioLM model catalog <https://biolm.ai/models>`_ — per-model actions, input
  types, params, and request/response schemas.
