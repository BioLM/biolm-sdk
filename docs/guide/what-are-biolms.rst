.. _what-are-biolms:

=========================
What are BioLMs?
=========================

BioLMs are biological language models — neural networks trained on proteins,
DNA, and antibodies to reason about sequence and structure. They power
embeddings (ESM2), structure prediction (ESMFold), inverse folding
(ProteinMPNN), and de novo sequence generation (ProGen2), among many others.
The BioLM API exposes each model behind a small, consistent interface so you can
call any of them the same way.

This page is the next step after the :doc:`quickstart`. The quickstart shows you
how to make your first call; this guide explains the *four inference actions*
that every model call maps to, the single pattern that ties them together, and
worked examples with real model slugs. Once you understand the pattern here, you
can run any model in the `BioLM catalog <https://biolm.ai/models>`_ without
learning a new API.

The four inference actions
==========================

Every BioLM model supports one or more of four actions. Each action is a method
on :class:`~biolm.models.Model`:

- **encode** — turn sequences into embeddings (numeric vectors). Use for
  similarity search, clustering, or as features for a downstream model. Example:
  ``esm2-8m``.
- **predict** — run the model forward to produce a structured result, such as a
  predicted 3D structure, a property score, or a classification. Example:
  ``esmfold``.
- **generate** — produce *new* sequences, optionally conditioned on a context or
  a scaffold structure. Example: ``progen2-oas``.
- **lookup** — retrieve precomputed or reference data associated with a model.
  Unlike the other three, ``lookup`` takes a ``query=`` argument instead of
  ``items=``.

A model only advertises the actions it supports. ESMFold predicts but does not
generate; ProGen2 generates but does not fold. To see which actions a given
model exposes — plus its input ``type`` and accepted ``params`` — check its page
in the `model catalog <https://biolm.ai/models>`_ or run ``biolm model show``
(below). This guide intentionally does not duplicate the per-model catalog.

The pattern
===========

Every inference follows the same shape: construct a ``Model`` with a slug, then
call the action method with ``items``, a ``type``, and optional ``params``.

.. code-block:: python

    from biolm import Model

    model = Model("<slug>")
    result = model.encode(   # or .predict(...), .generate(...)
        type="sequence",     # what kind of input you're passing
        items=[...],         # a single item or a list of items
        params={...},        # optional model-specific parameters
    )

The three arguments:

- ``items`` — a single sequence/structure or a list of them. Passing a list runs
  a batch; the SDK handles concurrency for you (see :doc:`batching`).
- ``type`` — the input kind the model expects, such as ``"sequence"``,
  ``"context"``, or ``"pdb"``. Required unless your items are already dicts.
- ``params`` — a dict of model-specific options (temperature, number of samples,
  normalization, etc.). Valid keys vary per model and per action.

The same three methods exist as thin ``biolm.encode``/``predict``/``generate``
helpers if you prefer a function call over an object, but ``Model`` is the
recommended interface when you make more than one call to the same model.

Worked examples
===============

Encode a sequence (ESM2-8M)
---------------------------

ESM2-8M turns an amino-acid sequence into an embedding vector you can use for
search or as features. It supports the ``encode`` action with ``type="sequence"``.

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")
    result = model.encode(type="sequence", items="MSILVTRPSPAGEEL")

Pass a list to ``items`` to encode many sequences in one batched call.

Predict a structure (ESMFold)
-----------------------------

ESMFold predicts a 3D protein structure from a sequence. It supports the
``predict`` action with ``type="sequence"`` and returns structure data
(coordinates plus confidence scores).

.. code-block:: python

    from biolm import Model

    model = Model("esmfold")
    result = model.predict(type="sequence", items=["MDNELE", "MENDEL"])

Generate sequences (ProGen2-OAS)
--------------------------------

ProGen2-OAS generates new antibody-like sequences from a starting context. It
supports the ``generate`` action with ``type="context"``, and uses ``params`` to
control sampling.

.. code-block:: python

    from biolm import Model

    model = Model("progen2-oas")
    result = model.generate(
        type="context",
        items="M",
        params={"temperature": 0.7, "num_samples": 2, "max_length": 17},
    )

A note on lookup
----------------

``lookup`` retrieves reference data rather than running inference, so it takes a
``query=`` dict (or list of dicts) instead of ``items=``:

.. code-block:: python

    model = Model("<slug>")
    result = model.lookup(query={"id": "..."})

Few models expose ``lookup`` today — check the catalog to see whether a model
supports it and what query fields it accepts.

From the command line
=====================

The CLI mirrors the Python actions, so you can explore and run models without
writing code. The ``biolm model`` command group covers the full loop:

.. code-block:: bash

    # Browse the catalog (filter, sort, export)
    biolm model list

    # Inspect one model: actions, input type, and JSON schemas
    biolm model show esm2-8m --include-schemas

    # Run an action against a file, stdin, or inline JSON
    echo '{"sequence": "MSILVTRPSPAGEEL"}' | biolm model run esm2-8m encode -i - --format json
    biolm model run esmfold predict -i sequences.fasta -o results.json

    # Print a copy-pasteable example for a model and action
    biolm model example progen2-oas --action generate

``biolm model run`` accepts the same four actions — ``encode``, ``predict``,
``generate``, ``lookup`` — and the same ``--type`` and ``--params`` options as
the Python interface. See :doc:`../cli/model` for the full command reference.

Where to go next
================

- :doc:`quickstart` — install, authenticate, and make your first call.
- :doc:`how-biolms-work` — slugs, schemas, and how items are normalized.
- :doc:`choosing-models` — discover models in the catalog, CLI, and Python.
- :doc:`running-inference` — run a model from Python or the CLI.
- :doc:`batching` — run many items efficiently with automatic concurrency.
- :doc:`error-handling` — inspect and recover from per-item and request errors.
- :doc:`../sdk/models` — the full ``Model`` API and more examples.
- :doc:`../cli/model` — complete ``biolm model`` command reference.
- `BioLM model catalog <https://biolm.ai/models>`_ — per-model actions, input
  types, params, and request/response schemas.
