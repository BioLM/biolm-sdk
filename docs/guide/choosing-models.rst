.. _choosing-models:

=========================
Choosing a model
=========================

The BioLM catalog has dozens of models — encoders, structure predictors,
generators — and more are added over time. This guide is about the step *before*
you write a call: figuring out **which** model fits your task and confirming the
exact actions, input ``type``, and ``params`` it accepts. Once you know that, the
call itself follows the single pattern from :doc:`what-are-biolms`.

There are three ways to discover models — the web catalog, the CLI, and the
Python API. They all draw from the same source, so pick whichever fits your
workflow: browse visually, script against the terminal, or stay in a notebook.

Start from the task, not the model
===================================

Most searches start with a task. Map the task to one of the four inference
actions, then look for a model that advertises that action:

- **Embeddings / similarity / features** → the **encode** action. Reach for an
  encoder such as ``esm2-8m`` (proteins) and use the vectors for search,
  clustering, or as inputs to a downstream model.
- **Structure or property prediction** → the **predict** action. ``esmfold``
  predicts 3D structure from a sequence; other predictors return property scores
  or classifications.
- **New sequences (de novo or conditioned)** → the **generate** action.
  ``progen2-oas`` generates antibody-like sequences; inverse-folding models like
  ProteinMPNN generate sequences for a target backbone.
- **Reference / precomputed data** → the **lookup** action, which takes a
  ``query=`` instead of ``items=``.

A model only exposes the actions it supports, so "which model?" and "which
action?" are really one question. The tools below all report the supported
actions next to each model.

The web catalog
===============

The `model catalog <https://biolm.ai/models>`_ is the fastest way to browse. Each
model has a page listing its supported actions, expected input ``type``,
accepted ``params``, and the request/response schemas — plus copy-pasteable
snippets. Use it when you are exploring or want the authoritative per-model
reference; this documentation intentionally does not duplicate it.

From the command line
=====================

The ``biolm model`` command group turns discovery into a scriptable loop:

.. code-block:: bash

    # List the catalog; filter, sort, and export for scripting
    biolm model list
    biolm model list --filter encoder=true --sort model_name
    biolm model list --format json --output models.json

    # Inspect one model: actions, input type, and full JSON schemas
    biolm model show esmfold --include-schemas

    # Print a copy-pasteable example for a model and action
    biolm model example progen2-oas --action generate

``biolm model list`` accepts ``--filter`` (e.g. ``encoder=true``,
``model_name=esm2``), ``--sort`` (prefix a field with ``-`` for descending), and
``--format`` (``table``, ``json``, ``yaml``, or ``csv``, optionally written to a
file with ``-o``). ``biolm model show`` prints a model's metadata; add
``--include-schemas`` to see the exact request and response shapes for every
action. ``biolm model example`` emits ready-to-run code — in Python, Markdown,
RST, or JSON — so you can go from "which model?" to a working call in one step.

See :doc:`../cli/model` for the complete command reference.

From Python
===========

The same discovery lives in the SDK, which is handy inside notebooks and scripts:

.. code-block:: python

    from biolm import list_models, get_example, Model

    # The full catalog as a list of dicts (name, slug, supported actions, ...)
    models = list_models()

    # A copy-pasteable example for a model and action
    print(get_example("esmfold", action="predict"))

    # Or ask a Model instance directly
    model = Model("progen2-oas")
    print(model.get_example(action="generate"))   # one action
    print(model.get_examples())                    # every supported action

``list_models()`` returns the catalog so you can filter it however you like in
Python. ``get_example()`` (and the equivalent ``Model.get_example`` /
``Model.get_examples`` methods) generates usage snippets; all of them accept a
``format=`` argument (``"python"``, ``"markdown"``, ``"rst"``, or ``"json"``).
This means you can confirm a model's action and required arguments and get
working code without leaving your session.

Discovering a local catalog
============================

If you route inference through a self-hosted `biolm-hub
<https://github.com/BioLM/biolm-hub>`_ gateway rather than ``biolm.ai``, point
the CLI and SDK at it with ``biolm hub set`` (defaulting to
``http://127.0.0.1:8000``). Afterward, ``biolm model list``, ``show``, and
``example`` — and the Python discovery helpers — resolve against the hub's
catalog, so you discover exactly the models your gateway serves. You can also
browse them in a browser at ``http://127.0.0.1:8000/catalog`` while the gateway
is running. See :doc:`biolm-hub` for setup details.

Where to go next
================

- :doc:`what-are-biolms` — the four actions and the call pattern, with examples.
- :doc:`running-inference` — run a model from Python or the CLI.
- :doc:`sequence-and-structure-data` — load sequences and structures from local files.
- :doc:`finetuning-models` — train models on your labeled data.
- :doc:`biolm-hub` — route inference through a local or self-hosted gateway.
- :doc:`quickstart` — install, authenticate, and make your first call.
- :doc:`../cli/model` — complete ``biolm model`` command reference.
- :doc:`../cli/hub` — route discovery and inference through a local gateway.
- :doc:`../sdk/models` — the full ``Model`` API, including ``get_example``.
- `BioLM model catalog <https://biolm.ai/models>`_ — per-model actions, input
  types, params, and request/response schemas.
