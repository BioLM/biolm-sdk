.. _running-inference:

=========================
Running an inference
=========================

You have picked a model and confirmed the action it supports. This guide is
about the next step: actually *running* one inference and getting results back.
It focuses on the mechanics — how to invoke a model from Python or the command
line, how to hand it your input, how to pass parameters, and how to read (or
save) what comes back. The four actions themselves are covered in
:doc:`what-are-biolms`, and the two client styles in :doc:`client-interfaces`;
here we assume you know *which* model and action you want and just need to fire
it.

There are two ways to run a single inference: the Python ``Model`` class and the
``biolm model run`` CLI command. Both hit the same API, accept the same input
``type`` and ``params``, and return the same results — so the choice is about
where you work, not what you can do.

From Python
===========

:class:`~biolm.models.Model` is the recommended interface. Construct it once
with a slug, then call the action method — ``encode``, ``predict``,
``generate``, or ``lookup`` — with ``items``, a ``type``, and optional
``params``:

.. code-block:: python

    from biolm import Model

    model = Model("esm2-8m")
    result = model.encode(type="sequence", items="MSILVTRPSPAGEEL")

``items`` is a single value or a list; ``type`` names the input kind the model
expects (``"sequence"``, ``"context"``, ``"pdb"``, …) unless your items are
already dicts. ``lookup`` is the exception — it retrieves reference data, so it
takes ``query=`` instead of ``items=``. Reusing one ``Model`` across calls keeps
the underlying HTTP connection pool warm, which matters once you make more than
one request.

You may also see the legacy one-shot :func:`~biolm.biolm` function, which runs a
call in a single expression:

.. code-block:: python

    from biolm import biolm

    result = biolm(entity="esm2-8m", action="encode",
                         type="sequence", items="MSILVTRPSPAGEEL")

It still works and is fine for a throwaway snippet, but ``Model`` is the
interface to prefer for anything beyond a single call.

From the command line
=====================

``biolm model run`` runs the same actions without writing code, which is ideal
for shell scripts, quick checks, and piping between tools:

.. code-block:: bash

    biolm model run <slug> <action> -i INPUT --params JSON -o OUTPUT --format FMT

The positional ``<action>`` is one of ``encode``, ``predict``, ``generate``, or
``lookup``. Reach for the CLI when your data already lives in a FASTA or CSV
file, when you want results written straight to disk, or when inference is one
stage in a larger pipeline of command-line tools. Reach for Python when you need
to shape inputs, inspect results programmatically, or feed the output into
further analysis.

Feeding in input
=================

The CLI accepts input three ways through ``-i/--input``:

- **A file** — ``-i sequences.fasta``. The format is auto-detected from the
  extension (``.fasta``, ``.csv``, ``.json``/``.jsonl``, ``.pdb``); override it
  with ``--input-format`` if detection is ambiguous.
- **Standard input** — ``-i -`` reads from a pipe. Because there is no filename
  to sniff, you must state the format with ``--input-format`` (or ``--format``
  as a fallback).
- **Inline JSON** — echo a JSON object into stdin, the quickest way to run one
  item:

.. code-block:: bash

    echo '{"sequence": "MSILVTRPSPAGEEL"}' | biolm model run esm2-8m encode -i - --format json

FASTA is the natural choice for batches of sequences; JSON lets you attach
per-item fields; CSV suits tabular inputs. In Python the equivalent is simply
the ``items`` argument — a string, a list, or dicts you build yourself (the
:doc:`../sdk/io` helpers such as ``load_fasta`` turn files into that list). See
:doc:`sequence-and-structure-data` for a full walkthrough.

Passing parameters
===================

Model-specific options — temperature, number of samples, normalization — go in
``params``. In Python it is a dict:

.. code-block:: python

    model = Model("progen2-oas")
    result = model.generate(type="context", items="M",
                            params={"temperature": 0.7, "num_samples": 2})

On the CLI, ``--params`` takes either an inline JSON string or a path to a JSON
file:

.. code-block:: bash

    biolm model run progen2-oas generate -i - --input-format json \
        --params '{"temperature": 0.7, "num_samples": 2}'

Valid keys vary by model and action; confirm them in the catalog or with
``biolm model show`` (see :doc:`choosing-models`).

Reading and saving results
==========================

A single-item call returns one result dict; a list of items returns a list of
dicts in the same order as the input, so you can zip results back to their
sequences. Per-item failures come back as error dicts alongside the successes
rather than raising — see :doc:`error-handling` for how to detect and recover
from them.

For large jobs you often want results on disk instead of in memory. Pass
``output="disk"`` (and optionally ``file_path=``) to any action; the SDK streams
each result to a JSONL file as it arrives:

.. code-block:: python

    model = Model("esmfold")
    model.predict(type="sequence", items=big_list,
                  output="disk", file_path="folded.jsonl")

Without ``file_path``, the SDK writes to ``<slug>_<action>_output.jsonl``. The
CLI does the same through ``-o/--output``; ``--format`` (``json``, ``fasta``,
``csv``, ``pdb``) picks the output shape, and it is auto-detected from the
output file's extension when omitted. With no ``-o``, results print to stdout so
you can pipe them onward.

Where to go next
================

- :doc:`what-are-biolms` — the four actions and the ``items``/``type``/``params`` pattern.
- :doc:`how-biolms-work` — slugs, schemas, and how items are normalized.
- :doc:`choosing-models` — confirm a model's actions, input type, and params.
- :doc:`sequence-and-structure-data` — load FASTA, CSV, JSON, and PDB into ``items``.
- :doc:`managing-datasets` — upload and download platform datasets via MLflow.
- :doc:`error-handling` — inspect and recover from per-item and request errors.
- :doc:`client-interfaces` — sync vs. async and when to use each.
- :doc:`../sdk/models` — the full ``Model`` API reference.
- :doc:`../cli/model` — complete ``biolm model`` command reference.
