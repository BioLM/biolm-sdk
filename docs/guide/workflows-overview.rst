.. _workflows-overview:

========================================
Orchestrating molecular design workflows
========================================

You already know how to make a single model call — bind a
:class:`~biolm.models.Model`, hand it some sequences, and read the results.
Real design campaigns rarely stop there. You generate candidates, score them,
filter, embed, cluster, and score again. This page helps you choose *how* to
compose those steps without hand-rolling glue code, retries, and caching every
time.

There are four ways to run BioLM work. Three are execution tiers; the fourth,
the CLI, is a front-end over the same tiers rather than a separate engine.

- **Model one-off** — a single call to one endpoint, in your process.
- **Protocol** — a declarative YAML workflow that runs server-side on BioLM.
- **Pipeline** — a multi-stage Python framework that runs locally with caching
  and resume.
- **CLI** — ``biolm ...`` commands that drive Models and Protocols from your
  terminal.

Choosing a path
===============

.. list-table::
   :header-rows: 1
   :widths: 16 16 16 16 14 22

   * - 
     - Model one-off
     - Protocol
     - Pipeline
     - CLI
   * - Where it runs
     - Your process
     - BioLM servers
     - Your machine
     - Wraps Model/Protocol
   * - Defined in
     - Python call
     - YAML spec
     - Python configs
     - Shell command
   * - Caching / resume
     - None
     - Server-managed run
     - DuckDB, resumable
     - Inherits target
   * - Install
     - ``biolm-sdk``
     - ``biolm-sdk``
     - ``biolm-sdk[pipeline]``
     - ``biolm-sdk``
   * - CLI support
     - Yes
     - Partial
     - No
     - —
   * - Best for
     - One prediction/embed
     - Shareable server jobs
     - Local iterative design
     - Scripting, CI, ad hoc

Model one-off
=============

**When to use:** you need one prediction, embedding, or generation and you want
the result back in memory now. No orchestration, no persistence.

.. code-block:: python

   from biolm import Model

   result = Model("esmfold").predict(type="sequence", items=["MKTAYIAKQRQ"])

This is the foundation every other tier builds on. For the full menu of
synchronous and async clients — ``Model``, ``biolm()``, ``BioLMApi``,
``BioLMApiClient`` — see :doc:`../guide/client-interfaces`.

Protocol
========

**When to use:** you want a multi-step workflow that runs *on BioLM's
infrastructure*, defined declaratively so it is reproducible, shareable, and
independent of your local machine. Protocols are ideal when a run should
outlive your terminal session or be handed to a teammate as a slug.

.. code-block:: python

   from biolm import run_protocol

   results = run_protocol("my-protocol-slug", inputs={"sequences": ["MKTAYIAKQRQ"]})

A protocol is a YAML document describing inputs, ordered tasks, and outputs. The
server manages execution and run state. Author and validate specs against the
field-level :doc:`../yaml/protocol-schema`, and see :doc:`../sdk/protocols` for
programmatic submission, progress tracking, and result download.

Pipeline
========

**When to use:** you are iterating locally on a design campaign — generate,
score, filter, cluster — and you want DuckDB caching so re-runs skip completed
work, plus resumability when something fails midway.

The pipeline framework is **opt-in**. Install the extra, or importing
``biolm.pipeline`` raises an ``ImportError`` listing the missing dependencies:

.. code-block:: bash

   pip install "biolm-sdk[pipeline]"

For a single cached step, the convenience wrappers mirror a Model call but
return a DataFrame and persist results:

.. code-block:: python

   from biolm.pipeline import Predict

   df = Predict("temberture-regression", sequences=["MKTAYIAKQRQ"], extractions="prediction")

For multi-stage design, compose configs. For example, a
``SaturationMutagenesisConfig`` builds a single-mutant library and scores it
inside a ``GenerativePipeline``:

.. code-block:: python

   from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

   config = SaturationMutagenesisConfig(parent_sequence="MKTAYIAKQRQ", top_n=10)
   results = GenerativePipeline(configs=[config]).run()

The full PETase and antibody design walkthroughs live in :doc:`../sdk/pipeline`.
For a step-by-step saturation mutagenesis recipe, see :doc:`saturation-mutagenesis`.

CLI
===

**When to use:** you want to drive **Model** calls from the terminal — for quick
experiments, shell scripts, or CI — without writing a Python file. The CLI is a
front-end, not a fourth engine: ``biolm model`` calls the same endpoints as
:class:`~biolm.models.Model`.

Protocol authoring and validation also work from the CLI (``biolm protocol
init``, ``show``, ``validate``, ``log``). Submitting and monitoring runs is
**Python-only today** — ``biolm protocol run`` and ``biolm protocol list`` are
placeholders. See :doc:`protocol-workflows`.

.. code-block:: bash

   biolm model run esmfold predict -i sequences.fasta -o results.json
   biolm protocol validate my-protocol.yaml

See :doc:`../cli/model` and :doc:`../cli/protocol` for full command reference.

Rule of thumb
=============

Work down this checklist and stop at the first match:

#. **Just one call?** Use a **Model** one-off.
#. **From a shell script or CI, no Python file?** Use the **CLI** for model
   calls (``biolm model``). Protocol runs still require Python today.
#. **Needs to run server-side, be reproducible, or shared as a slug?** Use a
   **Protocol**.
#. **Iterating locally with generate → score → filter loops, and you want
   caching and resume?** Use a **Pipeline**.
#. **Still unsure?** Start with a Protocol for portability; reach for a Pipeline
   once local caching and fast iteration matter more than server execution.

.. tip::

   Want to see these paths applied end-to-end on real science problems? Browse
   the runnable notebooks and case studies at
   `jupyter.biolm.ai <https://jupyter.biolm.ai>`_.
