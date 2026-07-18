:orphan:

Documentation Authoring Guide
==============================

This guide explains where to write documentation and where auto-generated documentation appears.

Documentation Structure
-----------------------

**Manually written:**

- Guide — ``docs/guide/`` (instructional content; flat layout)
- CLI — ``docs/cli/index.rst`` (command index) and ``docs/cli/*.rst`` (per-command pages)
- SDK — ``docs/sdk/index.rst`` and ``docs/sdk/*.rst`` (curated module reference)
- YAML — ``docs/yaml/`` (declarative file-format references)
- Changelog — ``docs/changelog.rst`` (includes root ``CHANGELOG.md``)
- Notes — ``docs/notes/`` (misc pages; listed in nav only where needed, e.g. migration under Release notes)

**Guide left-nav captions** (pages live under ``docs/guide/``, flat layout):

- Getting started — quickstart, install, auth, concepts, SDK overview, FAQ
- How the client works — client interfaces, batching, errors, concurrency, rate limiting
- Running BioLM inferences — what BioLMs are, how they work, choosing models, running inference, biolm-hub (growing)
- Working with biological data — sequence/structure files, platform datasets (growing)
- Orchestrating molecular design workflows — workflows overview, protocols, pipelines, caching, saturation mutagenesis, iterative masking DMS, structure-conditioned generation (growing)
- Model finetuning — XGBoost on embeddings, DSM stages (growing)

**Manifest ``kind``** (emitted by ``scripts/generate_docs_manifest.py`` for the main site):

- ``guide`` — all slugs under ``guide/``
- ``reference`` — slugs under ``sdk/``, ``cli/``, or ``yaml/``
- ``notes`` — ``changelog`` or slugs under ``notes/``

**Auto-generated:**

- ``docs/api-reference/`` — full module tree (sphinx-apidoc from ``biolm/``)
- ``docs/reference/cli.rst`` — monolithic CLI dump (legacy; prefer ``docs/cli/``)

**Guide snippet doctests:** Offline-safe shapes from the guide (config
construction, ``biolm.io``, ``prepare_items_for_api``, imports) live in
``docs/notes/snippet-doctests.rst`` and run via ``make docs-doctest``. Prefer
adding a ``.. testcode::`` block there when you introduce a new construct-only
example. Live API / CLI network examples stay as narrative ``code-block`` and
are not executed.

Where to Write Documentation
-----------------------------

CLI Documentation
~~~~~~~~~~~~~~~~~

**Write here:**

- ``docs/cli/index.rst`` — CLI command index (grouped tables, links to per-command pages)
- ``docs/cli/*.rst`` — Per-command pages (e.g. ``cli/model.rst`` for ``biolm model``)

**Auto-generated:**

- ``docs/reference/cli.rst`` — Full command reference (edit docstrings in ``biolm/cli/``)

SDK Documentation
~~~~~~~~~~~~~~~~~

**Write here (Guide nav):**

- ``docs/guide/*.rst`` — Onboarding (Getting started) and client mechanics (How the client works)

**Write here (SDK nav):**

- ``docs/sdk/index.rst`` — SDK landing page
- ``docs/sdk/*.rst`` — Per-module reference pages (edit docstrings in ``biolm/`` for API detail)

Each SDK page should follow: **what it is** → **when to use** → **minimal example** →
**primary autoclass/autofunction** → **see also** (guide, CLI, yaml, api-reference).

**Product vs core:** Curated pages under ``docs/sdk/`` document product APIs
(``Model``, ``Protocol``, pipelines, platform types). ``docs/sdk/core.rst`` covers
``biolm.core`` — HTTP clients, ``biolm()``, and legacy imports. Nudge users toward
product APIs; link to core for advanced control.

Package layout:

- ``biolm/models/`` — ``Model`` (``__init__.py``) and ``examples.py`` (catalog and example generation)
- ``biolm/plugins/`` — optional third-party backends (e.g. ``plugins/mlflow`` for datasets and protocol logging). Document on ``sdk/index`` under **Plugins (optional)**; require ``pip install biolm-sdk[mlflow]`` where applicable. Do not import plugins from ``biolm/__init__.py``.

In ``.rst`` files, use double backticks for inline code (``like this``), not Markdown
single backticks. Do not nest bold (``**``) inside literals — Sphinx renders the
backticks literally.

**Auto-generated:**

- ``docs/api-reference/`` — Full module index (sphinx-apidoc)

Protocol Schema
~~~~~~~~~~~~~~~

**Write here:**

- ``docs/yaml/protocol-schema.rst`` — Protocol YAML structure, semantics, and JSON schema
- ``docs/yaml/dataset-schema.rst`` — Local ``dataset.yaml`` field reference
- ``docs/yaml/seqframe-schema.rst`` — SeqFrame Parquet key-value metadata (``seqframe.version`` / ``seqframe.schema``)
- ``docs/sdk/protocols.rst`` — When to use protocols in Python/CLI (links to Reference)

**Schema source:** ``schema/protocol_schema.json``

Building docs
-------------

.. code-block:: bash

   make docs          # HTML
   tox -e build_docs  # CI equivalent

``make docs`` runs ``sphinx-apidoc -o docs/api-reference biolm`` before building.
