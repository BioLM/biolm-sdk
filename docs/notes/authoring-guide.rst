:orphan:

Documentation Authoring Guide
==============================

This guide explains where to write documentation and where auto-generated documentation appears.

Documentation Structure
-----------------------

**Manually written:**

- Introduction — ``docs/intro/`` (onboarding and client usage)
- CLI — ``docs/cli/index.rst`` (command index) and ``docs/cli/*.rst`` (per-command pages)
- SDK — ``docs/sdk/index.rst`` and ``docs/sdk/*.rst`` (curated module reference)
- YAML — ``docs/yaml/`` (declarative file-format references)
- Changelog — ``docs/changelog.rst`` (includes root ``CHANGELOG.md``)
- Notes — ``docs/notes/`` (misc pages; listed in nav only where needed, e.g. migration under Release notes)

**Auto-generated:**

- ``docs/api-reference/`` — full module tree (sphinx-apidoc from ``biolm/``)
- ``docs/reference/cli.rst`` — monolithic CLI dump (legacy; prefer ``docs/cli/``)

Where to Write Documentation
-----------------------------

CLI Documentation
~~~~~~~~~~~~~~~~~

**Write here:**

- ``docs/cli/index.rst`` — CLI command index (grouped tables, links to per-command pages)
- ``docs/cli/*.rst`` — Per-command pages (e.g. ``cli/model.rst`` for ``biolm model``)

**Auto-generated:**

- ``docs/reference/cli.rst`` — Full command reference (edit docstrings in ``biolm/cli.py``)

SDK Documentation
~~~~~~~~~~~~~~~~~

**Write here (Introduction nav):**

- ``docs/intro/*.rst`` — Onboarding, client interfaces, batching, errors, etc.

**Write here (SDK nav):**

- ``docs/sdk/index.rst`` — SDK landing page
- ``docs/sdk/*.rst`` — Per-module reference pages (edit docstrings in ``biolm/`` for API detail)

Each SDK page should follow: **what it is** → **when to use** → **minimal example** →
**primary autoclass/autofunction** → **see also** (intro, CLI, yaml, api-reference).

In ``.rst`` files, use double backticks for inline code (``like this``), not Markdown
single backticks. Do not nest bold (``**``) inside literals — Sphinx renders the
backticks literally.

**Auto-generated:**

- ``docs/api-reference/`` — Full module index (sphinx-apidoc)

Protocol Schema
~~~~~~~~~~~~~~~

**Write here:**

- ``docs/yaml/protocol-schema.rst`` — Protocol YAML structure, semantics, and JSON schema
- ``docs/sdk/protocols.rst`` — When to use protocols in Python/CLI (links to Reference)

**Schema source:** ``schema/protocol_schema.json``

Building docs
-------------

.. code-block:: bash

   make docs          # HTML
   tox -e build_docs  # CI equivalent

``make docs`` runs ``sphinx-apidoc -o docs/api-reference biolm`` before building.
