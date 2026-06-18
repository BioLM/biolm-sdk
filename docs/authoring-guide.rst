Documentation Authoring Guide
==============================

This guide explains where to write documentation and where auto-generated documentation appears.

Documentation Structure
-----------------------

**Manually Written Documentation:**
  - Getting Started guides (``docs/getting-started/``)
  - CLI usage guides (``docs/cli/usage/``)
  - SDK usage guides (``docs/sdk/usage/``)
  - Protocol usage guides (``docs/protocols/``)
  - Overview pages (``docs/cli/overview.rst``, ``docs/sdk/overview.rst``, ``docs/protocols/about.rst``)
  - Resources (``docs/resources/``)

**Auto-Generated Documentation:**
  - CLI reference (``docs/cli/reference.rst``) - Generated from ``biolm.cli`` using ``sphinx-click``
  - API reference (``docs/api-reference/``) - Generated from Python code using ``sphinx-apidoc``
  - Protocol YAML reference - Content in ``docs/protocols/``; full JSON Schema in :doc:`protocols/schema`

Where to Write Documentation
-----------------------------

CLI Documentation
~~~~~~~~~~~~~~~~~

**Write here:**
  - ``docs/cli/overview.rst`` - CLI overview and general information
  - ``docs/cli/usage/*.rst`` - Usage guides for each topic (authenticating, workspaces, models, protocols, datasets)

**Auto-generated:**
  - ``docs/cli/reference.rst`` - Command reference (uses ``sphinx-click`` directives pointing to ``biolm.cli``)

**Note:** The CLI reference page uses ``.. click::`` directives that pull documentation from the Click command definitions in ``biolmai/cli.py``. To update command documentation, edit the docstrings and help text in the CLI code.

SDK Documentation
~~~~~~~~~~~~~~~~~

**Write here:**
  - ``docs/sdk/overview.rst`` - SDK overview and general information
  - ``docs/sdk/usage/*.rst`` - Usage guides (batching, error-handling, async-sync, rate_limiting, io, usage; disk output is covered in the Usage page)

**Auto-generated:**
  - ``docs/api-reference/`` - API reference (generated from Python docstrings using ``sphinx-apidoc``)

**Note:** The API reference is generated from docstrings in the Python code. To update API documentation, edit docstrings in ``biolmai/*.py`` files.

Protocol Schema Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Write here:**
  - ``docs/protocols/about.rst``, ``inputs.rst``, ``execution.rst``, ``tasks.rst``, ``output.rst`` - Protocol YAML structure and semantics

**Schema:** The formal JSON Schema is in :doc:`protocols/schema`; the client validates protocol YAML against it.
