.. _cli-overview:

Overview
========

The BioLM command-line interface (CLI) provides convenient access to BioLM services from your terminal.

Documentation Structure
-----------------------

**Where to write CLI documentation:**

- **Usage guides:** ``docs/cli/usage/`` - Write usage guides for each topic (authenticating, workspaces, models, protocols, datasets)
- **This overview page:** ``docs/cli/overview.rst`` - General CLI information and overview

**Auto-generated documentation:**

- **Command reference:** ``docs/cli/reference.rst`` - Auto-generated from ``biolm.cli`` using ``sphinx-click``

**How to update CLI documentation:**

1. **Command reference (auto-generated):** Edit docstrings and help text in ``biolmai/cli.py``. The ``sphinx-click`` extension will pull documentation from the Click command definitions.

2. **Usage guides (manually written):** Edit the files in ``docs/cli/usage/`` to add examples, explanations, and tutorials.

3. **Overview (manually written):** Edit this file (``docs/cli/overview.rst``) for general CLI information.

For more details, see the :doc:`../authoring-guide` section.
