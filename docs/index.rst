===================================
Welcome to biolm-sdk Documentation
===================================

.. include:: getting-started/quickstart.rst
   :start-line: 1
   :end-line: 27

.. include:: getting-started/overview.rst
   :start-line: 1
   :end-line: 17

Documentation Authors
---------------------

If you're writing or updating documentation, see the :doc:`authoring-guide` for information on where to write documentation vs where auto-generated documentation appears.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:
   :hidden:

   getting-started/installation
   getting-started/authentication
   getting-started/quickstart
   getting-started/concepts
   getting-started/migration-1.0
   tutorials_use_cases/notebooks

.. toctree::
   :maxdepth: 1
   :caption: CLI:
   :hidden:

   cli/login
   cli/logout
   cli/status
   cli/hub
   cli/workspace
   cli/model
   cli/protocol
   cli/dataset

.. toctree::
   :maxdepth: 2
   :caption: SDK:
   :hidden:

   sdk/overview
   sdk/models
   sdk/protocols
   sdk/pipeline
   sdk/workspaces
   sdk/volumes
   sdk/io
   sdk/usage/usage
   sdk/usage/batching
   sdk/usage/error-handling
   sdk/usage/async-sync
   sdk/usage/rate_limiting
   sdk/usage/io
   sdk/faq

.. toctree::
   :maxdepth: 1
   :caption: Protocol Schema:
   :hidden:

   protocols/about
   protocols/inputs
   protocols/execution
   protocols/tasks
   protocols/output
   protocols/schema

.. toctree::
   :maxdepth: 2
   :caption: API References:
   :hidden:

   cli/reference
   sdk/api-reference/index
   resources/rest-api
