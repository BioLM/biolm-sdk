``biolm.workspaces``
====================

Workspaces group BioLM resources (runs, data, and configuration) under a named
scope. The intended Python API surface is ``list``, ``create``, and ``get`` on
:class:`~biolm.workspaces.Workspace`.

.. note::
   Python SDK workspace management is coming soon. The ``Workspace`` class in
   ``biolm.workspaces`` is not yet implemented. Use :doc:`../cli/workspace` today.

Intended API
------------

When implemented, ``Workspace`` will support:

- ``list()`` — enumerate available workspaces
- ``create(name)`` — create a new workspace
- ``get(name)`` — fetch workspace metadata

API
---

.. autoclass:: biolm.workspaces.Workspace
   :members: list, create, get
   :undoc-members:

See also
--------

- :doc:`../cli/workspace` — workspace CLI (supported today)
- :doc:`../api-reference/biolm` — ``biolm.workspaces`` module reference
