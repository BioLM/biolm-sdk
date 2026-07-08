``biolm.volumes``
=================

Volumes provide persistent storage for pipeline artifacts and large datasets.
The intended Python API surface is ``list``, ``create``, and ``get`` on
:class:`~biolm.volumes.Volume`.

.. note::
   Python SDK volume management is coming soon. The ``Volume`` class in
   ``biolm.volumes`` is not yet implemented. Use :doc:`../cli/hub` and hub
   storage workflows today.

Intended API
------------

When implemented, ``Volume`` will support:

- ``list()`` — enumerate available volumes
- ``create(name)`` — create a new volume
- ``get(name)`` — fetch volume metadata

API
---

.. autoclass:: biolm.volumes.Volume
   :members: list, create, get
   :undoc-members:

See also
--------

- :doc:`../cli/hub` — hub and storage CLI
- :doc:`../api-reference/biolm` — ``biolm.volumes`` module reference
