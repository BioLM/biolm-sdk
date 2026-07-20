``biolm.volumes``
=================

There is no local SDK API for BioLM volumes. Modal-backed volumes are mounted
server-side into Jupyter and protocol runtimes; they are not general-purpose
cloud storage that local Python code can list, create, fetch, or delete.

Use the BioLM console or Jupyter environment to manage runtime data. For
protocol results needed locally, use protocol result downloads or exports.

Deprecated compatibility placeholder
------------------------------------

:class:`~biolm.volumes.Volume` remains importable in this major version so
existing ``from biolm import Volume`` imports and constructor calls do not
break. Constructing it emits :class:`DeprecationWarning`. Its ``list``,
``create``, ``get``, and ``delete`` methods raise ``NotImplementedError``
because direct local volume management is unsupported.

Volume compatibility API
------------------------

.. autoclass:: biolm.volumes.Volume
   :members: list, create, get, delete
   :undoc-members:

Volume documentation
--------------------

- :doc:`../guide/protocol-workflows` — run protocols and download results
- :doc:`../api-reference/biolm` — ``biolm.volumes`` module reference
