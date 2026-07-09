Protocols
=========

Use the ``biolm protocol`` commands to list, show, run, validate, and initialize protocols.

**Local vs hosted**

- ``biolm protocol validate`` and ``biolm protocol show`` work on YAML files without extra deps.
- ``biolm protocol run`` executes **locally** via :mod:`biolm.protocols.runtime` (requires ``pip install "biolm-sdk[pipeline]"``). It compiles supported protocols to a :class:`~biolm.pipeline.DataPipeline` and runs them on your machine.
- To submit runs to the **hosted** platform, use :func:`biolm.run_protocol` or :class:`~biolm.protocols.ProtocolClient` from Python.

See :doc:`../../sdk/protocols` and :doc:`../../notes/local-protocol-profile-v1` for supported local features.

Command reference: :doc:`../protocol`.
