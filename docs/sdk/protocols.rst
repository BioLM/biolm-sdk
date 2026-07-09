``biolm.protocols``
===================

Protocol YAML workflows define multi-step BioLM jobs: inputs, ordered tasks, and
optional MLflow outputs. The :mod:`biolm.protocols` package covers validation,
**local execution** (via the pipeline stack), and **hosted execution** (via the
BioLM platform API).

Package layout
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - :mod:`biolm.protocols`
     - Public exports: ``Protocol``, ``ProtocolClient``, ``run_local_protocol``, validation types
   * - :mod:`biolm.protocols.validation`
     - JSON Schema + semantic validation
   * - :mod:`biolm.protocols.model`
     - :class:`~biolm.protocols.Protocol` — load YAML, validate, execute locally, inspect
   * - :mod:`biolm.protocols.runs`
     - :class:`~biolm.protocols.ProtocolClient` — submit hosted runs
   * - :mod:`biolm.protocols.runtime`
     - Local compiler + executor (``biolm[pipeline]`` required)

Legacy import paths ``biolm.protocol_runs`` and ``biolm.protocol_runtime`` remain
as compatibility shims.

When to use which
-----------------

**Local (SDK + pipeline)**

- ``biolm protocol validate`` — YAML/schema checks
- ``biolm protocol run PROTOCOL.yaml --input key=value`` — compile and run locally
- :meth:`Protocol.execute <biolm.protocols.Protocol.execute>` / :func:`~biolm.protocols.run_local_protocol`
- Requires ``pip install "biolm-sdk[pipeline]"``
- Supported features: see :doc:`../notes/local-protocol-profile-v1`

**Hosted (platform API)**

- :func:`biolm.run_protocol` — submit by slug and block until results
- :class:`~biolm.protocols.ProtocolClient` — submit, poll, download, cancel
- Full protocol feature set (gather, foreach, task-output expressions, etc.)

Schema reference
----------------

Field-level protocol schema (inputs, tasks, execution, outputs, full JSON Schema):
:doc:`../yaml/protocol-schema`.

Examples
--------

Validate a protocol file:

.. code-block:: bash

   biolm protocol validate my-protocol.yaml

Validate from Python:

.. code-block:: python

   from biolm.protocols import Protocol

   result = Protocol.validate("my-protocol.yaml")
   if not result.is_valid:
       for err in result.errors:
           print(err.path, err.message)

Run locally from Python:

.. code-block:: python

   from biolm.protocols import Protocol

   protocol = Protocol("my-protocol.yaml")
   result = protocol.execute(inputs={"sequence": "MKLLIV"})
   print(result.records)

Run locally from the CLI:

.. code-block:: bash

   pip install "biolm-sdk[pipeline]"
   biolm protocol run my-protocol.yaml --input sequence=MKLLIV --json

Run on the platform from Python:

.. code-block:: python

   from biolm import run_protocol

   results = run_protocol("my-protocol-slug", inputs={"sequences": ["MKTAYIAKQRQ"]})

Programmatic hosted runs
------------------------

For progress tracking, cancellation, and result download, use
:class:`~biolm.protocols.ProtocolClient` directly:

.. code-block:: python

   from biolm.protocols import ProtocolClient

   client = ProtocolClient()
   run = client.submit("my-protocol-slug", inputs={"sequences": ["MKTAYIAKQRQ"]})
   run.wait()
   print(run.results())

API
---

.. autoclass:: biolm.protocols.Protocol
   :members: validate, execute
   :undoc-members:

.. autofunction:: biolm.run_protocol

.. autofunction:: biolm.protocols.run_local_protocol

.. autoclass:: biolm.protocols.ProtocolClient
   :members: submit, run_and_wait, get_run, list
   :undoc-members:

.. autoclass:: biolm.protocols.ProtocolRun
   :members: wait, results, cancel, download
   :undoc-members:

.. autoclass:: biolm.protocols.runtime.LocalRunResult
   :members:
   :undoc-members:

See also
--------

- :doc:`../cli/protocol` — CLI validate and run
- :doc:`../yaml/protocol-schema` — protocol YAML schema
- :doc:`../notes/local-protocol-profile-v1` — local execution profile (v1)
- :doc:`../api-reference/biolm.protocols` — full package reference
