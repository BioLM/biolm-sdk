``biolm.protocols``
===================

Protocol YAML workflows define multi-step BioLM jobs: inputs, ordered tasks, and
optional MLflow outputs. Use them from the CLI, the :class:`biolm.protocols.Protocol`
class, or :class:`biolm.protocol_runs.ProtocolClient` for programmatic submission.

When to use which
-----------------

- ``biolm protocol validate`` — quick YAML checks against the JSON schema
- ``biolm protocol run`` — submit and wait from the terminal
- ``Protocol`` — load, validate, and inspect YAML locally
- ``ProtocolClient`` / ``run_protocol()`` — submit runs from Python with progress and results

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

Run from Python:

.. code-block:: python

   from biolm import run_protocol

   results = run_protocol("my-protocol-slug", inputs={"sequences": ["MKTAYIAKQRQ"]})

Programmatic runs
-----------------

For progress tracking, cancellation, and result download, use
:class:`~biolm.protocol_runs.ProtocolClient` directly:

.. code-block:: python

   from biolm.protocol_runs import ProtocolClient

   client = ProtocolClient()
   run = client.submit("my-protocol-slug", inputs={"sequences": ["MKTAYIAKQRQ"]})
   run.wait()
   print(run.results())

API
---

.. autoclass:: biolm.protocols.Protocol
   :members: validate
   :undoc-members:

.. autofunction:: biolm.run_protocol

.. autoclass:: biolm.protocol_runs.ProtocolClient
   :members: submit, run_and_wait, get_run, list
   :undoc-members:

.. autoclass:: biolm.protocol_runs.ProtocolRun
   :members: wait, results, cancel, download
   :undoc-members:
   :noindex:

See also
--------

- :doc:`../cli/protocol` — CLI validate and run
- :doc:`../yaml/protocol-schema` — protocol YAML schema
- :doc:`../api-reference/biolm` — ``biolm.protocol_runs`` module reference
