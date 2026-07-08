biolm.protocols
===============

Protocol YAML workflows define multi-step BioLM jobs: inputs, ordered tasks, and
optional MLflow outputs. Use them from the CLI, the :class:`biolm.protocols.Protocol`
class, or :class:`biolm.protocol_runs.ProtocolClient` for programmatic submission.

When to use which
-----------------

- **``biolm protocol validate``** — quick YAML checks against the JSON schema
- **``biolm protocol run``** — submit and wait from the terminal
- **``Protocol``** — load, validate, and inspect YAML locally
- **``ProtocolClient`` / ``run_protocol()``** — submit runs from Python with progress and results

Schema reference
----------------

Field-level protocol schema (inputs, tasks, execution, outputs, full JSON Schema):
:doc:`../yaml/protocol-schema`.

Examples
--------

Validate a protocol file:

.. code-block:: bash

   biolm protocol validate my-protocol.yaml

Run from Python:

.. code-block:: python

   from biolm import run_protocol

   results = run_protocol("my-protocol-slug", inputs={"sequences": ["MKTAYIAKQRQ"]})

See also :doc:`../cli/protocol` and :doc:`../api-reference/modules` (``biolm.protocols``, ``biolm.protocol_runs``).
