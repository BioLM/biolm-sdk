Full JSON Schema
================

The formal JSON Schema for BioLM Protocol YAML (inputs, tasks, execution, MLflow outputs) is defined below. The Python client validates protocol YAML against this schema (e.g. ``biolm protocol validate``).

.. literalinclude:: ../../schema/protocol_schema.json
   :language: json
   :linenos:
   :caption: protocol_schema.json
