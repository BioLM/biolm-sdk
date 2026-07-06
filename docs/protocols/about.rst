About Protocols
===============

A protocol defines a workflow for the BioLM server: inputs, tasks (model or gather), and optional MLflow outputs. Validate with the Python client (e.g. ``biolm protocol validate``).

**Minimal example:**

.. code-block:: yaml

    name: my-protocol
    inputs:
      sequences:
        type: list_of_str
        label: Sequences
        required: true
    tasks:
      - slug: esmfold
        action: predict
        request_body:
          items: "${{ sequences }}"

**Required top-level keys:** ``name``, ``inputs`` (map of input names to InputSpec — see :doc:`inputs`), and ``tasks`` (array of model or gather tasks — see :doc:`tasks` and :doc:`execution`).

**Optional:** ``description``, ``example_inputs``, ``progress``, ``ranking``, ``writing``, ``concurrency``, ``outputs`` (MLflow), ``schema_version`` (default 1).

See :doc:`schema` for the full JSON Schema.
