Protocol Schema Reference
=========================

A protocol defines a workflow for the BioLM server: inputs, tasks (model or gather), and optional MLflow outputs. Validate YAML with ``biolm protocol validate`` or the :class:`biolm.protocols.Protocol` class.

Minimal example
---------------

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

Top-level structure
-------------------

**Required keys:** ``name``, ``inputs`` (map of input names to InputSpec), and ``tasks`` (array of model or gather tasks).

**Optional keys:** ``description``, ``example_inputs``, ``progress``, ``ranking``, ``writing``, ``concurrency``, ``outputs`` (MLflow), ``schema_version`` (default 1).

Inputs (InputSpec)
------------------

Each entry under ``inputs`` is an InputSpec: a ``type`` (e.g. text, float, integer, boolean, select, list_of_str, pdb_text, multiselect) plus optional ``label``, ``required``/``optional``, ``help_text``, ``initial``, ``min``/``max``, ``min_length``/``max_length``, ``choices`` (for select/multiselect), ``advanced``, and ``step``.

.. code-block:: yaml

    inputs:
      sequences:
        type: list_of_str
        label: Protein sequences
        required: true
      temperature:
        type: float
        label: Sampling temperature
        initial: 0.7
        min: 0
        max: 2

Tasks
-----

**Model task** — calls a model. Use ``slug`` and ``action`` (e.g. esmfold / predict). The request body must include ``items`` (array, object, or expression) and can include ``params``. Optional: ``response_mapping``, ``depends_on``, ``foreach``, ``skip_if``, ``skip_if_empty``, ``subtasks``.

.. code-block:: yaml

    - id: predict
      slug: esmfold
      action: predict
      request_body:
        items: "${{ sequences }}"
        params: {}

**Gather task** — collects fields from another task or from an input. Set ``type`` to ``gather``, ``from`` to a task ID or input name, and ``fields`` to the list of field names. Optional: ``into``, ``depends_on``, ``skip_if_empty``.

.. code-block:: yaml

    - type: gather
      from: predict
      fields: [pdb, mean_plddt]

Execution
---------

Tasks run in order. You can make a task wait for others (``depends_on``), run conditionally (``skip_if``, ``skip_if_empty``), or repeat over a list (``foreach``).

.. code-block:: yaml

    tasks:
      - id: encode
        slug: esm2-8m
        action: encode
        request_body:
          items: "${{ sequences }}"
      - id: predict
        slug: esmfold
        action: predict
        depends_on: [encode]
        request_body:
          items: "${{ tasks.encode.response.results }}"

Response mapping and outputs
----------------------------

On model tasks, ``response_mapping`` can be a string expression (e.g. to pull a field from the API response) or an object with ``path``, optional ``explode``, and optional ``prefix``. The top-level ``outputs`` array defines MLflow output rules for logging.

.. code-block:: yaml

    - id: predict
      slug: esmfold
      action: predict
      request_body:
        items: "${{ sequences }}"
      response_mapping: "${{ response.results[*].pdb }}"

Full JSON Schema
----------------

The formal JSON Schema for BioLM Protocol YAML is defined below. The Python client validates protocol YAML against this schema.

.. literalinclude:: ../../schema/protocol_schema.json
   :language: json
   :linenos:
   :caption: protocol_schema.json
