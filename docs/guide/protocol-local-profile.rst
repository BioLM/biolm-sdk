.. _protocol-local-profile:

==========================================
Local Protocol Profile v1
==========================================

This page defines what **local-compatible** means for BioLM Protocol YAML when
executed via :mod:`biolm.protocols.runtime` (SDK-side) rather than the hosted
``biolm_web`` runner.


Purpose
==================================

Phase 1 provides a minimal **Protocol YAML → pipeline compiler** and **local
executor** so users can run supported protocols on their machine using
``biolm[pipeline]``, DuckDB caching, and the existing ``DataPipeline`` /
``PredictionStage`` stack.


Supported (v1)
==================================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Notes
   * - ``ApiTask`` with ``slug`` + ``action``
     - Required task ``id``
   * - ``depends_on``
     - Stage ordering preserved in compiled pipeline
   * - ``request_body.items`` / ``params``
     - ``${{ }}`` expressions evaluated against **inputs only**
   * - ``response_mapping``
     - Literal keys, ``${{ response... }}`` JMESPath subset, or ``{path: ...}`` objects
   * - ``outputs[]`` selection
     - ``where`` / ``order_by`` / ``limit`` applied to final ``records`` after the run
   * - Results
     - Wide table: ``sequence`` + metric columns from mappings


Inputs contract
==================================

Users pass a plain ``dict`` to :meth:`Protocol.execute` or
:func:`run_local_protocol`. ``InputSpec`` entries in the YAML are **metadata
only** for v1 (defaults may be merged when inputs omit a key).

Primary sequence input is resolved as:

1. First ``list_of_str`` input from protocol ``inputs``
2. Key named ``sequences`` or ``sequence``
3. Single text input coerced to a one-element list


Results contract
==================================

:class:`~biolm.protocols.runtime.LocalRunResult` exposes:

- ``dataframe`` — pipeline ``get_final_data()`` output
- ``records`` — ``list[dict]`` rows with a ``sequence`` field (compatible with
  MLflow protocol logging)
- ``selected_records`` / ``output_selections`` — when the protocol defines
  ``outputs[]``


Unsupported (fail fast)
==================================

``profile.check_supported()`` raises ``UnsupportedProtocolFeature`` before
compile:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Reason
   * - ``class`` / ``app`` / ``method`` tasks
     - Not ApiTask
   * - ``gather``, ``foreach``, ``subtasks``
     - Control-flow not implemented
   * - ``skip_if``, ``skip_if_empty``
     - Conditional execution deferred
   * - ``tasks.<id>...`` in expressions
     - Task-output references require hosted runtime


Phase 2 deferrals
==================================

- ``ExecutionPlan`` JSON serialization for ``biolm_web`` workers
- Full hosted parity (gather/foreach/subtasks on local runner)
- Streaming ``outputs[]`` collection during execution (local applies rules after the run)


Example protocols
==================================

Encode only
-----------

.. code-block:: yaml

    name: local-encode
    inputs:
      - name: sequences
        type: list_of_str
    tasks:
      - id: encode
        type: ApiTask
        slug: esm2-8m
        action: encode
        request_body:
          items: ${{ sequences }}
        response_mapping:
          embedding: mean_representations

Predict structure
-----------------

.. code-block:: yaml

    name: local-esmfold
    inputs:
      - name: sequence
        type: text
    tasks:
      - id: fold
        type: ApiTask
        slug: esmfold
        action: predict
        request_body:
          items: ${{ [sequence] }}
        response_mapping:
          mean_plddt: mean_plddt
          pdb: pdb

Two-task DAG
------------

.. code-block:: yaml

    name: local-encode-then-score
    inputs:
      - name: sequences
        type: list_of_str
    tasks:
      - id: encode
        type: ApiTask
        slug: esm2-8m
        action: encode
        request_body:
          items: ${{ sequences }}
        response_mapping:
          embedding: mean_representations
      - id: score
        type: ApiTask
        slug: temberture-regression
        action: predict
        depends_on: [encode]
        request_body:
          items: ${{ sequences }}
        response_mapping:
          stability: stability


Usage
==================================

.. code-block:: python

    from biolm.protocols import Protocol

    protocol = Protocol("my_protocol.yaml")
    result = protocol.execute(inputs={"sequences": ["MKLLIV"]})
    print(result.records)

.. code-block:: bash

    pip install "biolm[pipeline]"
    biolm protocol run-local my_protocol.yaml --input sequences='["MKLLIV"]'

Hosted runs of a registered protocol slug use ``biolm protocol run SLUG`` (no
pipeline extra). See :doc:`protocol-hosted-execution`.


SeqFrame bridge (optional)
==================================

After a local run you can materialize a :class:`~biolm.seqframe.SeqFrame`:

.. code-block:: python

    result = protocol.execute(inputs={"sequence": "MKLLIV"})
    sf = result.to_seqframe()  # requires biolm[seqframe]

See :doc:`protocol-local-execution` for details.
