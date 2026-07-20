.. _protocol-local-execution:

==========================================
Running Protocols Locally
==========================================

Local execution compiles a **Local Protocol Profile**–compatible YAML file into
a :class:`~biolm.pipeline.DataPipeline`, runs ApiTask stages on your machine,
and returns a wide results table. Install the pipeline extra first:

.. code-block:: bash

    pip install "biolm-sdk[pipeline]"

Supported features and limitations are documented in
:doc:`protocol-local-profile`.


CLI: ``biolm protocol run-local``
==================================

.. code-block:: bash

    biolm protocol run-local my_protocol.yaml --input sequence=MKLLIV
    biolm protocol run-local my_protocol.yaml --input sequences='["MKLLIV","MKTAY"]' --json

``--input`` accepts ``key=value`` pairs. Values are parsed as JSON when
possible (lists, numbers, booleans); otherwise they are treated as plain
strings. Use ``--json`` to print the full ``records`` list. Use
``--output-dir`` to set the pipeline DuckDB cache directory.

For hosted platform runs of a registered protocol slug, use
``biolm protocol run SLUG`` instead — see :doc:`protocol-hosted-execution`.


Python: :meth:`Protocol.execute`
==================================

.. code-block:: python

    from biolm.protocols import Protocol

    protocol = Protocol("my_protocol.yaml")
    result = protocol.execute(inputs={"sequence": "MKLLIV"})

    print(result.run_id)
    print(result.records)           # list[dict] — all rows
    print(result.selected_records)  # rows matching outputs[] rules (if any)
    print(result.dataframe)         # pandas DataFrame

The convenience function :func:`~biolm.protocols.run_local_protocol` accepts a
parsed protocol ``dict`` instead of a file path.


Results and ``outputs[]`` selection
==================================

After a local run, :class:`~biolm.protocols.runtime.LocalRunResult` exposes:

- ``records`` — every row from the final pipeline DataFrame (JSON-safe dicts).
- ``selected_records`` — union of rows selected by the protocol's ``outputs[]``
  rules (``where``, ``order_by``, ``limit``), using the same logic as MLflow
  protocol logging.
- ``output_selections`` — per-rule breakdown (``rule_index``, ``rule``,
  ``records``).

Example protocol snippet:

.. code-block:: yaml

    outputs:
      - where: "${{ mean_plddt > 70 }}"
        order_by:
          - field: mean_plddt
            order: desc
        limit: 10

.. code-block:: python

    result = protocol.execute(inputs={"sequences": ["MKLLIV", "MKTAY"]})
    top = result.selected_records
    for rule_sel in result.output_selections:
        print(rule_sel.rule_index, len(rule_sel.records))


SeqFrame bridge (optional)
==================================

If you have the seqframe extra installed, materialize results as a
:class:`~biolm.seqframe.SeqFrame`:

.. code-block:: python

    sf = result.to_seqframe()  # requires pip install "biolm[seqframe]"


Unsupported features
==================================

Protocols using ``gather``, ``foreach``, ``subtasks``, non-ApiTask runners, or
``tasks.<id>`` expressions in templates fail fast at compile time with
``UnsupportedProtocolFeature``. See :doc:`protocol-local-profile` for the full
matrix.


See also
==================================

- :doc:`protocol-local-profile` — Local Protocol Profile v1 reference
- :doc:`protocol-authoring` — validate before running
- :doc:`../sdk/protocols` — API reference
- :doc:`../cli/protocol` — CLI options for ``run-local``
