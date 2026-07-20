Dataset Schema Reference
========================

A local dataset is a directory containing a ``dataset.yaml`` file. Discovery
looks for these files under configured roots (``~/.biolm/datasets``,
``./.biolm/datasets``, and optional ``dataset_roots`` in ``~/.biolm/config.yaml``).
Validate by loading with :class:`biolm.datasets.DatasetClient`.

See :doc:`../guide/managing-datasets` for usage.

Minimal example
---------------

.. code-block:: yaml

    schema_version: 1
    id: finetuning-v1
    type: files

Recommended example
-------------------

.. code-block:: yaml

    schema_version: 1
    id: finetuning-v1
    description: Paired sequences for ESM2 LoRA fine-tune
    created_at: 2026-07-17T18:00:00+00:00
    type: files
    tags:
      - finetune
      - esm2
    attrs:
      modality: protein
      source_protocol_run: run_abc123

Top-level fields
----------------

**Required**

- ``schema_version`` — integer; currently ``1``.
- ``id`` — stable slug used as the dataset identifier everywhere.

**Optional**

- ``description`` — human-readable context.
- ``created_at`` — ISO-8601 timestamp (``create`` / ``init`` stamp this).
- ``type`` — soft label (default ``files``). Use ``seqframe`` when the primary
  artifact is a SeqFrame Parquet (opened via ``Dataset.open_seqframe()``). Other
  reserved labels include ``protocol_results``; inventory filtering is the only
  core interpretation of ``type``.
- ``tags`` — list of strings for inventory filters.
- ``attrs`` — arbitrary mapping of extra metadata. For SeqFrame datasets,
  ``attrs.seqframe_path`` may point at the Parquet file relative to the dataset
  root (set automatically by ``SeqFrame.to_dataset()``).

What is not in the schema
-------------------------

File manifests, checksums, version history, and remote publish state are out of
scope for schema version 1. Content files live beside ``dataset.yaml`` (typically
under a ``data/`` subdirectory created by ``biolm dataset create``).

Layout
------

.. code-block:: text

    ~/.biolm/datasets/finetuning-v1/
    ├── dataset.yaml
    └── data/
        ├── train.csv
        └── validation.csv

Python
------

.. code-block:: python

    from biolm.datasets import DatasetClient

    client = DatasetClient()
    ds = client.create("finetuning-v1", tags=["finetune"])
    ds.add("train.csv")
    print(ds.id, ds.path, ds.files())
