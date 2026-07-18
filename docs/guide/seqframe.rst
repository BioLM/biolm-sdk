.. _working-with-seqframe:

====================
Working with SeqFrame
====================

*Sequence-centric dataframes over Parquet, with DuckDB-backed query and enrichment.*

Install the optional extra:

.. code-block:: console

    $ pip install "biolm-sdk[seqframe]"

(The same DuckDB / pandas / pyarrow stack is also included in ``biolm-sdk[pipeline]``.)

A **SeqFrame** is to biological sequences what a DataFrame is to tabular data.
It is complementary to :doc:`managing-datasets`: datasets inventory bags of
files; SeqFrame queries, filters, and enriches sequence tables.


Quick start
===========

.. code-block:: python

    from biolm import SeqFrame

    sf = SeqFrame.from_fasta("proteins.fasta")
    sf = sf.query.filter("length < 300").query.limit(10)
    sf.io.to_parquet("filtered.parquet")
    sf2 = SeqFrame.read("filtered.parquet")


Namespaced APIs
===============

- ``sf.query`` — ``filter``, ``select``, ``join``, ``sort``, ``limit``, ``group_by``
- ``sf.io`` — ``to_fasta``, ``to_csv``, ``to_jsonl``, ``to_parquet``
- ``sf.bio`` — length / type helpers (``translate`` needs biopython)
- ``sf.models`` — ``predict`` / ``embed`` via the BioLM API
- ``sf.protocols`` — run a protocol and join results (default join key ``id``;
  ``query.join`` defaults to ``sequence_hash``)
- ``sf.lab`` — LLTP bridge stubs (not implemented yet)


Datasets bridge
===============

Write a SeqFrame into a local dataset (``type: seqframe``) and open it again:

.. code-block:: python

    from biolm.datasets import DatasetClient

    client = DatasetClient()
    ds = sf.to_dataset("my-proteins", client=client, tags=["design"])
    sf2 = ds.open_seqframe()
    # or: SeqFrame.from_dataset(ds)

Resolution uses ``attrs.seqframe_path`` when set, otherwise exactly one
``.parquet`` file under the dataset.

Parquet metadata (``seqframe.version`` / ``seqframe.schema``) is documented in
:doc:`../yaml/seqframe-schema`.

See :doc:`../api-reference/biolm.seqframe` and :doc:`../sdk/seqframe`.
