``biolm.seqframe``
==================

Sequence-centric dataframe abstraction (optional ``biolm-sdk[seqframe]`` extra).

See :doc:`../guide/seqframe` for a usage guide,
:doc:`../yaml/seqframe-schema` for Parquet metadata fields, and
:doc:`../api-reference/biolm.seqframe` for the module reference.

.. code-block:: python

    from biolm import SeqFrame

    sf = (
        SeqFrame.from_fasta("proteins.fasta")
        .query.filter("length < 300")
        .query.limit(10)
    )
    ds = sf.to_dataset("my-proteins")
    sf2 = ds.open_seqframe()
