SeqFrame Metadata Schema Reference
===================================

A **SeqFrame** Parquet file is ordinary Apache Parquet plus namespaced key-value
metadata on the Arrow/Parquet schema. Readers use that metadata to recover
column roles and biological semantics. Files without it are rejected by
:meth:`biolm.seqframe.SeqFrame.read`.

Current metadata version: **``0.1``**.

See :doc:`../guide/seqframe` for usage and
:class:`biolm.seqframe.SeqFrameMetadata` for the Python model.

Encoding
--------

Metadata is stored as Parquet/Arrow schema key-value pairs (byte strings):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Value
   * - ``seqframe.version``
     - Schema version string (currently ``0.1``).
   * - ``seqframe.schema``
     - UTF-8 JSON object describing column mapping and molecule semantics
       (fields below). The JSON may also include a ``version`` field; on read,
       ``seqframe.version`` wins if both are present.

Canonical table columns
-----------------------

On import, SeqFrame writes (at least) these columns into the Parquet table:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Notes
   * - ``id``
     - Row identity (source-derived or generated).
   * - ``sequence``
     - Sequence string.
   * - ``sequence_hash``
     - ``SHA-256`` of the uppercased sequence, truncated to 16 hex chars
       (stable join key).
   * - ``length``
     - ``len(sequence)``.

Additional source columns (CSV/JSONL/protocol results, etc.) are preserved
alongside these. ``sequence_column`` / ``id_column`` in metadata name which
logical columns play those roles (defaults ``sequence`` / ``id``).

``seqframe.schema`` fields
--------------------------

Minimal example (JSON stored under ``seqframe.schema``)::

    {
      "sequence_column": "sequence",
      "id_column": "id",
      "molecule_type": "protein",
      "created_by": "biolm-sdk/1.2.0",
      "extensions": [],
      "version": "0.1"
    }

**Required for a valid SeqFrame file**

Both Parquet keys ``seqframe.version`` and ``seqframe.schema`` must be present.
Missing either causes ``SeqFrame.read`` to raise ``ValueError``.

**Fields inside ``seqframe.schema``**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Required
     - Notes
   * - ``sequence_column``
     - no
     - Default ``sequence``. Name of the sequence string column.
   * - ``id_column``
     - no
     - Default ``id``. Name of the row-id column.
   * - ``molecule_type``
     - no
     - Default ``unknown``. One of ``protein``, ``dna``, ``rna``, ``unknown``.
   * - ``alphabet``
     - no
     - Optional alphabet hint; omitted from JSON when unset.
   * - ``created_by``
     - no
     - Default empty. Typically ``biolm-sdk/<version>`` when written by the SDK.
   * - ``extensions``
     - no
     - Default ``[]``. List of extension identifiers for future plugins.
   * - ``version``
     - no
     - Mirrors ``seqframe.version``; readers prefer the Parquet KV key.

What is not in the schema
-------------------------

- Directory / archive layouts (``.seqframe/``), remote URIs, and multi-file
  datasets — those are inventory concerns for :doc:`dataset-schema`, not
  SeqFrame table metadata.
- Prediction/embedding column conventions (enrichment adds ordinary columns).
- Lab-in-the-Loop (LLTP) envelopes (``sf.lab`` stubs).

Relationship to datasets
------------------------

A local dataset may set ``type: seqframe`` and
``attrs.seqframe_path`` (relative Parquet path). That is **dataset** metadata
(:doc:`dataset-schema`). The Parquet file itself must still carry the SeqFrame
keys above so ``Dataset.open_seqframe()`` / ``SeqFrame.read`` succeed.

Python
------

.. code-block:: python

    from biolm import SeqFrame
    from biolm.seqframe import SeqFrameMetadata, SEQFRAME_VERSION

    sf = SeqFrame.from_fasta("proteins.fasta")
    sf.io.to_parquet("proteins.parquet")

    meta = SeqFrame.read("proteins.parquet").schema
    assert meta.version == SEQFRAME_VERSION
    assert meta.sequence_column == "sequence"
    print(meta.molecule_type, meta.created_by)
