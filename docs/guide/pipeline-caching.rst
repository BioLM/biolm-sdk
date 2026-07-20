.. _pipeline-caching:

==========================================
Pipeline Caching and Resumability
==========================================

*Orchestrating molecular design workflows*

Every pipeline is backed by a local `DuckDB <https://duckdb.org/>`_ database. As
stages run, predictions, embeddings, structures, and generation provenance are
written to that file, so a second run over overlapping sequences reuses the
stored values instead of re-calling the API, and a crashed run resumes instead
of starting over. This page is a deep dive on that cache: where it lives, what
goes in it, and how to resume or reconnect. For the broader picture — choosing a
pipeline class, composing stages, reading results — start with
:doc:`pipeline-workflows`.

Caching is part of the opt-in pipeline extra, so make sure it is installed:

.. code-block:: bash

    pip install "biolm-sdk[pipeline]"

.. contents::
   :local:
   :depth: 1


Where the cache lives
==================================

When you do not pass a ``datastore``, the pipeline provisions one automatically
at:

.. code-block:: text

    .biolm/pipelines/<pipeline_id>/pipeline.duckdb

The ``<pipeline_id>`` is a timestamped, unique run identifier. Everything a run
produces lands in that single DuckDB file, so a cache is trivially portable —
copy the file and you copy the results.

The :attr:`~biolm.pipeline.BasePipeline.metadata` property exposes exactly where
things are, which is what you need to reconnect later:

.. code-block:: python

    pipeline = DataPipeline(sequences=my_sequences)
    pipeline.run()

    meta = pipeline.metadata
    print(meta.pipeline_id)   # "20260716_104512_a1b2c3d4"
    print(meta.cache_dir)     # ".biolm/pipelines/20260716_104512_a1b2c3d4"
    print(meta.db_path)       # ".../pipeline.duckdb"

To choose the location yourself, pass a path as the ``datastore`` — the file is
created if it does not exist and reused if it does:

.. code-block:: python

    pipeline = DataPipeline(sequences=my_sequences, datastore="run.duckdb")


What gets cached
==================================

The DuckDB file holds a handful of tables, each keyed on an internal
``sequence_id`` so results can be joined back to their inputs:

- ``sequences`` — the input sequences, deduplicated by content hash.
- ``predictions`` — scalar prediction values (Tm, ΔΔG, pLDDT, logits, …) with
  model name and prediction type.
- ``embeddings`` — embedding vectors (stored in companion Parquet files, with the
  path recorded here).
- ``structures`` — predicted structures (PDB/CIF) and their pLDDT.
- ``generation_metadata`` — provenance for generated sequences (source model,
  temperature, sampling params, label).
- ``stage_completions`` — which stages finished, with input/output counts. This
  is what makes resume possible.

Because sequences are deduplicated by hash, feeding the same sequence twice — or
across separate runs that share a datastore — stores it once and scores it once.


Only uncached sequences hit the API
==================================

Caching is not just for whole-run re-execution; it works at the granularity of a
single sequence-model pair. Before a :class:`~biolm.pipeline.PredictionStage`
calls the API, it asks the datastore which of its inputs are missing a result
via :meth:`~biolm.pipeline.datastore_duckdb.DuckDBDataStore.get_uncached_sequence_ids`.
That method runs a vectorized anti-join — a single ``LEFT JOIN`` against the
``predictions`` table — and returns only the sequence IDs with no cached value.
Those are the only sequences sent to the model; everything already present is
loaded straight from DuckDB and merged back in.

The practical effect: add ten new sequences to a batch of a thousand you already
scored, and only the ten new ones cost an API call. This applies to the
convenience wrappers too — :func:`~biolm.pipeline.Predict` and
:func:`~biolm.pipeline.Embed` build a one-stage pipeline under the hood, so they
cache and deduplicate on exactly the same path.


Resuming a crashed run
==================================

If a run dies partway through — a kernel restart, a dropped connection, a
``Ctrl-C`` — pass ``resume=True`` to pick up where it left off. Completed stages
(recorded in ``stage_completions``) are reloaded from DuckDB rather than
recomputed, and processing continues from the first unfinished stage:

.. code-block:: python

    pipeline = DataPipeline(
        sequences=my_sequences,
        datastore="run.duckdb",
        resume=True,
    )
    pipeline.run()

``resume=True`` also works on a :class:`~biolm.pipeline.GenerativePipeline`, and
can be passed at call time instead:

.. code-block:: python

    pipeline.run(resume=True)

On resume the pipeline verifies that the reloaded sequence counts match what the
stage expects; a mismatch raises rather than silently continuing with missing
rows, so a corrupted or partial cache fails loudly instead of producing quietly
wrong results.


Reconnecting after the object is gone
==================================

``resume=True`` assumes you still hold the Python pipeline object. After a kernel
death that object is gone, but the DuckDB file is not — and it stores the
pipeline's own definition. :meth:`~biolm.pipeline.BasePipeline.from_db`
rebuilds the entire pipeline (stages, configs, and all) from that saved
definition, so you can resume in a fresh session:

.. code-block:: python

    pipeline = DataPipeline.from_db("run.duckdb")
    pipeline.run(resume=True)

This is the durable-recovery path: as long as the ``pipeline.duckdb`` file
survives, the campaign can be reconstructed and continued from anywhere.


Closing the connection
==================================

DuckDB holds a file handle for the life of the datastore. Pipelines are context
managers, so a ``with`` block closes the connection deterministically when the
block exits — even on error:

.. code-block:: python

    with DataPipeline(sequences=my_sequences, datastore="run.duckdb") as pipeline:
        pipeline.run()
        df = pipeline.results()
    # connection closed here; the .duckdb file is safe to copy or reopen

This matters when you want to reopen the same file (for example with
``from_db``) later in the same process, or hand it to another tool — closing
first avoids a lingering lock on the database.


See also
==================================

- :doc:`pipeline-workflows` — composing and running multi-stage pipelines
- :doc:`structure-conditioned-generation` — generating sequences from a backbone,
  with the same caching and resume behavior
- :doc:`../sdk/pipeline` — Python API reference for pipeline classes, the
  datastore, and metadata
