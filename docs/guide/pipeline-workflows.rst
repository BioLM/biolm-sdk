.. _pipeline-workflows:

==========================================
Pipeline Workflows
==========================================

*Orchestrating molecular design workflows*

Pipelines let you compose a multi-stage molecular design campaign — generate
candidates, score them, filter, embed, cluster — as ordinary Python objects
that run **on your machine**. The framework handles dependency ordering,
concurrency, and (crucially) a local DuckDB cache, so re-runs skip work that
has already completed and a crashed run resumes instead of starting over.

Where a :doc:`protocol <protocol-workflows>` is a declarative YAML graph that
executes on BioLM's servers and is shared as a slug, a pipeline is Python-native
and local: results, embeddings, and structures land in a DuckDB file under your
working directory. Reach for a pipeline when you are iterating quickly and want
caching and resumability more than server-side execution. For the full decision
matrix, see :doc:`workflows-overview`.

.. contents::
   :local:
   :depth: 1


Installation
==================================

The pipeline framework is **opt-in**. It pulls in DuckDB, pandas, numpy, and
pyarrow, which the core API client does not require:

.. code-block:: bash

    pip install "biolm-sdk[pipeline]"

If those dependencies are missing, importing ``biolm.pipeline`` raises an
``ImportError`` that names exactly what to install — the package never
half-imports and leaves you with a confusing traceback:

.. code-block:: text

    ImportError: biolm.pipeline requires optional dependencies that are not
    installed: duckdb, pyarrow.

    Install with:

        pip install 'biolm[pipeline]'

Authentication is the same as everywhere else in the SDK: set ``BIOLM_TOKEN``
(or run ``biolm login``). See :doc:`authentication`.


Choosing a pipeline class
==================================

There are two pipeline entry points, distinguished by *where the sequences come
from*:

- :class:`~biolm.pipeline.DataPipeline` — you already have sequences (a list, a
  DataFrame, or a CSV/FASTA path) and want to predict, filter, cluster, or
  embed them.
- :class:`~biolm.pipeline.GenerativePipeline` — the pipeline *produces* the
  sequences from one or more generation configs, then runs the same downstream
  prediction and filtering stages.

A ``GenerativePipeline`` is driven by **config objects** that describe how
candidates are made. The three you will reach for most:

- ``SaturationMutagenesisConfig`` — enumerate every single-point substitution of
  a parent sequence and rank them by a scoring model (e.g. ΔΔG from ThermoMPNN-D).
- ``IterativeMaskingDMSConfig`` — build multi-point variants by greedy argmax
  masking with a masked language model over several rounds.
- ``DirectGenerationConfig`` — structure- or sequence-conditioned generation with
  models like ProteinMPNN, AntiFold, LigandMPNN, or DSM. See
  :doc:`structure-conditioned-generation`.

Each config's fields and recipes are documented on the SDK reference pages; the
saturation-mutagenesis walkthrough lives at :doc:`saturation-mutagenesis`.


One cached step: ``Predict()``
==================================

For a single prediction with caching, the convenience wrapper mirrors a
``Model`` call but persists results and returns a DataFrame directly:

.. code-block:: python

    from biolm.pipeline import Predict

    df = Predict(
        "temberture-regression",           # model slug is spelled "temberture"
        sequences=["MKTAYIAKQRQ", "MENDEL"],
        extractions="prediction",          # response key to pull out
        columns="tm",
    )

``Predict()`` (and its sibling ``Embed()``) build a one-stage pipeline under the
hood, run it, and hand back ``get_final_data()`` for you — no cache bookkeeping
required.


A multi-stage ``DataPipeline``
==================================

Compose stages explicitly with ``add_prediction()`` and ``add_filter()``. Here
a thermostability prediction feeds a threshold filter and a top-N ranking:

.. code-block:: python

    from biolm.pipeline import DataPipeline
    from biolm.pipeline.filters import ThresholdFilter, RankingFilter

    pipeline = DataPipeline(sequences=my_sequences)
    pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
    pipeline.add_filter(ThresholdFilter("tm", min_value=48.0))
    pipeline.add_filter(RankingFilter("tm", top_n=10))

    pipeline.run()                 # returns dict[str, StageResult]
    df = pipeline.results()        # the surviving sequences as a DataFrame

.. important::

   ``pipeline.run()`` returns a **dict of** :class:`~biolm.pipeline.StageResult`
   objects keyed by stage name — one per stage, carrying input/output counts,
   cache hits, and timing. It does **not** return a DataFrame. To get the final
   rows, call :meth:`~biolm.pipeline.BasePipeline.results` (an alias for
   :meth:`~biolm.pipeline.BasePipeline.get_final_data`).


A generative pipeline
==================================

A ``GenerativePipeline`` takes one or more configs and generates before it
scores. This minimal example builds a single-mutant library and keeps the
top-scoring variants:

.. code-block:: python

    from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

    config = SaturationMutagenesisConfig(
        parent_sequence="MKTAYIAKQRQ",
        scoring_model="esm2-650m",
        score_field="logits",
        top_n=20,
    )

    pipeline = GenerativePipeline(configs=[config])
    stage_results = pipeline.run()       # dict[str, StageResult]
    df = pipeline.get_final_data()       # generated + scored variants

The same rule applies: ``run()`` gives you the per-stage summary dict, and
``results()`` / ``get_final_data()`` gives you the DataFrame. You can still add
downstream stages (``pipeline.add_prediction(...)``, ``pipeline.add_filter(...)``)
before running, exactly as with a ``DataPipeline``.


Caching, resume, and reconnecting
==================================

When you do not pass a ``datastore``, the pipeline creates one automatically at
``.biolm/pipelines/<run_id>/pipeline.duckdb``. Predictions, embeddings, and
structures are cached there, so a second run over overlapping sequences reuses
the stored values instead of re-calling the API. See :doc:`pipeline-caching` for
a full walkthrough of the cache layout, per-sequence deduplication, and recovery
after a kernel death.

To pick up where a previous run left off after a crash or kernel death, pass
``resume=True``. Completed stages are reloaded from DuckDB rather than recomputed:

.. code-block:: python

    pipeline = DataPipeline(sequences=my_sequences, datastore="run.duckdb", resume=True)
    pipeline.run()

The :attr:`~biolm.pipeline.BasePipeline.metadata` property tells you where the
cache lives so you can reconnect later:

.. code-block:: python

    meta = pipeline.metadata
    print(meta.pipeline_id)    # "20260716_104512_a1b2c3d4"
    print(meta.cache_dir)      # ".biolm/pipelines/20260716_104512_a1b2c3d4"
    print(meta.db_path)        # path to the DuckDB file

If the Python object is gone but the DuckDB file remains, rebuild the whole
pipeline from its saved definition with
:meth:`~biolm.pipeline.BasePipeline.from_db` and resume:

.. code-block:: python

    pipeline = DataPipeline.from_db("run.duckdb")
    pipeline.run(resume=True)

Pipelines also work as context managers (``with DataPipeline(...) as p:``) to
close the DuckDB connection deterministically.


No CLI for pipelines
==================================

Pipelines are a **Python-only** interface. Unlike models and protocols, there is
no ``biolm pipeline`` command — pipelines are defined by composing Python
configs and stages, so drive them from a script or notebook. The CLI covers
model calls and protocol authoring; see :doc:`workflows-overview` for how the
tiers compare.


See also
==================================

- :doc:`workflows-overview` — choosing between model calls, protocols, and pipelines
- :doc:`protocol-workflows` — the server-side, YAML-defined counterpart
- :doc:`pipeline-caching` — DuckDB cache layout, resume, and reconnecting
- :doc:`saturation-mutagenesis` — a full saturation-mutagenesis design walkthrough
- :doc:`iterative-masking-dms` — greedy MLM argmax deep mutational scanning
- :doc:`structure-conditioned-generation` — inverse folding and DSM with ``DirectGenerationConfig``
- :doc:`../sdk/pipeline` — Python API reference for pipeline classes and configs
