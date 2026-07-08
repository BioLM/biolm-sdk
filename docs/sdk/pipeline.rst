``biolm.pipeline``
==================

Multi-stage protein design pipelines with DuckDB caching, resumability, and
dependency resolution. See :doc:`../intro/sdk-overview` for SDK onboarding.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

The pipeline framework provides:

- **Multi-stage orchestration** with automatic dependency resolution
- **DuckDB caching** for predictions, embeddings, and structures
- **Resumability** — skip completed stages on re-run
- **Streaming** for large datasets
- **Visualization** — funnel plots, PCA/UMAP, distributions

Quick start
-----------

.. code-block:: python

   from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

   config = SaturationMutagenesisConfig(
       parent_sequence="MKTAYIAKQRQ",
       scoring_model="esm2-650m",
       scoring_action="predict",
       score_field="logits",
       top_n=10,
   )
   pipeline = GenerativePipeline(configs=[config])
   results = pipeline.run()

Config hierarchy
----------------

All pipeline configs inherit from :class:`~biolm.pipeline.generative.ScoringProtocolConfig`.
Use ``isinstance`` to dispatch on config type:

.. code-block:: python

   from biolm.pipeline.generative import (
       ScoringProtocolConfig,
       GenerativeProtocolConfig,
       SaturationMutagenesisConfig,
       IterativeMaskingDMSConfig,
       DirectGenerationConfig,
   )

   if isinstance(config, SaturationMutagenesisConfig):
       pipeline = GenerativePipeline(configs=[config])
   elif isinstance(config, DirectGenerationConfig):
       pipeline = GenerativePipeline(configs=[config])

.. code-block:: text

   ScoringProtocolConfig
   ├── SaturationMutagenesisConfig
   ├── IterativeMaskingDMSConfig
   └── DirectGenerationConfig (extends GenerativeProtocolConfig)

Model quick-reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Config type
     - Model
     - Action
     - Notes
   * - SaturationMutagenesisConfig
     - esm2-650m
     - predict
     - Single-mutant library + scoring
   * - IterativeMaskingDMSConfig
     - esm2-650m
     - predict
     - Greedy MLM argmax DMS
   * - DirectGenerationConfig
     - proteinmpnn / dsm / antifold
     - generate
     - Structure-conditioned generation

Config classes
--------------

Field details are documented on each class below.

.. autoclass:: biolm.pipeline.generative.ScoringProtocolConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: biolm.pipeline.generative.GenerativeProtocolConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: biolm.pipeline.generative.SaturationMutagenesisConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: biolm.pipeline.generative.IterativeMaskingDMSConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: biolm.pipeline.generative.DirectGenerationConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. _sdk-pipeline-examples:

Pipeline examples
-----------------

Saturation mutagenesis funnel:

.. code-block:: python

   from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

   config = SaturationMutagenesisConfig(
       parent_sequence="MKTAYIAKQRQ",
       scoring_model="esm2-650m",
       scoring_action="predict",
       score_field="logits",
       top_n=10,
   )
   pipeline = GenerativePipeline(configs=[config])
   results = pipeline.run()

Direct generation (ProteinMPNN):

.. code-block:: python

   from biolm.pipeline import GenerativePipeline, DirectGenerationConfig

   config = DirectGenerationConfig(
       model_name="protein-mpnn",
       structure_path="design.pdb",
       num_sequences=10,
   )
   pipeline = GenerativePipeline(configs=[config])
   results = pipeline.run()

Serialization and resumability
------------------------------

Pipeline definitions can be serialized with :class:`~biolm.pipeline.pipeline_def.PipelineDef`
for round-trip storage. DuckDB caches predictions and embeddings so re-runs skip
completed work.

.. _sdk-pipeline-see-also:

Related
-------

- :doc:`../intro/sdk-overview` — SDK overview
- :doc:`../api-reference/biolm.pipeline` — full pipeline module reference
