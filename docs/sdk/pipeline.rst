.. _sdk-pipeline:

biolm.pipeline
==============

The ``biolm.pipeline`` module provides a config-driven framework for
multi-stage protein design workflows.  Generation, scoring, prediction, and
filtering stages are declared as typed config objects, linked into a
:class:`~biolm.pipeline.generative.GenerativePipeline`, and executed
asynchronously with DuckDB-backed caching and resumability.

.. contents:: On this page
   :local:
   :depth: 2


Config Class Hierarchy
----------------------

All generation and scoring configs inherit from one of two marker base classes:

.. code-block:: text

    ScoringProtocolConfig
    └── SaturationMutagenesisConfig   (single-mutant library + scoring)

    GenerativeProtocolConfig
    ├── DirectGenerationConfig         (ProteinMPNN / DSM / AntiFold)
    └── IterativeMaskingDMSConfig      (greedy MLM argmax DMS)

Use :func:`isinstance` to dispatch on config type in pipeline runners:

.. code-block:: python

    from biolm.pipeline.generative import (
        ScoringProtocolConfig,
        GenerativeProtocolConfig,
    )

    if isinstance(config, ScoringProtocolConfig):
        ...   # stage scores and ranks a fixed variant library
    elif isinstance(config, GenerativeProtocolConfig):
        ...   # stage emits new sequences


Base Classes
------------

.. autoclass:: biolm.pipeline.generative.ScoringProtocolConfig
   :members:
   :undoc-members:

.. autoclass:: biolm.pipeline.generative.GenerativeProtocolConfig
   :members:
   :undoc-members:


SaturationMutagenesisConfig
----------------------------

Enumerates every single amino-acid substitution at the specified positions,
scores each variant with a BioLM prediction model (e.g. ThermoMPNN-D,
ESM2StabP), and returns the top-*N* variants ranked by the chosen score field.

.. autoclass:: biolm.pipeline.generative.SaturationMutagenesisConfig
   :members:
   :undoc-members:
   :show-inheritance:

**Fields**

.. list-table::
   :header-rows: 1
   :widths: 20 10 60

   * - Field
     - Default
     - Description
   * - ``parent_sequence``
     - *(required)*
     - Wild-type amino-acid sequence to mutate.
   * - ``scoring_model``
     - *(required)*
     - BioLM model slug used to score variants, e.g. ``'thermompnn-d'``.
   * - ``positions``
     - ``None``
     - 0-indexed positions to enumerate.  ``None`` enumerates all positions.
   * - ``alphabet``
     - ``'ACDEFGHIKLMNPQRSTVWY'``
     - Amino acids to substitute at each position.
   * - ``scoring_action``
     - ``'predict'``
     - API method to call: ``'predict'`` or ``'score'``.
   * - ``scoring_params``
     - ``{}``
     - Extra params forwarded to the scoring model API.
   * - ``score_field``
     - ``'ddg'``
     - Key in the API response holding the numeric score.  Supports dotted
       access, e.g. ``'result.ddg'`` (max 3 components).
   * - ``top_n``
     - ``50``
     - Number of variants to retain after ranking.  ``None`` keeps all.
   * - ``ascending``
     - ``True``
     - ``True`` = lower score is better (e.g. negative ΔΔG = stabilising).
   * - ``exclude_synonymous``
     - ``True``
     - Skip substitutions identical to the wild-type residue.
   * - ``batch_size``
     - ``8``
     - Sequences per API request.
   * - ``pdb_str``
     - ``None``
     - Raw PDB file contents as a string.  Required for structure-aware models
       such as ThermoMPNN-D; leave ``None`` for sequence-only models.
   * - ``chain``
     - ``'A'``
     - Chain identifier used in structure-aware scoring items.
   * - ``label``
     - ``None``
     - Human-readable label stored as ``source_label`` in results.

**Example**

.. code-block:: python

    from biolm.pipeline import SaturationMutagenesisConfig, GenerativePipeline

    config = SaturationMutagenesisConfig(
        parent_sequence="MKTAYIAKQRQ",
        scoring_model="thermompnn-d",
        positions=[3, 7, 10],          # only probe these three residues
        score_field="ddg",
        top_n=25,
        ascending=True,
        pdb_str=open("protein.pdb").read(),
    )

    pipeline = GenerativePipeline(configs=[config])
    df = pipeline.run().get_final_data()
    # df contains columns: sequence, sat_position, sat_wt_aa, sat_mut_aa, ddg


IterativeMaskingDMSConfig
--------------------------

Implements a greedy-argmax masking procedure using a masked language model
(ESM2, ESMC) to build multi-point variant sequences without sampling:

1. **Round 1** — For each target position, mask it in the parent sequence and
   query the model for the highest-probability residue (argmax, not sampled).
2. **Round 2** — Apply the round-1 substitution, then mask each *other* target
   position and collect the round-2 argmax.  Output is all unique 2-point
   combination variants (deduped).

This matches the ESM2 two-round DMS design pattern used in EGF generation.

.. autoclass:: biolm.pipeline.generative.IterativeMaskingDMSConfig
   :members:
   :undoc-members:
   :show-inheritance:

**Fields**

.. list-table::
   :header-rows: 1
   :widths: 20 10 60

   * - Field
     - Default
     - Description
   * - ``parent_sequence``
     - *(required)*
     - Starting amino-acid sequence.
   * - ``model_name``
     - *(required)*
     - MLM model slug, e.g. ``'esm2-650m'``, ``'esmc-300m'``.
   * - ``positions``
     - ``None``
     - 0-indexed positions to probe.  ``None`` probes all positions.
   * - ``rounds``
     - ``2``
     - Number of sequential masking rounds.  ``1`` yields single-point
       variants; ``2`` (default) yields 2-point combination variants.
   * - ``mask_token``
     - ``'<mask>'``
     - Token inserted at masked positions in the query sequence.
   * - ``alphabet``
     - ``'ACDEFGHIKLMNPQRSTVWY'``
     - Canonical amino-acid vocabulary used to extract argmax from logits.
   * - ``exclude_synonymous``
     - ``True``
     - Skip round-1 positions where the argmax matches the wild-type residue.
   * - ``batch_size``
     - ``32``
     - Sequences per API request.
   * - ``action``
     - ``'predict'``
     - API action for the model.  Must be ``'predict'`` (logits required);
       any other value raises ``ValueError``.
   * - ``label``
     - ``None``
     - Human-readable label stored as ``source_label`` in results.

**Example**

.. code-block:: python

    from biolm.pipeline import IterativeMaskingDMSConfig, GenerativePipeline

    config = IterativeMaskingDMSConfig(
        parent_sequence="MKTAYIAKQRQ",
        model_name="esm2-650m",
        positions=[2, 5, 8],   # 0-indexed
        rounds=2,
        exclude_synonymous=True,
    )

    pipeline = GenerativePipeline(configs=[config])
    df = pipeline.run().get_final_data()
    # df contains columns:
    #   sequence, dms_round, dms_pos1, dms_aa1, dms_pos2, dms_aa2


DirectGenerationConfig
-----------------------

Calls inherently generative models — ProteinMPNN, HyperMPNN, LigandMPNN,
SolubleMPNN, AntiFold, DSM — to produce new sequences.  The caller is
responsible for providing the correct ``item_field`` and ``params`` for the
target model; no auto-detection is performed.

.. autoclass:: biolm.pipeline.generative.DirectGenerationConfig
   :members:
   :undoc-members:
   :show-inheritance:

**Fields**

.. list-table::
   :header-rows: 1
   :widths: 25 15 55

   * - Field
     - Default
     - Description
   * - ``model_name``
     - *(required)*
     - BioLM model slug, e.g. ``'protein-mpnn'``, ``'dsm-150m-base'``.
   * - ``structure_path``
     - ``None``
     - Path to a PDB or CIF file on disk.
   * - ``structure_column``
     - ``None``
     - DataFrame column holding PDB strings from an upstream pipeline stage.
   * - ``sequence``
     - ``None``
     - Parent sequence for sequence-conditioned models (DSM).
   * - ``item_field``
     - ``'pdb'``
     - The item dict key sent to the model API.  Use ``'pdb'`` for
       structure-conditioned models; ``'sequence'`` for DSM.
   * - ``params``
     - ``{}``
     - Model-specific params dict.  Keys must exactly match the model's API
       param names.  When empty, falls back to the ``num_sequences`` /
       ``temperature`` convenience fields.
   * - ``num_sequences``
     - ``100``
     - Fallback number of sequences when ``params`` is empty.
   * - ``temperature``
     - ``1.0``
     - Fallback sampling temperature when ``params`` is empty.
   * - ``structure_from_stage``
     - ``None``
     - Read structures from an upstream stage's DuckDB structures table.
   * - ``structure_from_model``
     - ``None``
     - Filter the upstream structures by model name (e.g. ``'esmfold'``).
   * - ``n_runs``
     - ``1``
     - Run generation ``n_runs`` times in parallel; outputs are pooled and
       deduplicated.  Useful for large stochastic models.
   * - ``label``
     - ``None``
     - Human-readable label stored as ``source_label`` in results.

**Model quick-reference**

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Model slug
     - ``item_field``
     - Key ``params``
   * - ``protein-mpnn``, ``hyper-mpnn``, ``ligand-mpnn``, ``soluble-mpnn``
     - ``'pdb'``
     - ``num_sequences``, ``temperature``
   * - ``antifold``
     - ``'pdb'``
     - ``heavy_chain``, ``light_chain``, ``num_seq_per_target``, ``sampling_temp``
   * - ``dsm-150m-base``, ``dsm-650m-base``
     - ``'sequence'``
     - ``num_sequences``, ``temperature``, ``remasking``
       (``'random'`` / ``'low_confidence'`` / ``'high_confidence'``),
       ``step_divisor``

**Example**

.. code-block:: python

    from biolm.pipeline import DirectGenerationConfig, GenerativePipeline

    # Structure-conditioned design with ProteinMPNN
    cfg_mpnn = DirectGenerationConfig(
        model_name="protein-mpnn",
        structure_path="protein.pdb",
        item_field="pdb",
        params={"num_sequences": 100, "temperature": 0.1},
    )

    # Sequence-conditioned design with DSM
    cfg_dsm = DirectGenerationConfig(
        model_name="dsm-150m-base",
        sequence="MKTAYIAKQRQ",
        item_field="sequence",
        params={"num_sequences": 100, "temperature": 1.0,
                "remasking": "low_confidence"},
    )

    # Mix both in one pipeline — configs run in parallel
    pipeline = GenerativePipeline(configs=[cfg_mpnn, cfg_dsm])
    pipeline.add_prediction("esmfold", extractions="mean_plddt", columns="plddt")
    df = pipeline.run().get_final_data()


Full Pipeline Example
---------------------

The following example shows a complete design funnel: generate with
ProteinMPNN, score stability with ThermoMPNN-D, filter by pLDDT, and rank
the final candidates.

.. code-block:: python

    from biolm.pipeline import (
        GenerativePipeline,
        DirectGenerationConfig,
        SaturationMutagenesisConfig,
    )
    from biolm.pipeline.filters import ThresholdFilter, RankingFilter

    pdb_str = open("protein.pdb").read()

    pipeline = GenerativePipeline(
        configs=[
            DirectGenerationConfig(
                model_name="protein-mpnn",
                structure_path="protein.pdb",
                item_field="pdb",
                params={"num_sequences": 500, "temperature": 0.2},
            ),
        ]
    )

    # Predict structure quality on all 500 designs
    pipeline.add_prediction("esmfold", extractions="mean_plddt", columns="plddt")

    # Keep only high-confidence designs
    pipeline.add_filter(ThresholdFilter("plddt", min_value=75.0))

    # Score stability of the survivors with ThermoMPNN-D
    pipeline.add_prediction("thermompnn-d", extractions="ddg", columns="ddg")

    # Keep top-50 most stabilising
    pipeline.add_filter(RankingFilter("ddg", n=50, ascending=True))

    results = pipeline.run()
    df = pipeline.get_final_data()
    print(df[["sequence", "plddt", "ddg"]].sort_values("ddg").head(10))


Serialization and Recovery
--------------------------

Every config and stage serializes to a JSON-compatible dict via ``to_spec()``,
stored in the pipeline's DuckDB database.  After a kernel crash or network
interruption the pipeline can be reconstructed and resumed:

.. code-block:: python

    # Initial run
    pipeline = GenerativePipeline(configs=[...], datastore="design.duckdb")
    pipeline.run()

    # Reconstruct from DB and resume (only re-runs incomplete stages)
    pipeline = GenerativePipeline.from_db("design.duckdb", resume=True)
    pipeline.run(resume=True)

The ``to_spec()`` / ``from_db()`` roundtrip is handled by
``biolm.pipeline.pipeline_def.pipeline_from_definition()``.  Large strings
(e.g. PDB file contents) are stored in a separate ``pipeline_blobs`` table and
resolved transparently on load.


Quickstart examples
-------------------

Install the pipeline extra: ``pip install "biolm-sdk[pipeline]"``.

**Saturation mutagenesis** — single-mutant library + scoring:

.. code-block:: python

    import asyncio
    from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

    config = SaturationMutagenesisConfig(
        scoring_model="esm2-650m",
        scoring_action="score",
        score_field="score",
        positions=[10, 15, 42],
        batch_size=32,
    )
    pipeline = GenerativePipeline(configs=[config])
    df = asyncio.run(pipeline.run_async(sequences=["MKTAYIAKQRQISFVKSHFSRQ"]))

**Direct generation** (ProteinMPNN / DSM / AntiFold):

.. code-block:: python

    from biolm.pipeline import GenerativePipeline, DirectGenerationConfig

    config = DirectGenerationConfig(
        model="proteinmpnn-v48-002",
        action="generate",
        n_runs=10,
        params={"temperature": 0.1},
    )
    pipeline = GenerativePipeline(configs=[config])
    df = asyncio.run(pipeline.run_async(pdb_strings=[pdb_str]))


See Also
--------

- :doc:`overview` — SDK overview and quick start
- :doc:`../api-reference/modules` — Full autodoc API reference
