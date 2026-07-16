.. _structure-conditioned-generation:

================================
Structure-conditioned generation
================================

*Orchestrating molecular design workflows.*

Inverse-folding and diffusion models turn a *scaffold* into new sequences: give
ProteinMPNN a backbone and it proposes amino acids that fold into it; give DSM a
parent sequence and it resamples masked regions. The SDK exposes both through a
single source config, ``DirectGenerationConfig``, which you drop into a
:class:`~biolm.pipeline.generative.GenerativePipeline`. The pipeline fires every
config in parallel, deduplicates the output, and hands you a tidy DataFrame — with
DuckDB caching and resume for free.

Two conditioning modes
=======================

``DirectGenerationConfig`` covers two families of generative model, distinguished
only by what you condition on and which ``item_field`` the API expects:

- **Structure-conditioned** (inverse folding). Models such as ``protein-mpnn``,
  ``hyper-mpnn``, ``soluble-mpnn``, ``ligand-mpnn``, and ``antifold`` take a PDB
  or CIF structure and emit sequences that should fold into it. These use
  ``item_field="pdb"`` and read the backbone from ``structure_path``.
- **Sequence-conditioned** (masked diffusion). ``dsm-150m-base`` and
  ``dsm-650m-base`` take a parent ``sequence`` and resample it. These use
  ``item_field="sequence"``.

The config does not guess model-specific parameters for you. The caller supplies
the correct ``item_field`` and ``params`` for the target model, which vary per
model and are documented in the BioLM API schema
(``/schema/<model>/generate/``). When ``params`` is left empty, the stage falls
back to sending ``{'num_sequences': num_sequences, 'temperature': temperature}``
— convenient only for models that accept those exact names. For anything
non-trivial, pass ``params`` explicitly.

A minimal working example
=========================

The example below designs sequences for a backbone with ProteinMPNN and, in the
same pipeline, resamples a parent sequence with DSM. Both configs run
concurrently; their outputs merge into one funnel.

.. code-block:: python

   from biolm.pipeline import DirectGenerationConfig, GenerativePipeline

   # Structure-conditioned: ProteinMPNN reads a PDB backbone
   mpnn = DirectGenerationConfig(
       model_name="protein-mpnn",
       structure_path="protein.pdb",
       item_field="pdb",                       # default; shown for clarity
       params={"batch_size": 50, "temperature": 0.1},
       label="mpnn_T0.1",                      # -> source_label in results
   )

   # Sequence-conditioned: DSM resamples a parent sequence
   dsm = DirectGenerationConfig(
       model_name="dsm-150m-base",
       sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
       item_field="sequence",
       params={"num_sequences": 100, "temperature": 1.0,
               "remasking": "low_confidence", "step_divisor": 8},
       label="dsm_baseline",
   )

   pipeline = GenerativePipeline(configs=[mpnn, dsm])
   pipeline.run()

   df = pipeline.results()
   print(df[["sequence", "source_label"]])

Note the ``params`` keys differ by model: MPNN variants count samples with
``batch_size``, while DSM uses ``num_sequences`` plus its own ``remasking`` and
``step_divisor`` controls. AntiFold expects ``heavy_chain`` and ``light_chain``
identifiers alongside ``num_seq_per_target`` and ``sampling_temp``. Always check
the model's schema before setting ``params``.

Key configuration fields
========================

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Field
     - Default
     - Meaning
   * - ``model_name``
     - —
     - BioLM model slug, e.g. ``'protein-mpnn'``, ``'antifold'``,
       ``'dsm-150m-base'`` (required).
   * - ``structure_path``
     - ``None``
     - Path to a PDB/CIF file for structure-conditioned models.
   * - ``structure_column``
     - ``None``
     - DataFrame column holding PDB strings, when a structure arrived from an
       upstream stage.
   * - ``sequence``
     - ``None``
     - Parent sequence for sequence-conditioned models (DSM).
   * - ``item_field``
     - ``'pdb'``
     - Item dict key the API expects — ``'pdb'`` for inverse folding,
       ``'sequence'`` for DSM.
   * - ``params``
     - ``{}``
     - Model-specific params dict. Required for non-trivial calls; keys must match
       the model's API names exactly.
   * - ``num_sequences``
     - ``100``
     - Fallback sample count used only when ``params`` is empty.
   * - ``temperature``
     - ``1.0``
     - Fallback temperature used only when ``params`` is empty.
   * - ``structure_from_stage`` / ``structure_from_model``
     - ``None``
     - Read backbones from the DuckDB ``structures`` table populated by an
       upstream prediction stage (see below).
   * - ``n_runs``
     - ``1``
     - Repeat the same call in parallel; total output is samples-per-call ×
       ``n_runs``, then deduplicated.
   * - ``label``
     - ``None``
     - Human-readable tag surfaced as ``source_label`` in the results.

Chaining from predicted structures
===================================

You do not need a PDB on disk. If an upstream stage folds sequences (for example
ESMFold), point the generation config at that stage's output with
``structure_from_stage`` and scope it to the folding model with
``structure_from_model``. The stage then reads backbones straight from the
DuckDB ``structures`` table for the sequences in the current run:

.. code-block:: python

   DirectGenerationConfig(
       model_name="protein-mpnn",
       item_field="pdb",
       structure_from_stage="fold",         # name of the upstream fold stage
       structure_from_model="esmfold",      # select the right structures
       params={"batch_size": 20, "temperature": 0.2},
   )

Setting ``structure_from_stage`` without ``structure_from_model`` warns and uses
structures from *every* upstream model — pass both to avoid mixing backbones when
more than one folding stage has run.

Fan out across models and temperatures
======================================

Because every config in a ``GenerativePipeline`` runs in parallel, you can scan a
grid of models and temperatures in one shot, tagging each with a ``label`` so the
``source_label`` column tells you which combination produced each survivor. The
reference script ``scripts/pipeline_mpnn_multi.py`` builds exactly this: five MPNN
variants × three temperatures = fifteen ``DirectGenerationConfig`` objects fired
together, deduplicated, then run through downstream ``temberture-regression`` (Tm)
and ``biolmsol`` (solubility) prediction stages, a threshold filter, and a
top-``N`` ranking. It is the canonical template for a full design funnel:

.. code-block:: python

   configs = [
       DirectGenerationConfig(
           model_name=model,
           structure_path="protein.pdb",
           item_field="pdb",
           params={"batch_size": 100, "temperature": temp},
           label=f"{model}_T{temp}",
       )
       for model in ("protein-mpnn", "hyper-mpnn", "soluble-mpnn")
       for temp in (0.1, 0.3, 0.5)
   ]
   pipeline = GenerativePipeline(configs=configs, deduplicate=True)

Reading the results
===================

Call ``pipeline.run()`` first — it returns a ``dict`` of per-stage
``StageResult`` summaries, **not** rows — then ``pipeline.results()`` (an alias
for :meth:`~biolm.pipeline.BasePipeline.get_final_data`) to materialize the pandas
DataFrame. Every row carries ``sequence_id``, ``sequence``, ``length``, ``hash``,
and ``source_label`` (the config's ``label``, or ``None`` when unset — always
present, so you never need to guard on the column), plus one column per prediction
type added by downstream stages.

Where to go next
================

- :doc:`pipeline-workflows` — compose multi-stage generate / predict / filter loops.
- :doc:`saturation-mutagenesis` — exhaustive single-mutant libraries, scorer-ranked.
- :doc:`iterative-masking-dms` — let a masked language model propose substitutions.
- :doc:`../sdk/pipeline` — the full ``GenerativePipeline`` API and end-to-end
  design walkthroughs.
