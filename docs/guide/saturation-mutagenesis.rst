.. _saturation-mutagenesis:

======================
Saturation mutagenesis
======================

*Orchestrating molecular design workflows.*

Saturation mutagenesis is the workhorse of protein engineering: take a parent
sequence, enumerate **every** single amino-acid substitution at the positions
you care about, score each variant with a model, then keep the best ones. The
SDK packages this as a single declarative config, ``SaturationMutagenesisConfig``,
that you drop into a :class:`~biolm.pipeline.generative.GenerativePipeline`. The
pipeline builds the single-mutant library internally, scores it in batches, sorts
by your chosen field, and returns the top ``top_n`` variants — with DuckDB caching
and resume for free.

What it does
============

Given a ``parent_sequence`` of length *L*, the config generates up to
``19 × L`` variants (every non-wild-type residue at every position), or just the
positions you list. Each variant is sent to ``scoring_model``, the numeric value
at ``score_field`` is extracted, and the library is ranked. Only the top
``top_n`` rows survive into the results. No separate generation step is needed —
the candidate library *is* the enumeration.

When to use it
==============

Reach for saturation mutagenesis whenever you want an exhaustive, unbiased sweep
of point mutations rather than sampled novelty. The canonical case is **stability
engineering**: find substitutions that lower the folding free energy (ΔΔG).

- **ThermoMPNN-D** (``thermompnn-d``) — a structure-aware ΔΔG predictor. It needs
  a PDB structure and reports a ``ddg`` field per mutation. This is the primary
  example below and matches the class defaults (``score_field="ddg"``,
  ``ascending=True``).
- **ESM2StabP** (``esm2stabp``) — a sequence-only stability predictor. Use it as a
  drop-in alternative when you have no structure: leave ``pdb_str=None`` and set
  ``score_field`` to whatever key that model returns.

The same pattern extends to any per-variant scalar — activity proxies, solubility,
expression — as long as the model exposes it in its response.

A worked example
=================

The example below scans three positions of a parent sequence with ThermoMPNN-D,
keeps the 25 most stabilizing variants, and (optionally) folds each survivor with
ESMFold to attach a confidence score.

.. code-block:: python

   from biolm.pipeline import GenerativePipeline, SaturationMutagenesisConfig

   config = SaturationMutagenesisConfig(
       parent_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
       scoring_model="thermompnn-d",
       positions=[3, 7, 10],          # 0-indexed; None -> scan all positions
       score_field="ddg",            # key in the API response holding ΔΔG
       top_n=25,                      # keep the 25 best variants
       ascending=True,                # lower ΔΔG = more stabilizing
       pdb_str=open("protein.pdb").read(),  # structure-aware models require this
       batch_size=8,
   )

   pipeline = GenerativePipeline(configs=[config])
   pipeline.run()

   df = pipeline.results()
   print(df[["sequence", "sat_position", "sat_wt_aa", "sat_mut_aa", "ddg"]])

Key configuration fields
========================

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Field
     - Default
     - Meaning
   * - ``parent_sequence``
     - —
     - Wild-type sequence to mutate (required).
   * - ``scoring_model``
     - —
     - BioLM model slug used to score variants, e.g. ``'thermompnn-d'``.
   * - ``positions``
     - ``None``
     - 0-indexed positions to enumerate. ``None`` scans every position.
   * - ``score_field``
     - ``'ddg'``
     - Key inside each model response holding the numeric score. Dotted paths
       (``'a.b'``) are supported for nested responses.
   * - ``top_n``
     - ``50``
     - Number of top-ranked variants to keep. ``None`` keeps all.
   * - ``ascending``
     - ``True``
     - If ``True``, lower scores rank higher (correct for ΔΔG).
   * - ``exclude_synonymous``
     - ``True``
     - Skip substitutions equal to the wild-type residue.
   * - ``pdb_str``
     - ``None``
     - Raw PDB file contents (a string, not a path). Required by structure-aware
       models like ThermoMPNN-D; leave ``None`` for ESM2StabP.

``scoring_action``: predict vs. score
======================================

``scoring_action`` selects which client method runs the variants. The default,
``'predict'``, calls the model's ``predict`` endpoint — correct for ThermoMPNN-D
and ESM2StabP. Set ``scoring_action="score"`` only when your model exposes a
dedicated ``score`` action. Both values are allowlisted; anything else raises a
``ValueError`` at construction.

Optional downstream prediction
==============================

Because a ``GenerativePipeline`` is a full pipeline, you can chain more stages
onto the ranked survivors. A common follow-up is folding the top variants to
gauge structural confidence:

.. code-block:: python

   pipeline = GenerativePipeline(configs=[config])
   pipeline.add_prediction("esmfold", extractions="mean_plddt", columns="plddt")
   pipeline.run()
   df = pipeline.results()

This adds a ``plddt`` column to the results, computed only for the variants that
passed ranking — so you fold 25 sequences, not the whole library.

Reading the results
====================

``pipeline.results()`` (an alias for :meth:`~biolm.pipeline.BasePipeline.get_final_data`)
returns a pandas DataFrame — call it **after** ``pipeline.run()``, which returns
per-stage summaries, not rows. Alongside ``sequence`` and the score column
(named after ``score_field``, e.g. ``ddg``), the frame carries provenance columns:

- ``sat_position`` — the 0-indexed position that was mutated.
- ``sat_wt_aa`` — the wild-type residue at that position.
- ``sat_mut_aa`` — the substituted residue.
- ``source_label`` — the config's ``label`` (``None`` if unset); always present.

Interpretation
==============

For stability work with ThermoMPNN-D, the defaults already point you the right
way: ``ascending=True`` with ``score_field="ddg"`` surfaces the **most
stabilizing** mutants first, because a more negative ΔΔG means a more stable fold.
Flip ``ascending=False`` when a *higher* score is better (e.g. an activity proxy).

One caveat: ``top_n`` is a ceiling, not a guarantee. If the model fails to return
a value for some variants — a missing ``score_field``, an errored batch — those
rows are dropped before ranking, so you may see fewer than ``top_n`` results. Widen
``positions`` or inspect the per-item errors if the survivor count looks low.

Where to go next
================

- :doc:`workflows-overview` — choose between Model, Protocol, and Pipeline tiers.
- :doc:`pipeline-workflows` — compose multi-stage generate/score/filter loops.
- :doc:`../sdk/pipeline` — the full ``GenerativePipeline`` API and end-to-end
  design walkthroughs.
