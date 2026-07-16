.. _iterative-masking-dms:

======================
Iterative masking DMS
======================

*Orchestrating molecular design workflows.*

Iterative masking deep mutational scanning (DMS) turns a masked language model
into a *proposer* of substitutions. Instead of enumerating every point mutation
and scoring it externally, you mask a position, ask the model which residue it
would most like to see there, and take that argmax as the preferred variant. The
SDK packages this as ``IterativeMaskingDMSConfig``, a source config you drop into
a :class:`~biolm.pipeline.generative.GenerativePipeline`. The pipeline runs the
masking rounds, decodes the logits, deduplicates the variants, and hands you a
tidy DataFrame — with DuckDB caching and resume for free.

What it does
============

Given a ``parent_sequence`` and a list of ``positions``, the config walks up to
two greedy rounds:

1. **Round 1 (single-point).** Each target position is masked in the parent and
   sent to ``model_name``. The model returns per-token logits; the SDK takes the
   argmax over the amino-acid alphabet and records that residue as the position's
   preferred substitution. With ``exclude_synonymous=True`` (the default), any
   argmax that equals the wild-type residue is dropped.
2. **Round 2 (two-point).** Each round-1 variant becomes the new starting
   sequence. The SDK masks every *other* target position, queries the model
   again, and combines the round-1 and round-2 argmax residues into a two-point
   mutant. Duplicate sequences are collapsed.

Only ``rounds=1`` and ``rounds=2`` are implemented; ``rounds > 2`` raises a
``ValueError`` at construction, as does ``rounds < 1``. Because the procedure
needs per-token logit arrays, ``action`` must be ``'predict'`` — the sole
allowlisted value.

How it differs from saturation mutagenesis
===========================================

Both configs produce mutant libraries, but the search strategy is opposite. In
:doc:`saturation-mutagenesis`, you exhaustively enumerate every substitution and
lean on an **external scorer** (ThermoMPNN-D, ESM2StabP) to rank them. Here, the
**model itself picks** the substitution at each position via a logits argmax, so
there is no separate scoring model and no ranking step — the candidate set is
whatever the language model prefers. That makes iterative masking far cheaper
(one masked query per position, not 19) and biased toward residues the model
considers natural, rather than an unbiased sweep. Reach for it when you want the
model's opinion on plausible substitutions; reach for saturation mutagenesis when
you want a complete, scorer-ranked map of a fitness landscape.

A worked example
================

The example below runs a two-round greedy DMS at three positions of a parent
sequence using ESM2-650M, then reads the results.

.. code-block:: python

   from biolm.pipeline import GenerativePipeline, IterativeMaskingDMSConfig

   config = IterativeMaskingDMSConfig(
       parent_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
       model_name="esm2-650m",
       positions=[2, 5, 8],       # 0-indexed; None -> probe every position
       rounds=2,                  # round 1: single-point, round 2: two-point
       exclude_synonymous=True,   # skip argmax residues equal to wild type
       batch_size=32,
   )

   pipeline = GenerativePipeline(configs=[config])
   pipeline.run()

   df = pipeline.results()
   print(df[["sequence", "dms_round", "dms_pos1", "dms_aa1", "dms_pos2", "dms_aa2"]])

Key configuration fields
========================

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Field
     - Default
     - Meaning
   * - ``parent_sequence``
     - —
     - Wild-type sequence to mutate (required; cannot be empty).
   * - ``model_name``
     - —
     - MLM model slug returning logits, e.g. ``'esm2-650m'`` (required).
   * - ``positions``
     - ``None``
     - 0-indexed positions to probe. ``None`` probes every position. Out-of-range
       indices raise a ``ValueError``.
   * - ``rounds``
     - ``2``
     - Number of masking rounds. Only ``1`` (single-point) and ``2`` (two-point)
       are supported.
   * - ``mask_token``
     - ``'<mask>'``
     - Token inserted at masked positions before each query.
   * - ``alphabet``
     - ``'ACDEFGHIKLMNPQRSTVWY'``
     - Amino-acid vocabulary used to identify valid residues in the logits.
   * - ``exclude_synonymous``
     - ``True``
     - Skip round-1 variants whose argmax equals the wild-type residue.
   * - ``batch_size``
     - ``32``
     - Sequences per API request.
   * - ``label``
     - ``None``
     - Optional label stored as ``source_label`` in the results.
   * - ``action``
     - ``'predict'``
     - API action. Only ``'predict'`` is allowed; the argmax procedure requires
       per-token logit arrays.

Reading the results
===================

Call ``pipeline.run()`` first — it returns per-stage summaries, **not** rows —
then ``pipeline.results()`` (an alias for
:meth:`~biolm.pipeline.BasePipeline.get_final_data`) to get the pandas DataFrame.
Alongside ``sequence``, each row carries provenance columns describing how the
variant was built:

- ``dms_round`` — ``1`` for single-point variants, ``2`` for two-point variants.
- ``dms_pos1`` / ``dms_aa1`` — the first mutated position and its argmax residue.
- ``dms_pos2`` / ``dms_aa2`` — the second position and residue (populated only for
  round-2 rows).
- ``source_label`` — the config's ``label`` (``None`` if unset); always present.

Interpretation
==============

The variant counts are worth anticipating. Round 1 yields at most one row per
position — fewer when ``exclude_synonymous`` drops synonymous argmax calls or a
model returns no usable logits. Round 2 expands each round-1 variant against the
other positions, then deduplicates, so the count is bounded by the number of
distinct two-point combinations rather than a fixed ``top_n``. If a batch errors
or the model omits ``vocab_tokens`` for a logits response, the affected rows are
skipped, so a low survivor count usually points to failed batches or a model that
does not expose a decodable vocabulary.

Because a ``GenerativePipeline`` is a full pipeline, you can chain downstream
prediction stages onto the generated variants — for example, folding the survivors
or scoring them with a stability model — exactly as in the saturation-mutagenesis
workflow.

Where to go next
================

- :doc:`saturation-mutagenesis` — the exhaustive, scorer-ranked counterpart.
- :doc:`pipeline-workflows` — compose multi-stage generate/score/filter loops.
- :doc:`../sdk/pipeline` — the full ``GenerativePipeline`` API and end-to-end
  design walkthroughs.
