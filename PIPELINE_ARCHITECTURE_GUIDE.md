# Pipeline Architecture Guide

Technical reference for `biolm.pipeline` internals.

---

## Config Class Hierarchy

```
object
├── ScoringProtocolConfig          # marker: stage scores/ranks existing sequences
│   └── SaturationMutagenesisConfig
│       - generates all single-mutant variants at specified positions
│       - scores each with a BioLM scoring model (action="score" or "predict")
│       - _ALLOWED_SCORING_ACTIONS = {"predict", "score"}
│
└── GenerativeProtocolConfig       # marker: stage produces NEW sequences
    ├── DirectGenerationConfig
    │   - calls inherently generative models (ProteinMPNN, DSM, AntiFold)
    │   - action="generate"
    │
    └── IterativeMaskingDMSConfig
        - greedy MLM argmax 2-point DMS
        - calls MLM model with action="predict" to get per-position logits
        - _ALLOWED_MLM_ACTIONS = {"predict"}
```

### Why two base classes?

`ScoringProtocolConfig` stages operate on an existing sequence and return fitness scores — they don't produce new sequences, they rank existing ones.  
`GenerativeProtocolConfig` stages produce genuinely new sequences (either from a structure, or by substituting residues via logit-guided search).

The split enforces valid `action` values at construction time and lets downstream pipeline code (`GenerationStage.process()`, test assertions, type narrowing) dispatch correctly.

---

## Data Alignment Guarantees

### SaturationMutagenesisConfig

`_run_saturation_mutagenesis` builds a flat `mutants` list in deterministic order:

```
outer loop: positions
  inner loop: alphabet
    → (seq, pos, wt_aa, mut_aa)
```

A flat `scores` list is accumulated across batches.  Per-batch invariants:
- **Under-return**: `batch_scores` is padded to `len(batch)` with `None` and a warning is logged.  The `None`-padded variant is excluded from the output (filtered at row construction time).
- **Over-return**: `batch_scores` is **capped** to `len(batch)` and a warning is logged.  Without this cap, extra scores shift every subsequent batch's alignment.

The final `zip(seqs, positions_list, wt_aas, mut_aas, scores)` is always length-safe because `scores` is guaranteed `== len(mutants)` after batching completes.

### IterativeMaskingDMSConfig

`_run_iterative_masking_dms` runs two rounds:

- **Round 1**: mask each position independently; collect top-K AAs per position from logits.  Per-batch: `try/except` + padding to `len(batch)` with `{}` on any API failure.
- **Round 2**: cross-product of top-K positions into 2-point variants; per-batch same try/except + padding.

Both rounds extend their respective result lists only after ensuring `len(raw) == len(batch)`.

---

## DuckDB Write Path

After `asyncio.gather()` returns all config results, `GenerationStage.process()` writes to DuckDB in a single-threaded sequential path:

1. `add_sequences_batch(df_generated)` → returns `seq_ids` (order-preserving hash lookup)
2. `add_predictions_batch(score_rows)` → only for columns in `_score_cols`
3. `add_generation_metadata_batch(meta_rows)` → includes `source_label`, `model_name`, etc.

Provenance columns specific to DMS configs (`sat_position`, `sat_wt_aa`, `sat_mut_aa`, `dms_round`, `dms_pos1`, `dms_aa1`, `dms_pos2`, `dms_aa2`, and the dynamic `score_field` column) are present in the in-memory `df_generated` returned by `process_ws()` but are **not** currently written to DuckDB — `_score_cols` only covers `('global_score', 'score', 'seq_recovery')`.  For long-running sessions, export `df_generated` to Parquet or CSV before the session ends if you need these columns persisted.

---

## WorkingSet and Stage Ordering

`WorkingSet` is backed by `frozenset` (immutable). `process_ws()` creates a new `WorkingSet` from `df_generated`; the input working-set is never mutated.  In `run_async()`, non-generation stages receive a `deepcopy` of the stage object so shared state cannot bleed across concurrent stage executions.

---

## Rate Limiting and Retry

All three config runners (`_run_saturation_mutagenesis`, `_run_iterative_masking_dms`, `_run_direct_generation`) construct a `BioLMApiClient` with a shared `asyncio.Semaphore` that limits concurrent requests.  Transient API failures are caught per-batch with `try/except Exception`; the batch is padded with empty results and a `logger.warning` is emitted.  The run continues to completion rather than aborting on a single failed batch.
