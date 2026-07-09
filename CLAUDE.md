# CLAUDE.md — BioLM Python SDK

This file captures key architectural knowledge, confirmed bugs, performance notes, and future directions for the `biolm` package, particularly the pipeline subsystem.

---

## Project Overview

`biolm` is an async-first Python SDK for the BioLM API (`api.biolm.ai`), which provides access to protein/DNA language models (ESM2, ESMFold, ProteinMPNN, ProGen2, etc.) for predictions, embeddings, structure prediction, and sequence generation.

The SDK has two layers:
- **Direct API client** (`biolm/core/http.py`): `BioLMApiClient` (native async), `BioLMApi` (sync wrapper via `synchronicity`)
- **Pipeline framework** (`biolm/pipeline/`): Multi-stage orchestration with DuckDB caching, streaming, resumability, and dependency resolution

**Auth**: `BIOLM_TOKEN` env var → Bearer token header. Get tokens at https://biolm.ai/ui/accounts/user-api-tokens/

---

## Architecture

### Pipeline Modules

| File | Purpose |
|------|---------|
| `pipeline/base.py` | `BasePipeline`, `Stage` (abstract), `StageResult` |
| `pipeline/data.py` | `DataPipeline`, `PredictionStage`, `FilterStage`, `ClusteringStage`, `Predict()`, `Embed()` |
| `pipeline/generative.py` | `GenerativePipeline`, `ScoringProtocolConfig`, `GenerativeProtocolConfig`, `SaturationMutagenesisConfig`, `IterativeMaskingDMSConfig`, `DirectGenerationConfig` |
| `pipeline/datastore_duckdb.py` | `DuckDBDataStore` — DuckDB + Parquet columnar storage |
| `pipeline/filters.py` | `ThresholdFilter`, `RankingFilter`, `DiversitySamplingFilter`, `HammingDistanceFilter`, etc. |
| `pipeline/mlm_remasking.py` | `MLMRemasker`, `RemaskingConfig` — iterative masked-LM variant generation |
| `pipeline/clustering.py` | `SequenceClusterer`, `DiversityAnalyzer` |
| `pipeline/visualization.py` | `PipelinePlotter` — funnel, PCA/UMAP, temperature scan, distributions |
| `pipeline/async_executor.py` | `AsyncBatchExecutor`, `CachingExecutor`, `StreamingExecutor` |
| `core/http.py` | `BioLMApiClient` (async), `HttpClient`, rate limiting, retry, connection pooling |

### Data Flow

```
sequences (list/CSV/FASTA)
       ↓
  DataPipeline._get_initial_data()
       ↓ (adds sequence_ids via DuckDBDataStore)
  BasePipeline.run_async()
       ├── _resolve_dependencies() → topological sort into levels
       └── for each level:
             ├── Single stage → _execute_stage() OR _execute_stage_streaming()
             └── Multiple stages → asyncio.gather(*[_execute_stage(...)])
                   ↓
             Stage.process(df, datastore) → StageResult
             (PredictionStage: calls BioLMApiClient, stores in DuckDB)
             (FilterStage: applies filter_func)
             (ClusteringStage: SequenceClusterer)
                   ↓
             stage_results[name] = result
             _stage_data[name] = df_out
             df_current = df_out
```

### DuckDB DataStore Design

```
DuckDB tables (pipeline.duckdb):
  sequences         — sequence_id, sequence, length, hash (SHA-256[:16], indexed unique)
  predictions       — prediction_id, sequence_id, model_name, prediction_type, value (DOUBLE), metadata (JSON)
  embeddings        — embedding_id, sequence_id, model_name, layer, embedding_path (Parquet file), dimension
  structures        — structure_id, sequence_id, model_name, format, structure_path, plddt
  pipeline_runs     — run_id, pipeline_type, config (JSON), status, created_at, updated_at
  stage_completions — stage_id, run_id, stage_name, status, input_count, output_count, completed_at

Parquet files (pipeline_data/):
  embeddings/emb_*.parquet  — actual numpy arrays (columnar, snappy-compressed)
  structures/               — PDB/CIF files
```

---

## CONFIRMED BUGS

> **Status**: All critical and high-severity bugs below were fixed in the `chance/pipelines` branch.
> The DataStore now has proper tables, bulk SQL methods, and the Stage protocol returns `(df, StageResult)`.

### CRITICAL — Crash at Runtime (FIXED)

**Bug 1: `mark_stage_complete()` receives unexpected `status` kwarg**
- `base.py:243` calls: `self.datastore.mark_stage_complete(stage_id=..., run_id=..., stage_name=..., input_count=..., output_count=..., status='completed')`
- `datastore_duckdb.py:501` defines: `def mark_stage_complete(self, run_id, stage_name, stage_id, input_count, output_count)` — no `status` param
- **Effect**: `TypeError` raised after every stage completes. No pipeline can finish successfully.
- **Fix**: Remove `status='completed'` from the call in `base.py:243`, or add `status: str = 'completed'` to the function signature.

**Bug 2: `add_generation_metadata()` not implemented in DuckDBDataStore**
- `generative.py:268` calls `datastore.add_generation_metadata(seq_id, model_name=..., temperature=..., ...)`
- `DuckDBDataStore` has no such method and no `generation_metadata` table
- **Effect**: `GenerativePipeline` crashes for every generated sequence
- **Fix**: Add `generation_metadata` table to `_init_schema()` and implement `add_generation_metadata()`

**Bug 3: `export_to_dataframe()` not implemented in DuckDBDataStore**
- `generative.py:502` calls `self.datastore.export_to_dataframe(include_sequences=True, include_generation_metadata=True)`
- `DuckDBDataStore` only has `export_to_parquet()`, not `export_to_dataframe()`
- **Effect**: `GenerativePipeline.run_async()` crashes after generation completes
- **Fix**: Implement `export_to_dataframe()` that joins sequences + predictions + generation_metadata into a flat DataFrame

**Bug 4: `pipeline_metadata` table missing in ClusteringStage**
- `data.py:599` executes: `"INSERT OR REPLACE INTO pipeline_metadata (key, value) VALUES (?, ?)"`
- No `pipeline_metadata` table is created in `DuckDBDataStore._init_schema()`
- **Effect**: `OperationalError` every time a `ClusteringStage` runs
- **Fix**: Add `pipeline_metadata` table to `_init_schema()`, or store clustering metadata as a prediction/metadata entry using existing tables

**Bug 5: `get_embeddings_by_sequence()` return type mismatch (tuple unpacking)**
- `get_embeddings_by_sequence()` returns `List[Dict]` (see `datastore_duckdb.py:463-470`)
- Called with tuple unpacking in two places:
  - `data.py:560` (ClusteringStage): `_, embedding = emb_list[0]` — fails, `emb_list[0]` is a dict
  - `data.py:1185` (Embed() function): `_, embedding = emb_list[0]` — same crash
- **Effect**: `ValueError: too many values to unpack` when using embedding-based clustering or `Embed()` convenience function
- **Fix**: Replace with `embedding = emb_list[0].get('embedding')` (after loading with `load_data=True`)

### HIGH — Logic Bugs (Silent Failures) (FIXED)

**Bug 6: FilterStage output discarded in batch mode**
- `FilterStage.process()` creates `df_filtered = self.filter_func(df)` (new DataFrame) but returns only `StageResult`
- `_execute_stage()` at `base.py:256` returns `(df_input, result)` — always the original unfiltered input
- **Effect**: **Filters are completely silently ignored in non-streaming batch mode**. All sequences pass every filter.
- Streaming mode works correctly via `_execute_stage_streaming()` because it applies `next_stage.filter_func(chunk_df)` directly
- **Fix**: `FilterStage.process()` must either modify `df` in-place or the `Stage` protocol must return `(pd.DataFrame, StageResult)`. Recommended: change `_execute_stage()` to return the df from the stage, and update `FilterStage.process()` to return a tuple.

**Bug 7: `PredictionStage` shuts down API client after each batch call**
- `data.py:380-384` has `finally: await self._api_client.shutdown()` — tears down the client at end of every `process()` call
- The client is stored as `self._api_client` for supposed reuse across calls
- **Effect**: Connection pooling is defeated; every stage run creates and destroys an HTTP client
- **Fix**: Remove the `shutdown()` from the `finally` block; let the connection pool manage lifecycle, or shut down only on pipeline completion

**Bug 8: `_count_existing_sequences()` uses unregistered DataFrame**
- `data.py:742-751`: creates `df_check = pd.DataFrame(...)` then passes it to `self.datastore.query("SELECT ... FROM df_check ...")`
- DuckDB's `query()` method calls `self.conn.execute(sql)` without registering `df_check` with the connection
- **Effect**: `CatalogException: Table with name df_check does not exist` in diff mode
- **Fix**: Register via `self.conn.register('df_check', df_check)` before the query, then unregister after

### MEDIUM — Design Gaps (FIXED)

**Bug 9: `GenerativePipeline._get_initial_data()` flow broken**
- `generative.py:356-367`: checks `self._stage_data['generation']` to return generated sequences
- But `_stage_data` is only populated by `run_async()` itself, and `_get_initial_data()` is called by `super().run_async()` which runs AFTER the generation stage override
- Flow: `run_async()` → runs gen stage → exports via `export_to_dataframe()` (Bug 3) → calls `super().run_async()` → calls `_get_initial_data()` (now works) → downstream stages see generated sequences
- The overall flow has the right intention but is broken by Bug 3 (`export_to_dataframe()` missing)

**Bug 10: `add_sequences_batch()` uses implicit DataFrame registration**
- `datastore_duckdb.py:217-246`: queries `FROM df_new` and `FROM df_to_insert` without calling `conn.register()`
- DuckDB's Python client supports querying local DataFrames by variable name in the calling scope, but this relies on undocumented variable scope introspection
- **Risk**: May break across DuckDB versions or in certain async contexts
- **Fix**: Use explicit `conn.register('df_new', df_new)` / `conn.unregister('df_new')` around queries

---

## DuckDB DataStore: Performance Notes

### What's Fast ✅
- Anti-join deduplication in `add_sequences_batch()`: vectorized `LEFT JOIN ... WHERE t.hash IS NULL`
- Batch inserts: `INSERT INTO ... SELECT FROM df` (no row-by-row inserts)
- Embeddings in separate Parquet files: avoids storing large arrays in RDBMS
- SHA-256 hash index for O(log n) sequence lookups
- `query_results()` with predicate pushdown: only loads matching rows

### Performance Bottlenecks (FIXED in `chance/pipelines`)
- **N+1 cache check → vectorized anti-join**: `PredictionStage` now calls `datastore.get_uncached_sequence_ids(sequence_ids, ...)` which does a single `LEFT JOIN ... WHERE p.prediction_id IS NULL` instead of N `has_prediction()` calls.
- **N+1 result merge → single JOIN**: `PredictionStage` now calls `datastore.get_predictions_bulk(sequence_ids, ...)` and merges with `df.merge()` — one SQL round-trip.
- **GenerationStage batch insert**: uses `add_sequences_batch()` for all generated sequences instead of per-row `add_sequence()`.
- **`_count_existing_sequences` → `count_matching_sequences()`**: uses hash-join in DuckDB with explicit `conn.register()`.
- **`_sequence_counter` in-memory drift** (unfixed): counter initialized from `MAX(sequence_id)` at startup; safe for single-process use but can drift under concurrent writers.

---

## Test Coverage Gaps

| Area | Status | Notes |
|------|--------|-------|
| Stage dependency resolution | ✅ Tested | `test_pipeline.py` |
| DuckDB batch ops, anti-join dedup | ✅ Tested | `test_duckdb_datastore.py` |
| Filter logic (isolated) | ✅ Tested | `test_filters.py` |
| MLM remasking | ✅ Tested | `test_mlm_remasking.py` |
| Clustering algorithms | ✅ Tested | `test_clustering.py` |
| **FilterStage actually filtering in batch mode** | ❌ NOT TESTED | Bug 6 would be caught here |
| **GenerativePipeline end-to-end** | ❌ NOT TESTED | Bugs 2, 3 would be caught |
| **mark_stage_complete() kwarg crash** | ❌ NOT TESTED | Bug 1 would be caught |
| **Embedding tuple-unpack** | ❌ NOT TESTED | Bug 5 would be caught |
| API client shutdown/reuse | ❌ NOT TESTED | Bug 7 would be caught |
| Diff mode `_count_existing_sequences()` | ❌ NOT TESTED | Bug 8 would be caught |
| Streaming actually streams incrementally | ❌ Structural only | Need timing-based test |
| Concurrency / race conditions | ❌ Not tested | |
| Large datasets (>10k sequences) | ❌ Not tested | Performance bottlenecks above |
| Pipeline cleanup / connection close | ❌ Not tested | Resource leak risk |

**Priority tests to write:**
1. `test_filter_stage_batch_mode()` — verify filtered rows are actually removed
2. `test_mark_stage_complete()` — verify no TypeError
3. `test_generative_pipeline_e2e()` — with mocked API
4. `test_embed_function()` — verify embedding dict access
5. `test_diff_mode_count_existing()` — verify DataFrame registration

---

## API Ecosystem

### API Client Patterns
```python
# Native async (use in pipeline stages):
from biolmai.client import BioLMApiClient
api = BioLMApiClient('esmfold', semaphore=asyncio.Semaphore(5), retry_error_batches=True)
results = await api.predict(items=[{'sequence': 'MKTAYIAKQRQ'}])
await api.shutdown()

# Sync convenience (scripts/notebooks):
import biolmai
result = biolmai.biolm(entity='esm2-8m', action='encode', items=['MKLLIV'])
```

### Rate Limiting
- `BioLMApiClient` auto-fetches `throttle_rate` from schema API (`/model/action/schema/`)
- Semaphore limits concurrent requests (default 16, configurable)
- Retry on network errors with exponential backoff (3 retries, 1s/2s/4s)
- Request compression: gzip for payloads >256 bytes

---

## Future Directions

1. **Return `(pd.DataFrame, StageResult)` from `Stage.process()`** — eliminates in-place mutation, fixes Bug 6 cleanly, enables proper parallel stage merging
2. **Vectorized cache check** — single anti-join `LEFT JOIN predictions WHERE sequence_id IN (...)` instead of N individual `has_prediction()` calls; 100-1000x faster for large batches
3. **`generation_metadata` table** — add to schema to make `GenerativePipeline` work (also `pipeline_metadata` for clustering)
4. **`export_to_dataframe()`** — implement in `DuckDBDataStore` by pivoting predictions table into wide format
5. **Pipeline as context manager** — `__enter__`/`__exit__` to auto-close DuckDB connection
6. **Embedding batch load** — `get_embeddings_bulk(sequences, model_name)` using one JOIN + batch Parquet reads
7. **Schema-driven value extraction** — `_extract_prediction_value()` is hardcoded for known field names; query the schema API to know response structure dynamically
8. **Streaming resumability** — checkpoint which chunks are processed for recovery after failure
9. **Integration tests with httpx mock transport** — verify actual batching, streaming, and error recovery end-to-end without real API calls
10. **Diff mode visualization** — `get_merged_results()` shows new vs. cached counts in pipeline summary

---

## Key File Locations

```
biolm/
  pipeline/
    base.py                  # BasePipeline, Stage, StageResult
    data.py                  # DataPipeline, PredictionStage, FilterStage, ClusteringStage
    generative.py            # GenerativePipeline, GenerationStage, GenerationConfig
                             # ScoringProtocolConfig, GenerativeProtocolConfig (base classes)
                             # SaturationMutagenesisConfig  — single-mutant library + scoring
                             # IterativeMaskingDMSConfig    — greedy MLM argmax DMS
                             # DirectGenerationConfig       — ProteinMPNN / DSM / AntiFold
    datastore_duckdb.py      # DuckDBDataStore (primary datastore — NOT datastore.py)
    filters.py               # All filter classes
    mlm_remasking.py         # MLMRemasker, RemaskingConfig
    clustering.py            # SequenceClusterer, DiversityAnalyzer
    visualization.py         # PipelinePlotter
    async_executor.py        # AsyncBatchExecutor, StreamingExecutor
    pipeline_def.py          # PipelineDef serialization (to_spec / from_db roundtrip)
  core/
    http.py                  # BioLMApiClient (native async), HttpClient, rate limiter
    auth.py                  # Token auth
  client.py                  # Re-exports BioLMApiClient
  models.py                  # Model class (sync user interface)
  biolmai.py                 # biolm() convenience function
  protocols/                 # Protocol YAML: validation, hosted runs, local runtime
    validation.py            # Schema + semantic validation
    model.py                 # Protocol class (load, validate, execute, inspect)
    runs.py                  # ProtocolClient, ProtocolRun (hosted API)
    runtime/                 # Local compiler + executor (biolm[pipeline])
tests/
  test_pipeline.py           # Core pipeline logic tests
  test_protocol_runtime.py   # Local protocol compiler/executor tests
  test_protocols.py          # Protocol validation tests
  test_duckdb_datastore.py   # DuckDB-specific tests
  test_advanced_features.py  # Streaming, resumability, error handling
  test_filters.py            # Filter unit tests
  test_clustering.py         # Clustering unit tests
  test_mlm_remasking.py      # MLM remasking tests
  test_design_primitives.py  # SaturationMutagenesis, IterativeMaskingDMS, pipeline shorthands
PIPELINE_ARCHITECTURE_GUIDE.md  # Technical deep-dive
PIPELINE_QUICKSTART.md           # User-facing quickstart
```
