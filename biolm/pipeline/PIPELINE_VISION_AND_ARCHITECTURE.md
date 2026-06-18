# BioLM Pipeline: Vision, Motivation & Architecture

**Last Updated**: 2026-03-18

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Vision: A Protein Design OS](#2-vision-a-protein-design-os)
3. [Layer 0 — API Client (`core/http.py`)](#3-layer-0--api-client)
4. [Layer 1 — DuckDB DataStore (`datastore_duckdb.py`)](#4-layer-1--duckdb-datastore)
5. [Layer 2 — Stage Abstractions (`base.py`)](#5-layer-2--stage-abstractions)
6. [Layer 3 — WorkingSet Transport](#6-layer-3--workingset-transport)
7. [Layer 4 — PredictionStage and Extraction (`data.py`)](#7-layer-4--predictionstage-and-extraction)
8. [Layer 5 — Filters (`filters.py`)](#8-layer-5--filters)
9. [Layer 6 — Generation (`generative.py`)](#9-layer-6--generation)
10. [Layer 7 — Pipeline Definition Persistence (`pipeline_def.py`)](#10-layer-7--pipeline-definition-persistence)
11. [Layer 8 — Multi-Column Inputs and Context (`base.py`, `data.py`)](#11-layer-8--multi-column-inputs-and-context)
12. [Execution Model](#12-execution-model)
13. [API Surface Design Decisions](#13-api-surface-design-decisions)
---

## 1. The Problem

Protein engineering with ML models involves a recurring workflow:

```
Generate candidates → Score/filter → Generate more → Score/filter → Analyze
```

Naively this means:
- Manual API calls per model, per sequence batch
- Re-running from scratch on any failure
- Keeping results in notebooks that break with large data
- No caching: the same ESMFold call for "MKTAYIAK..." costs real money and time every run
- No parallelism: most users call models sequentially

**The pain compounds quickly.** A realistic protein design campaign might involve:
1. 500 sequences from ProteinMPNN at T=0.3
2. 500 more from DSM at T=1.0
3. Filter for valid amino acids
4. Score all 1000 with ESMFold (expensive: ~2s each = 33 min)
5. Filter for pLDDT > 70
6. Score survivors with Tm predictor
7. Rank top 20 by Tm × solubility

If step 6 crashes, you want to resume from step 5 — not redo ESMFold for 500 sequences.

---

## 2. Vision: A Protein Design OS

The pipeline is designed to be the **orchestration layer** for protein ML experiments:

- **Declarative**: Describe *what* you want (stages, filters, dependencies), not *how* to call APIs
- **Resumable**: Any crash → resume from last checkpoint with zero re-computation
- **Cached**: Same sequence + same model → single DB lookup, no API call
- **Parallel**: Independent stages run concurrently by default
- **Memory-efficient**: 1M sequences tracked as 28 MB of integers, not 500 MB DataFrames
- **SQL-native**: Filter/query results with full SQL expressiveness via DuckDB
- **Recoverable**: Kernel dies overnight → `DataPipeline.from_db("my.duckdb")` reconstructs and resumes

The high-level user experience is intentionally simple:

```python
pipeline = DataPipeline(sequences=my_sequences)

pipeline.add_prediction("temberture-regression",
    extractions="prediction",   # API response key
    columns="tm",               # output column name
)
pipeline.add_filter(ThresholdFilter("tm", min_value=48.0))

results = pipeline.run()
df = pipeline.get_final_data()
```

Everything else — batching, caching, deduplication, DuckDB storage, parallel execution — is invisible.

---

## 3. Layer 0 — API Client

**File**: [`biolmai/core/http.py`](biolmai/core/http.py)

The foundation is an async HTTP client that handles all BioLM API communication:

```python
# From core/http.py
class BioLMApiClient:
    """Native async client for BioLM API with rate limiting, retry, connection pooling."""

    def __init__(self, entity: str, semaphore: asyncio.Semaphore = None, ...):
        # entity = model slug: "temberture-regression", "esm2-8m", etc.
        self._semaphore = semaphore or asyncio.Semaphore(16)

    async def predict(self, items: list[dict]) -> list[dict]: ...
    async def encode(self, items: list[dict]) -> list[dict]: ...
    async def score(self, items: list[dict]) -> list[dict]: ...
    async def generate(self, items: list[dict]) -> list[dict]: ...
```

**Key design decisions:**

| Decision | Justification |
|----------|---------------|
| **Native async** (`asyncio`) | Enables `asyncio.gather()` for concurrent batch dispatch. Sequential HTTP calls would be 5-10× slower. |
| **Semaphore per stage** | Each `PredictionStage` gets its own `asyncio.Semaphore(max_concurrent)` to cap in-flight requests without a global bottleneck. |
| **Auto-fetches throttle rate** | `BioLMApiClient` fetches `throttle_rate` from the model's schema API (`/model/action/schema/`) so the pipeline self-adapts to per-model rate limits. |
| **Exponential backoff** | 3 retries at 1s/2s/4s — handles transient API errors without crashing long jobs. |
| **gzip for large payloads** | Payloads >256 bytes are compressed before sending — meaningful for embedding requests with long sequences. |

The pipeline **never calls BioLM directly** — it always goes through `BioLMApiClient`. This enforces rate limiting and retry logic uniformly across all stages.

---

## 4. Layer 1 — DuckDB DataStore

**File**: [`biolmai/pipeline/datastore_duckdb.py`](biolmai/pipeline/datastore_duckdb.py)

The datastore is the single source of truth for all pipeline data. It was chosen over alternatives for specific reasons:

### Why DuckDB?

| Alternative | Problem |
|------------|---------|
| pandas DataFrame | Explodes in memory at >100K sequences with embeddings |
| SQLite | Row-oriented: slow for columnar aggregations and range scans |
| Redis | No persistence, no joins, extra service to run |
| PostgreSQL | Heavy setup, poor local dev experience |
| DuckDB | Columnar, in-process, fast analytics SQL, Parquet native, zero config |

### Schema Overview

```sql
-- Core sequence store: hash-indexed for O(log n) dedup
CREATE TABLE sequences (
    sequence_id INTEGER PRIMARY KEY,
    sequence    VARCHAR NOT NULL,
    length      INTEGER,
    hash        VARCHAR,   -- SHA-256[:16], UNIQUE INDEX
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions: EAV-style to support any model output
-- prediction_type stores the cache key: "{model}::{action}::{response_keys}"
CREATE TABLE predictions (
    prediction_id   INTEGER PRIMARY KEY,
    sequence_id     INTEGER,
    model_name      VARCHAR,
    prediction_type VARCHAR,    -- e.g. "temberture-regression::predict::prediction"
    value           DOUBLE,
    metadata        VARCHAR,    -- JSON for extra data
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings: arrays stored inline as FLOAT[] (no per-file Parquet overhead)
CREATE TABLE embeddings (
    embedding_id INTEGER PRIMARY KEY,
    sequence_id  INTEGER,
    model_name   VARCHAR,
    layer        INTEGER,
    values       FLOAT[],
    dimension    INTEGER,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Structures: PDB/CIF content stored as text (structure_str column)
CREATE TABLE structures (
    structure_id   INTEGER PRIMARY KEY,
    sequence_id    INTEGER,
    model_name     VARCHAR,
    format         VARCHAR,
    structure_path VARCHAR,
    structure_str  TEXT,
    plddt          DOUBLE,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resumability: which stages completed in which run
CREATE TABLE pipeline_runs (
    run_id        VARCHAR PRIMARY KEY,
    pipeline_type VARCHAR,
    config        VARCHAR,    -- JSON
    status        VARCHAR,
    definition_id VARCHAR,    -- FK to pipeline_definitions (for from_db() recovery)
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE stage_completions (
    stage_id     VARCHAR PRIMARY KEY,
    run_id       VARCHAR,
    stage_name   VARCHAR,
    status       VARCHAR,
    input_count  INTEGER,
    output_count INTEGER,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Filter resumability: which sequence_ids passed each filter stage
CREATE TABLE filter_results (
    run_id         VARCHAR,
    stage_name     VARCHAR,
    sequence_id    INTEGER,
    PRIMARY KEY    (run_id, stage_name, sequence_id)
);

-- Cross-session column collision detection
CREATE TABLE pipeline_definitions (
    definition_id    VARCHAR PRIMARY KEY,   -- SHA-256[:16] of stages_json
    pipeline_type    VARCHAR NOT NULL,
    input_schema_json TEXT,
    stages_json      TEXT NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE prediction_column_registry (
    column_name   VARCHAR NOT NULL,
    model_name    VARCHAR NOT NULL,
    action        VARCHAR NOT NULL,
    definition_id VARCHAR NOT NULL,
    stage_name    VARCHAR NOT NULL,
    PRIMARY KEY   (column_name, model_name, action)
);

-- Inter-stage communication
CREATE TABLE pipeline_context (
    run_id    VARCHAR,
    key       VARCHAR,
    value     VARCHAR,    -- JSON-encoded
    PRIMARY KEY (run_id, key)
);
```

### Key Performance Patterns

**Anti-join deduplication** — Inserting 10K sequences checks against existing hashes with a single SQL operation, not N lookups:

```python
# datastore_duckdb.py: add_sequences_batch()
self.conn.register('_df_new', df_new)
result = self.conn.execute("""
    SELECT n.sequence, n.hash, n.length
    FROM _df_new n
    LEFT JOIN sequences s ON n.hash = s.hash
    WHERE s.hash IS NULL          -- only NEW sequences
""").df()
self.conn.unregister('_df_new')
```

**Vectorized cache check** — Before calling the API, find which sequences lack predictions using a single anti-join:

```python
# datastore_duckdb.py: get_uncached_sequence_ids()
return self.conn.execute("""
    SELECT s.sequence_id
    FROM sequences s
    LEFT JOIN predictions p
      ON s.sequence_id = p.sequence_id
     AND p.prediction_type = ?
     AND p.model_name = ?
    WHERE p.prediction_id IS NULL
      AND s.sequence_id IN (SELECT UNNEST(?))
""", [cache_key, model_name, seq_ids]).df()
```

This replaces the original N calls to `has_prediction()`. For 10K sequences, that's 10K SQL round-trips → 1.

**Explicit DataFrame registration** — DuckDB's variable-scope `FROM df` is undocumented behavior that breaks in async contexts. All code uses explicit `conn.register()`:

```python
# Correct pattern — always explicit, always unregister
self.conn.register('_pred_df', df)
result = self.conn.execute("SELECT ... FROM _pred_df").df()
self.conn.unregister('_pred_df')
```

---

## 5. Layer 2 — Stage Abstractions

**File**: [`biolmai/pipeline/base.py`](biolmai/pipeline/base.py)

### `Stage` — The Core Contract

```python
# base.py:162
class Stage(ABC):
    def __init__(self, name, depends_on=None, model_name=None, max_concurrent=10):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @abstractmethod
    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """DuckDB-native processing. Stages operate on sequence_id sets."""
        pass

    def to_spec(self) -> dict:
        """Serialize to dict for pipeline definition persistence.
        Raises NotImplementedError — subclasses must override."""
        raise NotImplementedError(...)
```

**Why `process_ws` instead of `process(df)`?**

The original design passed DataFrames between stages. This caused:
- Memory explosions: 10K sequences × 1024-dim embeddings = 40 MB per stage × N stages
- Repeated materialization: each stage re-read from DuckDB then discarded the DataFrame
- No SQL pushdown: filters ran on pandas, not in DuckDB

The `WorkingSet` design delays materialization until `get_final_data()`.

### `StageResult` — Metrics for Free

```python
# base.py:140
@dataclass
class StageResult:
    stage_name: str
    input_count: int
    output_count: int
    filtered_count: int = 0
    cached_count: int = 0    # how many were in DuckDB cache
    computed_count: int = 0  # how many called the API
    elapsed_time: float = 0.0
```

Every stage returns this. The pipeline aggregates them into `pipeline.stats()` and `pipeline.summary()` — users see at a glance how effective caching is and where time was spent.

### Dependency Resolution — Topological Sort

```python
# base.py:504
def _resolve_dependencies(self) -> list[list[Stage]]:
    """Kahn's algorithm — levels enable parallel execution."""
    in_degree = {s.name: len(s.depends_on) for s in self.stages}
    levels = []
    while in_degree:
        current_level = [s for name, s in stage_map.items() if in_degree[name] == 0]
        levels.append(current_level)
        for stage in current_level:
            del in_degree[stage.name]
            # Decrement dependents' in-degree
            for other in self.stages:
                if stage.name in other.depends_on and other.name in in_degree:
                    in_degree[other.name] -= 1
    return levels
```

Stages without explicit `depends_on` run at level 0 — automatically parallel. Given this:

```python
pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm",
                        stage_name="predict_tm")
pipeline.add_prediction("biolmsol", extractions="solubility_score", columns="solubility",
                        stage_name="predict_sol")
pipeline.add_filter(RankingFilter("solubility", n=20),
                    depends_on=["predict_tm", "predict_sol"])
```

The execution is:
```
Level 0: [predict_tm, predict_sol]  → asyncio.gather() — concurrent!
Level 1: [ranking_filter]           → sequential, after both complete
```

No threading complexity. No locks. Just `asyncio.gather()` + DuckDB.

---

## 6. Layer 3 — WorkingSet Transport

**File**: [`biolmai/pipeline/base.py:91`](biolmai/pipeline/base.py)

```python
@dataclass(frozen=True)
class WorkingSet:
    """Lightweight set of sequence IDs — replaces DataFrame as inter-stage transport.

    Memory: 1M IDs ≈ 28 MB (frozenset[int]) vs 500 MB+ DataFrame.
    """
    sequence_ids: frozenset[int]

    def intersect(self, other: "WorkingSet") -> "WorkingSet":
        return WorkingSet(self.sequence_ids & other.sequence_ids)

    def union(self, other: "WorkingSet") -> "WorkingSet":
        return WorkingSet(self.sequence_ids | other.sequence_ids)
```

**The insight**: Stages don't need data, they need to know *which sequences survive*. All data lives in DuckDB. The WorkingSet is a scoped view — "apply this operation only to these sequence_ids."

**Parallel merge** uses intersection:

```python
# base.py: _execute_stage_ws() for parallel stages
working_set = level_results[0]
for ws in level_results[1:]:
    working_set = working_set.intersect(ws)  # frozenset & — pure Python, O(n)
```

Sequences that didn't complete in *all* parallel stages are dropped. This is correct behavior: if `predict_tm` cached only 800/1000 and `predict_sol` cached 900/1000, you want the 800 that have both.

**Materialization is lazy**:

```python
# base.py: get_final_data()
def get_final_data(self) -> pd.DataFrame:
    last_ws = self._working_sets.get(last_stage_name)
    if last_ws is not None:
        return self.datastore.materialize_working_set(last_ws)
    # Legacy fallback for streaming stages
    return self._stage_data.get(last_stage_name, pd.DataFrame())
```

`materialize_working_set()` does a single SQL JOIN across all prediction columns — one round-trip.

---

## 7. Layer 4 — PredictionStage and Extraction

**File**: [`biolmai/pipeline/data.py`](biolmai/pipeline/data.py)

### The Extraction Model

A key Phase 8 design decision: separate *what API key to read* from *what column to name it*.

**Before (confusing):**
```python
# Old API — prediction_type served double duty as cache key AND column name
pipeline.add_prediction("temberture-regression",
    prediction_type="tm",       # used as both cache key and output column
    extractions="prediction",   # API response key
)
```

**After (clean):**
```python
# New API — three separate concerns
pipeline.add_prediction("temberture-regression",
    extractions="prediction",   # (1) which API response key to read
    columns="tm",               # (2) what to call it in the output DataFrame
    # cache key auto-derived: "temberture-regression::predict::prediction"
)
```

**Why this matters:**
1. You can rename `"prediction"` → `"tm"` without invalidating the DuckDB cache
2. The cache key `"temberture-regression::predict::prediction"` is content-addressed — same model+action+response keys always hit the same cache entry
3. Multiple extractions from one API call are explicit: `extractions=["mean_plddt", "ptm"]`

### `ExtractionSpec` — Declarative Array Reduction

```python
# data.py
@dataclass
class ExtractionSpec:
    response_key: str               # API response dict key
    reduction: Optional[str] = None # "mean", "max", "sum" — collapses per-residue → scalar
```

ESMFold returns per-residue pLDDT (length 150 for a 150-AA sequence). To get a single score:

```python
pipeline.add_prediction("esmfold",
    extractions=[ExtractionSpec("mean_plddt", reduction="mean")],
    columns="plddt",
)
```

The stage applies `np.mean()` before storing to DuckDB. No extra code needed.

### `EmbeddingSpec` — Declarative Embedding Extraction

```python
# data.py
@dataclass
class EmbeddingSpec:
    key: str                    # API response dict key ("embeddings", "seqcoding", "embedding")
    layer: Optional[int] = None # for layered models (ESM2 has 33 layers)
    reduction: Optional[str] = None  # "mean" collapses token-level → sequence-level
```

Different models return embeddings under different keys with different shapes:
- `esm2-8m.encode` → `{"embeddings": [{"embedding": [320-dim]}]}`
- `ablang2.encode` → `{"seqcoding": [480-dim]}`
- `dnabert2.encode` → `{"embedding": [768-dim]}`

Instead of hardcoding key-guessing logic (the old approach), users declare exactly what to extract:

```python
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
)
```

### Cache Dedup + Batch API Flow

```python
# data.py: PredictionStage.process_ws()
async def process_ws(self, ws, datastore, **kwargs):
    all_seq_ids = ws.to_list()

    # 1. Single anti-join: which sequences lack predictions?
    uncached = datastore.get_uncached_sequence_ids(all_seq_ids, cache_key, model_name)
    cached_ids = set(all_seq_ids) - set(uncached)

    # 2. Call API only for uncached (batched, concurrent)
    if uncached:
        await self._run_batched(uncached, datastore)

    # 3. Result WS = all inputs that now have predictions (cached + computed)
    # Sequences where API failed are excluded automatically
    successful_ids = datastore.get_sequence_ids_with_prediction(all_seq_ids, ...)
    return WorkingSet.from_ids(successful_ids), StageResult(...)
```

**Batching + concurrency**:

```python
# data.py: _run_batched()
n_batches = ceil(len(uncached) / batch_size)
flight_semaphore = asyncio.Semaphore(self.max_concurrent)

async def _dispatch_batch(batch_idx):
    async with flight_semaphore:           # limit in-flight API calls
        results = await api.predict(items) # SDK handles further splitting by maxItems
        datastore.add_predictions_bulk(results)

await asyncio.gather(*[_dispatch_batch(i) for i in range(n_batches)])
```

Results stream into DuckDB as batches complete — memory holds at most `max_concurrent` batches at once, not the full dataset.

---

## 8. Layer 5 — Filters

**File**: [`biolmai/pipeline/filters.py`](biolmai/pipeline/filters.py)

### Two Execution Paths

```python
class BaseFilter(ABC):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...  # DataFrame fallback
    def to_sql(self, ws_table: str) -> Optional[str]: ...      # SQL fast path
    def to_spec(self) -> dict: ...                             # serialization for from_db()
```

**SQL-translatable filters** execute entirely in DuckDB — zero DataFrame materialization:

```python
# filters.py: ThresholdFilter.to_sql()
def to_sql(self, ws_table="_filter_ws"):
    parts = [f"SELECT w.sequence_id FROM {ws_table} w"]
    parts.append("INNER JOIN predictions p ON w.sequence_id = p.sequence_id")
    parts.append(f"WHERE p.prediction_type = '{self._cache_key}'")
    if self.min_value is not None:
        parts.append(f"AND p.value >= {self.min_value}")
    if self.max_value is not None:
        parts.append(f"AND p.value <= {self.max_value}")
    return " ".join(parts)
```

The execution in `FilterStage.process_ws()`:

```python
# data.py: FilterStage.process_ws()
sql = self.filter_func.to_sql()
if sql is not None:
    # Pure DuckDB: no DataFrame ever created
    surviving_ids = datastore.execute_filter_sql(ws, sql)
else:
    # Fallback: materialize → filter → extract IDs
    df = datastore.materialize_working_set(ws)
    df_filtered = self.filter_func(df).copy()
    surviving_ids = df_filtered["sequence_id"].tolist()

# Save results for resumability
datastore.save_filter_results(run_id, stage_name, surviving_ids)
return WorkingSet.from_ids(surviving_ids), StageResult(...)
```

**SQL-capable filters** (zero-copy):
- `ThresholdFilter` — `p.value >= X AND p.value <= Y`
- `RankingFilter` — `ORDER BY p.value DESC LIMIT N` scoped to working set
- `SequenceLengthFilter` — `s.length BETWEEN min AND max`
- `ValidAminoAcidFilter` — `regexp_matches(s.sequence, '^[ACDEFGHIKLMNPQRSTVWY]+$')`

**Fallback filters** (materialize then filter):
- `HammingDistanceFilter` — needs numpy pairwise comparison
- `DiversitySamplingFilter` — needs embedding cosine similarity matrix
- `ConservedResidueFilter` — character-level indexing
- `CustomFilter` — arbitrary user callable

This design means a `ThresholdFilter("tm", min_value=50)` on 1M sequences does a single B-tree index scan in DuckDB, returning the surviving `sequence_id`s directly. No Python, no pandas.

### Scoped RankingFilter — Critical Correctness Detail

`RankingFilter` must be scoped to the current working set, not the full database:

```python
# filters.py: RankingFilter.to_sql()
def to_sql(self, ws_table="_filter_ws"):
    direction = "ASC" if self.ascending else "DESC"
    return f"""
        SELECT w.sequence_id
        FROM {ws_table} w
        INNER JOIN predictions p ON w.sequence_id = p.sequence_id
        WHERE p.prediction_type = '{self._cache_key}'
        ORDER BY p.value {direction}
        LIMIT {self.n}
    """
```

The `INNER JOIN` with `ws_table` (the current working set) ensures "top 20 by Tm" means top 20 among sequences that *survived upstream filters*, not top 20 globally.

---

## 9. Layer 6 — Generation

**File**: [`biolmai/pipeline/generative.py`](biolmai/pipeline/generative.py)

### Two Generation Backends

**`DirectGenerationConfig`** — For structure-conditioned or sequence-conditioned models:

```python
# generative.py
@dataclass
class DirectGenerationConfig:
    model_name: str          # "protein-mpnn", "dsm-650m-base", "zymctrl", etc.
    item_field: str          # "pdb", "sequence", "ec_number", "context"
    structure_path: Optional[str] = None
    sequence: Optional[str] = None
    params: dict = field(default_factory=dict)
    structure_from_model: Optional[str] = None  # read structure from upstream stage
```

**`RemaskingConfig`** — Iterative MLM refinement (ESM2):

```python
# mlm_remasking.py
@dataclass
class RemaskingConfig:
    model_name: str          # "esm2-650m", "esm2-8m", etc.
    mask_fraction: float     # 0.15 = mask 15% of positions per round
    num_iterations: int      # how many mask-predict-replace rounds
    temperature: float       # sampling temperature for logit decoding
```

The remasking process runs locally:
1. Take parent sequence "MKLLIV..."
2. Randomly mask 15% of positions: "MKL\<mask>IV..."
3. Send to ESM2 predict API → get logit distributions per masked position
4. Sample from logits with temperature → fill masked positions
5. Repeat `num_iterations` times

This produces diverse variants without needing a dedicated generation API endpoint — it uses the standard `predict` action.

### Generation Stage and Downstream Flow

`GenerativePipeline.run_async()` runs the `GenerationStage` first, stores the generated sequences in DuckDB, builds a `WorkingSet` from the new IDs, then passes that WorkingSet to `super().run_async()` for downstream prediction and filter stages. Generated sequences are indistinguishable from CSV-loaded sequences at the DuckDB level.

**GEN-05 — WorkingSet ID forwarding**: `GenerationStage.process_ws()` forwards the input WorkingSet's IDs as `ws_ids` to `process()`. This scopes structure lookups in `DirectGenerationConfig` (via `structure_from_model`) to sequences that exist in the current pipeline run, preventing cross-run contamination when a DuckDB file is reused across experiments.

```python
# generative.py: GenerationStage.process_ws()
async def process_ws(self, ws, datastore, **kwargs):
    # Forward input WS IDs so structure lookups stay scoped to this run
    if ws and "ws_ids" not in kwargs:
        kwargs["ws_ids"] = ws.to_list()
    df_generated, result = await self.process(pd.DataFrame(), datastore, **kwargs)
    ...
```

### Structure Passing Between Stages

```python
# scripts/pipeline_antibody_antifold.py
config = DirectGenerationConfig(
    model_name="protein-mpnn",
    structure_from_model="esmfold",  # reads from DuckDB structures table
    params={"batch_size": 50},
)
```

`_run_direct_generation` reads from the `structures` table:

```python
# generative.py: _run_direct_generation()
if config.structure_from_model:
    # Parameterized query — no SQL injection risk
    rows = self.datastore.conn.execute(
        "SELECT sequence_id, structure_str FROM structures WHERE model_name = ?",
        [config.structure_from_model]
    ).df()
    # Use each stored structure as input to the generation model
```

The `PipelineContext` provides a higher-level API for this:

```python
# In a custom stage:
struct = pipeline.context.get_structure(seq_id, "esmfold")
all_structs = pipeline.context.get_structures_for_ws(working_set, "esmfold")
```

---

## 10. Layer 7 — Pipeline Definition Persistence

**File**: [`biolmai/pipeline/pipeline_def.py`](biolmai/pipeline/pipeline_def.py)

### The Problem It Solves

A user runs a 6-hour pipeline. The kernel dies at hour 5. Without definition persistence, they must:
1. Recreate the pipeline object from memory (if they remember the parameters)
2. Set `resume=True`
3. Reconstruct 6 `add_prediction()` / `add_filter()` calls exactly

With definition persistence:
```python
pipeline = DataPipeline.from_db("my_pipeline.duckdb")
pipeline.run(resume=True)  # Skips all completed stages
```

### Content-Hash Identity

```python
# pipeline_def.py
def _pipeline_def_hash(pipeline_type, input_schema_cols, stages_specs) -> str:
    payload = json.dumps({
        "pipeline_type": pipeline_type,
        "input_schema": sorted(input_schema_cols) if input_schema_cols else None,
        "stages": stages_specs,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

Same stages + same params = same hash = one row in `pipeline_definitions`. If you run the exact same pipeline 100 times (with `resume=True`), you get 100 rows in `pipeline_runs` but only 1 row in `pipeline_definitions`. The stages are stored once.

### Stage Serialization: `to_spec()`

Each stage implements `to_spec()` to return a fully serializable dict:

```python
# data.py: PredictionStage.to_spec()
def to_spec(self) -> dict:
    return {
        "type": "PredictionStage",
        "name": self.name,
        "model_name": self.model_name,
        "action": self.action,
        "resolved": [{"response_key": r.response_key, "column": r.column,
                       "reduction": r.reduction} for r in self._resolved],
        "params": self.params,
        "batch_size": self.batch_size,
        "max_concurrent": self.max_concurrent,
        "item_columns": self.item_columns,
        "embedding_extractor": (
            {"type": "EmbeddingSpec", "key": e.key, "layer": e.layer,
             "reduction": e.reduction}
            if isinstance(self._embedding_extractor, EmbeddingSpec) else None
        ),
        "depends_on": self.depends_on,
    }
```

`filter_from_spec()` and `stage_from_spec()` in `pipeline_def.py` reconstruct from these dicts. `CustomFilter` and custom callable `embedding_extractor` raise `NotImplementedError` with instructions to re-attach manually — they cannot be serialized.

### Cross-Session Column Collision Detection

The `prediction_column_registry` table ensures that if you reuse a DuckDB file and accidentally configure a stage that would overwrite an existing column with different model output:

```python
# base.py: add_stage()
for r in getattr(stage, "_resolved", []):
    existing = self.datastore.get_column_registry_entry(r.column)
    if existing and (existing["model_name"], existing["action"]) != (stage.model_name, stage.action):
        raise ValueError(
            f"Column '{r.column}' was previously used by "
            f"model='{existing['model_name']}', action='{existing['action']}'. "
            "Use a different column name via the 'columns' parameter."
        )
```

This fires at `add_stage()` time (when the pipeline is being configured), not at `run()` time — failing fast before any computation.

### `from_db()` — Full Recovery Walkthrough

After a kernel crash, the entire pipeline is reconstructable from the DuckDB file alone:

```python
pipeline = BasePipeline.from_db("petase.duckdb", run_id="petase_v1")
pipeline.run(resume=True)  # generation skipped, all predictions served from cache
```

**Step 1 — Look up the definition for that run**

`pipeline_runs.definition_id` links every run to its stored definition:

```sql
SELECT definition_id FROM pipeline_runs WHERE run_id = 'petase_v1'
-- → '873d0055e888847b3bb000f90b0f80a3'
```

**Step 2 — Load `stages_json` from `pipeline_definitions`**

```sql
SELECT stages_json FROM pipeline_definitions WHERE definition_id = '873d0055...'
```

`stages_json` is the complete, fully-parameterized stage array — not just names. Example entry for a generation stage:

```json
{
  "type": "GenerationStage",
  "name": "generation",
  "configs": [
    {
      "type": "RemaskingConfig",
      "model_name": "esmc-300m",
      "action": "predict",
      "mask_fraction": 0.15,
      "num_iterations": 3,
      "conserved_positions": [164, 209, 241],
      "parent_sequence": "SNPYARGPNPT...",
      "num_variants": 5
    },
    {
      "type": "DirectGenerationConfig",
      "model_name": "protein-mpnn",
      "item_field": "pdb",
      "params": {"batch_size": 5, "temperature": 0.1}
    }
  ]
}
```

And for a prediction stage:

```json
{
  "type": "PredictionStage",
  "name": "predict_tm",
  "model_name": "temberture-regression",
  "action": "predict",
  "resolved": [{"response_key": "prediction", "column": "tm", "reduction": null}],
  "depends_on": ["filter_length"]
}
```

**Step 3 — Deserialize each stage** (`pipeline_def.py: stage_from_spec()`)

| `"type"` | Reconstructed as | What's restored |
|---|---|---|
| `GenerationStage` | `GenerationStage(configs=[...])` | All generation configs including parent_sequence, conserved positions |
| `PredictionStage` | `PredictionStage(model_name=..., extractions=..., columns=...)` | Model, action, extraction→column mappings, batch size, item_columns |
| `FilterStage` | `FilterStage(filter_func=<filter>)` | Filter type + all parameters (column, min, max, n, method) |
| `ClusteringStage` | `ClusteringStage(method=..., n_clusters=...)` | Algorithm and hyperparameters |

`CustomFilter` and custom callable `embedding_extractor` raise `NotImplementedError` — they are not serializable by design. The error message instructs the user to re-attach manually.

**Step 4 — Resume execution**

`run(resume=True)` checks `stage_completions` before executing each stage:

```
stage_id                   │ status
───────────────────────────┼───────────
petase_v1_generation       │ completed   ← skip generation; reload from generation_metadata
petase_v1_filter_length    │ completed   ← reload passing IDs from filter_results
petase_v1_predict_tm       │ completed   ← all predictions in cache; 0 API calls
petase_v1_filter_tm        │ completed   ← reload passing IDs from filter_results
petase_v1_rank_sol         │ completed   ← reload passing IDs from filter_results
```

The **generation stage** has a special reload path: rather than re-running inference, it reads `generation_metadata WHERE run_id = ?` to reconstruct the exact set of sequence IDs from the original run — this is the only stage that uses `run_id`-scoped metadata for its working set.

All other stages reload via:
- **PredictionStage**: queries `predictions` table, intersects with current working set
- **FilterStage**: calls `get_filter_results(run_id, stage_name)` which returns the IDs that originally passed

### Why `run_id` Must Be Preserved

The resume check uses `stage_id = f"{run_id}_{stage_name}"` as the lookup key. If `from_db()` were given a new `run_id`, none of the `stage_completions` rows would match and every stage would re-run from scratch. `from_db(run_id="petase_v1")` explicitly preserves the original `run_id` so all stage completion records are found.

### What Cannot Be Recovered Automatically

| Scenario | Behaviour |
|---|---|
| `CustomFilter` | `NotImplementedError` — re-attach with `pipeline.stages[i].filter_func = my_filter` |
| Custom lambda `embedding_extractor` | `NotImplementedError` — use `EmbeddingSpec` instead |
| `CofoldingPredictionStage.static_entities` | `UserWarning` — re-attach manually before `.run()` |
| `stage_completions` row missing (crashed mid-stage) | Stage re-runs; prediction cache prevents redundant API calls |

---

## 11. Layer 8 — Multi-Column Inputs and Context

**File**: [`biolmai/pipeline/base.py`](biolmai/pipeline/base.py), [`biolmai/pipeline/data.py`](biolmai/pipeline/data.py)

### InputSchema — Antibodies and Multi-Chain Proteins

Antibody models take paired heavy+light chains. The naïve approach would be to join them into a single string — but then you lose the ability to use them as separate API fields.

```python
# base.py:23
@dataclass(frozen=True)
class InputSchema:
    columns: list[str]    # e.g. ["heavy_chain", "light_chain"]

    def hash_row(self, row: dict[str, str]) -> str:
        # SHA-256 of all column values sorted alphabetically
        # Same heavy + different light = different hash (correct)
        parts = [str(row.get(c, "")) for c in sorted(self.columns)]
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]
```

Columns are stored **directly on the `sequences` table** via `ALTER TABLE ADD COLUMN`:

```python
# datastore_duckdb.py: ensure_input_columns()
for col in columns:
    try:
        self.conn.execute(f"ALTER TABLE sequences ADD COLUMN {col} VARCHAR")
    except duckdb.CatalogException:
        pass  # column already exists (idempotent)
```

This means `materialize_working_set()` and SQL filters see `heavy_chain` and `light_chain` as real first-class columns — no EAV lookups, no JSON parsing.

### `item_columns` — API Field Mapping

```python
# data.py: PredictionStage
# item_columns = {api_field: df_column}
# e.g. {"heavy": "heavy_chain", "light": "light_chain"}

pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
)
```

When building API requests, the stage maps from DuckDB column names to the API's expected field names. Single-chain models use `{"sequence": "sequence"}` by default.

### `PipelineContext` — Inter-Stage Communication

```python
# base.py:51
class PipelineContext:
    """Key-value store backed by DuckDB pipeline_context table."""

    def set(self, key, value):
        self._datastore.set_context(self._run_id, key, value)

    def get(self, key, default=None):
        val = self._datastore.get_context(self._run_id, key)
        return val if val is not None else default

    def get_structure(self, sequence_id, model_name=None):
        """Convenience: fetch predicted structure for downstream generation."""
        return self._datastore.get_structure(sequence_id, model_name)
```

Context values are JSON-serialized to the `pipeline_context` table, scoped by `run_id`. This enables patterns like:

```python
pipeline.context.set("experiment", "thermostability_screen")
pipeline.context.set("config", {"target_tm": 65, "n_variants": 200})

# In a downstream custom stage:
config = pipeline.context.get("config")
```

---

## 12. Execution Model

### Full Data Flow

```
User code                   BasePipeline                  DuckDB
─────────────────────────   ──────────────────────────    ─────────────────────────
DataPipeline(sequences=…)
  → _get_initial_data()  →  add_sequences_batch()      →  INSERT INTO sequences
                              (anti-join dedup)              (skip existing hashes)

pipeline.run()
  → resolve_dependencies()  [Level 0 stages]            asyncio.gather():
  → _execute_stage_ws()  →  PredictionStage.process_ws:
                              get_uncached_seq_ids()    →  LEFT JOIN predictions
                              [call API for uncached]
                              add_predictions_bulk()    →  INSERT INTO predictions
                              get_seq_ids_with_pred()   →  SELECT sequence_id
                         ←   WorkingSet({ids…})

                            FilterStage.process_ws:
                              filter.to_sql()
                              execute_filter_sql()      →  SELECT w.sequence_id
                                                            FROM _filter_ws w
                                                            JOIN predictions p
                                                            WHERE p.value >= 50
                              save_filter_results()     →  INSERT INTO filter_results
                         ←   WorkingSet({surviving_ids})

get_final_data()         →  materialize_working_set() →   SELECT s.*, p.value AS tm
                                                            FROM sequences s
                                                            JOIN predictions p
                                                            WHERE s.sequence_id IN (…)
                         ←  pd.DataFrame
```

### Streaming Mode

For very large inputs (>100K sequences), `enable_streaming=True` switches to chunked processing:

```python
# base.py: _execute_stage_streaming()
async for chunk_df in stream_sequence_chunks(chunk_size=1000):
    chunk_ws = WorkingSet.from_ids(chunk_df["sequence_id"])
    chunk_ws_out, _ = await stage.process_ws(chunk_ws, datastore)
    # results stream to DuckDB as chunks complete
```

Streaming mode trades some latency (chunks must complete before the next stage sees them) for memory safety with massive datasets.

### Resume Logic

On `resume=True`:

```python
# base.py: run_async()
for level in levels:
    for stage in level:
        if resume:
            # Try to reload from DuckDB
            ws_reloaded = self._reload_stage_working_set(stage, ws_input)
            if ws_reloaded is not None:
                # Skip the stage — use the stored result
                ws_current = ws_reloaded
                continue
        # Not resumable → run stage normally
        ws_current, result = await self._execute_stage_ws(stage, ws_current)
```

For `PredictionStage`: reload = find all sequence_ids that have predictions for all resolved columns. For `FilterStage`: reload = read `filter_results` table. For `GenerationStage`: reload from generation metadata.

---

## 13. API Surface Design Decisions

### Decision 1: `extractions=` + `columns=` (Phase 8)

The original `prediction_type=` param conflated the cache key, the API response field, and the output column name — three distinct concerns.

The new design:
```python
pipeline.add_prediction("model",
    extractions="prediction",  # → what API response key (may differ per model)
    columns="tm",              # → what to call it in the output (user's name)
    # cache key auto-derived from model + action + extractions
)
```

This means renaming output columns never invalidates cache. Running the same model again with `columns="melting_point"` instead of `columns="tm"` hits the cache — no API call.

### Decision 2: `depends_on=[]` → Auto-Parallel

No dependency declaration means the stages are independent and run concurrently. This is the correct default — users who know their stages are independent shouldn't have to say so. Users who need ordering declare it explicitly.

### Decision 3: `datastore=` Required (No Magic Path)

```python
# data.py: DataPipeline.__init__()
if sequences is None and datastore is None:
    raise ValueError("Either sequences or datastore is required")
```

The pipeline does not silently create a `.biolm/` cache directory unless given explicit permission (via auto-default). This prevents accidental cache directories in production code. Users who want persistence pass `datastore="my.duckdb"` explicitly.

### Decision 4: `from_db()` Over `__init__(resume=True)`

Reconstruction is a class method, not a flag on `__init__`. This makes the intent explicit:

```python
# Recovery after kernel death
pipeline = DataPipeline.from_db("my_pipeline.duckdb")

# vs. normal run with resume
pipeline = DataPipeline(sequences=seqs, datastore="my.duckdb", resume=True)
```

Both support `resume=True` on `run()` — `from_db` simply reconstructs the stage graph from the database instead of requiring the user to re-declare it.

---

## File Index

| File | Role |
|------|------|
| [`biolmai/core/http.py`](biolmai/core/http.py) | `BioLMApiClient` — rate-limited async HTTP to biolm.ai |
| [`biolmai/pipeline/base.py`](biolmai/pipeline/base.py) | `BasePipeline`, `Stage`, `WorkingSet`, `InputSchema`, `PipelineContext`, `StageResult` |
| [`biolmai/pipeline/data.py`](biolmai/pipeline/data.py) | `DataPipeline`, `PredictionStage`, `FilterStage`, `ClusteringStage`, `ExtractionSpec`, `EmbeddingSpec` |
| [`biolmai/pipeline/generative.py`](biolmai/pipeline/generative.py) | `GenerativePipeline`, `GenerationStage`, `DirectGenerationConfig`, `RemaskingConfig` |
| [`biolmai/pipeline/datastore_duckdb.py`](biolmai/pipeline/datastore_duckdb.py) | `DuckDBDataStore` — all persistence, caching, dedup, materialization |
| [`biolmai/pipeline/filters.py`](biolmai/pipeline/filters.py) | `BaseFilter`, all filter classes, `to_sql()` for SQL fast path |
| [`biolmai/pipeline/pipeline_def.py`](biolmai/pipeline/pipeline_def.py) | `stage_from_spec()`, `filter_from_spec()`, `pipeline_from_definition()`, `_pipeline_def_hash()` |
| [`biolmai/pipeline/mlm_remasking.py`](biolmai/pipeline/mlm_remasking.py) | `MLMRemasker`, `RemaskingConfig` — iterative masked-LM variant generation |
| [`biolmai/pipeline/clustering.py`](biolmai/pipeline/clustering.py) | `SequenceClusterer`, `DiversityAnalyzer` |
| [`biolmai/pipeline/visualization.py`](biolmai/pipeline/visualization.py) | `PipelinePlotter` — funnel, PCA/UMAP, distributions |
| [`biolmai/pipeline/README.md`](biolmai/pipeline/README.md) | Module-level reference: all classes, all params |

---

*Last updated: 2026-03-18*
