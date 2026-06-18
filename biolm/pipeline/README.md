# BioLM Pipeline

Multi-stage orchestration framework for protein and DNA sequence generation, prediction, and analysis. The pipeline handles batching, caching, deduplication, parallel execution, and resumability — so you describe *what* to compute, not *how* to call the API.

---

## Overview

The pipeline system provides a declarative interface over the BioLM API. Define a graph of generation, prediction, and filter stages; the pipeline resolves dependencies, runs independent stages concurrently via `asyncio.gather()`, caches all results in a local DuckDB database, and streams API requests with bounded concurrency. If a run is interrupted, `resume=True` skips any stages that already completed without re-calling the API.

---

## Installation

```bash
pip install biolm[pipeline]
```

```bash
export BIOLMAI_TOKEN="your-token-here"
# Get tokens at https://biolm.ai/ui/accounts/user-api-tokens/
```

---

## Quick Start

### Predict melting temperature and filter

```python
from biolm.pipeline import DataPipeline
from biolm.pipeline.filters import ThresholdFilter

pipeline = DataPipeline(sequences=[
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG",
    "MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAM",
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSY",
])

pipeline.add_prediction(
    "temberture-regression",
    extractions="prediction",   # API response key
    columns="tm",               # output column name
)

pipeline.add_filter(
    ThresholdFilter("tm", min_value=48.0),
    depends_on=["predict_tm"],
)

pipeline.run()
df = pipeline.get_final_data()
print(df[["sequence", "tm"]])
```

### Parallel predictions with ranking

Stages without mutual dependencies execute concurrently.

```python
from biolm.pipeline import DataPipeline
from biolm.pipeline.filters import RankingFilter

pipeline = DataPipeline(sequences=my_sequences)

# These two predictions run in parallel (depends_on=[] prevents auto-chaining)
pipeline.add_prediction(
    "temberture-regression",
    extractions="prediction",
    columns="tm",
    stage_name="predict_tm",
    depends_on=[],
)
pipeline.add_prediction(
    "biolmsol",
    extractions="solubility_score",
    columns="solubility",
    stage_name="predict_sol",
    depends_on=[],
)

# Rank after both complete
pipeline.add_filter(
    RankingFilter("solubility", n=10, ascending=False),
    depends_on=["predict_tm", "predict_sol"],
)

pipeline.run()
df = pipeline.get_final_data()
print(df[["sequence", "tm", "solubility"]])
```

### Generate sequences, then score

```python
from biolm.pipeline import GenerativePipeline, DirectGenerationConfig
from biolm.pipeline.filters import RankingFilter

config = DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_path="my_protein.pdb",   # supply your own PDB file
    params={"batch_size": 100, "temperature": 0.3},
)

pipeline = GenerativePipeline(generation_configs=[config])
pipeline.add_prediction(
    "temberture-regression",
    extractions="prediction",
    columns="tm",
)
pipeline.add_filter(
    RankingFilter("tm", n=20, ascending=False),
    depends_on=["predict_tm"],
)

pipeline.run()
df = pipeline.get_final_data()
```

---

## Extraction: Reading API Responses

Every prediction and embedding stage requires explicit extraction — you declare which key in the API response holds the value, and what to name the output column.

### Prediction extraction (`extractions=` and `columns=`)

```python
# Single key → single column
pipeline.add_prediction("temberture-regression",
    extractions="prediction",   # reads result["prediction"]
    columns="tm",               # output column in DataFrame
)

# Multiple keys from one API call
pipeline.add_prediction("esmfold",
    extractions=["mean_plddt", "ptm"],
    columns={"mean_plddt": "plddt"},  # rename; "ptm" keeps its name
)

# Array reduction (per-residue → scalar)
from biolm.pipeline.data import ExtractionSpec

pipeline.add_prediction("esmfold",
    extractions=[ExtractionSpec("mean_plddt", reduction="mean")],
    columns="plddt",
)
```

The cache key is derived automatically from `(model_name, action, sorted_response_keys)`. Renaming output columns with `columns=` does not invalidate the cache.

| `ExtractionSpec` field | Description |
|---|---|
| `response_key` | Key in the API response dict |
| `reduction` | For arrays: `"mean"`, `"max"`, `"min"`, `"sum"` |

### Known API response keys

| Model | Action | Response key | `extractions=` |
|---|---|---|---|
| `temberture-regression` | predict | `prediction` | `extractions="prediction"` |
| `biolmsol` | predict | `solubility_score` | `extractions="solubility_score"` |
| `esmc-300m` | score | `log_prob` | `extractions="log_prob"` |
| `esm2-8m` | encode | `embeddings` | `EmbeddingSpec(key="embeddings")` |
| `ablang2` | encode | `seqcoding` | `EmbeddingSpec(key="seqcoding")` |
| `dnabert2` | encode | `embedding` | `EmbeddingSpec(key="embedding")` |

### Embedding extraction (`embedding_extractor=`)

```python
from biolm.pipeline.data import EmbeddingSpec

# ESM2 — specific layer, mean-pooled
pipeline.add_prediction("esm2-650m", action="encode",
    embedding_extractor=EmbeddingSpec(key="embeddings", layer=33, reduction="mean"),
)

# AbLang2 — paired antibody
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
)

# DNABERT2
pipeline.add_prediction("dnabert2", action="encode",
    embedding_extractor=EmbeddingSpec(key="embedding"),
)

# Custom callable: (dict) -> list[tuple[np.ndarray, Optional[int]]]
pipeline.add_prediction("my-model", action="encode",
    embedding_extractor=lambda r: [(np.array(r["my_key"]), None)],
)
```

| `EmbeddingSpec` param | Description |
|---|---|
| `key` | API response dict key |
| `layer` | `None` = all layers, `int` = specific layer |
| `reduction` | `"mean"`, `"first"`, `"last"`, `"sum"` — collapses token-level to sequence-level |

---

## Multi-Column Input

For models that take multiple chains (antibody H+L, paired proteins), pass a DataFrame with `input_columns`:

```python
import pandas as pd
from biolm.pipeline import DataPipeline
from biolm.pipeline.data import EmbeddingSpec

df = pd.DataFrame({
    "heavy_chain": ["EVQLVESGGGLVQ...", "QVQLQESGPGLVK..."],
    "light_chain": ["DIQMTQSPSSLSA...", "SYELTQPPSVSSGA..."],
})

pipeline = DataPipeline(
    sequences=df,
    input_columns=["heavy_chain", "light_chain"],
)

# Predict Tm on heavy chain only
pipeline.add_prediction("temberture-regression",
    extractions="prediction",
    columns="tm_heavy",
    item_columns={"sequence": "heavy_chain"},
    stage_name="predict_tm_heavy",
)

# Predict Tm on light chain (runs in parallel)
pipeline.add_prediction("temberture-regression",
    extractions="prediction",
    columns="tm_light",
    item_columns={"sequence": "light_chain"},
    stage_name="predict_tm_light",
)

# Paired embeddings
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
    stage_name="embed",
)

pipeline.run()
df = pipeline.get_final_data()
# Columns: heavy_chain, light_chain, tm_heavy, tm_light
```

Key behaviors:
- Deduplication hashes across all input columns — same heavy + different light = distinct rows.
- Input columns are stored as real columns on the DuckDB `sequences` table, not as EAV attributes.
- `item_columns` maps `{"api_field": "df_column"}` — controls what gets sent to the API.

---

## Filters

```python
from biolm.pipeline.filters import (
    ThresholdFilter,          # Numeric min/max threshold
    RankingFilter,            # Top-N or bottom-N by column
    SequenceLengthFilter,     # Min/max sequence length
    ValidAminoAcidFilter,     # Remove non-standard residues
    HammingDistanceFilter,    # Distance to a reference sequence
    ConservedResidueFilter,   # Required residues at specific positions
    DiversitySamplingFilter,  # Subsample for sequence diversity
    CustomFilter,             # Any callable (DataFrame) -> DataFrame
    combine_filters,          # Chain multiple filters
)
```

| Filter | SQL-native | Description |
|---|---|---|
| `ThresholdFilter(col, min_value, max_value)` | Yes | Numeric threshold |
| `RankingFilter(col, n, ascending)` | Yes | Top-N / bottom-N by column value |
| `SequenceLengthFilter(min_length, max_length)` | Yes | Sequence length bounds |
| `ValidAminoAcidFilter(column)` | Yes | Remove sequences with non-standard residues |
| `HammingDistanceFilter(ref, max_distance)` | No | Distance to reference sequence |
| `ConservedResidueFilter(positions)` | No | Required residues at given positions |
| `DiversitySamplingFilter(n_samples, method)` | No | Subsample for diversity |
| `CustomFilter(func)` | No | Arbitrary callable |

SQL-native filters (`ThresholdFilter`, `RankingFilter`, `SequenceLengthFilter`, `ValidAminoAcidFilter`) execute directly in DuckDB with zero DataFrame materialization. On large datasets this means a single index scan instead of loading all rows into Python.

`RankingFilter` is scoped to the current working set — "top 20 by Tm" means top 20 among sequences that survived all upstream filters, not top 20 globally.

```python
# Example: combine filters
combined = combine_filters(
    ThresholdFilter("tm", min_value=50),
    SequenceLengthFilter(min_length=100),
)

# ValidAminoAcidFilter on a specific column
ValidAminoAcidFilter(column="heavy_chain")

# Top 20 by Tm
RankingFilter("tm", n=20, ascending=False)
```

Per-sequence filters (`ThresholdFilter`, `SequenceLengthFilter`, `HammingDistanceFilter`, `ConservedResidueFilter`, `CustomFilter`) can be applied in streaming mode as results arrive. Aggregate filters (`RankingFilter`, `DiversitySamplingFilter`) require the full working set.

---

## Generation Models

### DirectGenerationConfig

Structure-conditioned or sequence-conditioned generation:

```python
from biolm.pipeline import DirectGenerationConfig, GenerativePipeline

# Inverse folding — MPNN family
config = DirectGenerationConfig(
    model_name="protein-mpnn",    # also: hyper-mpnn, ligand-mpnn, soluble-mpnn
    item_field="pdb",
    structure_path="protein.pdb",
    params={"batch_size": 50, "temperature": 0.3},
)

# Temperature scanning — multiple configs run in parallel
configs = [
    DirectGenerationConfig(
        model_name="protein-mpnn",
        item_field="pdb",
        structure_path="protein.pdb",
        params={"batch_size": 20, "temperature": t},
    )
    for t in [0.1, 0.3, 0.5, 1.0]
]
pipeline = GenerativePipeline(generation_configs=configs, deduplicate=True)
```

```python
# DSM de novo generation
config = DirectGenerationConfig(
    model_name="dsm-650m-base",
    item_field="sequence",
    sequence="MKTAYIAK...",
    params={"num_sequences": 20, "temperature": 1.0},
)

# ZymCTRL — enzyme generation conditioned on EC number
config = DirectGenerationConfig(
    model_name="zymctrl",
    item_field="ec_number",
    sequence="3.1.1.101",
    params={"temperature": 1.0, "max_length": 300},
)

# ProGen2-OAS — antibody generation from VH seed
config = DirectGenerationConfig(
    model_name="progen2-oas",
    item_field="context",
    sequence="EVQLVES",
    params={"temperature": 1.0, "max_length": 120},
)
```

The `n_runs` parameter calls the same generation config multiple times concurrently:

```python
DirectGenerationConfig(
    model_name="zymctrl",
    item_field="ec_number",
    sequence="3.1.1.101",
    params={"temperature": 1.0, "max_length": 300},
    n_runs=5,   # 5 parallel API calls
)
```

### Structure from upstream stage

Pass structures predicted by an earlier pipeline stage (e.g. ESMFold → MPNN):

```python
config = DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_from_model="esmfold",   # reads from DuckDB structures table
    params={"batch_size": 50, "temperature": 0.3},
)
```

### MLM Remasking

Iterative masked-LM refinement using ESM2 or DSM. Positions are masked, sent to the API, and filled by sampling from the returned logits (ESM2) or directly (DSM):

```python
from biolm.pipeline.mlm_remasking import RemaskingConfig

config = RemaskingConfig(
    model_name="esm2-650m",
    action="predict",               # "predict" for ESM2, "generate" for DSM
    mask_fraction=0.15,             # mask 15% per round
    num_iterations=3,               # 3 rounds of mask-predict-replace
    temperature=1.0,
    conserved_positions=[107, 109], # never mask active-site residues
    top_k=5,                        # optional top-k sampling
    top_p=0.95,                     # optional nucleus sampling
)
config.parent_sequence = "SNPYARGPNPTAASLEASAG..."
config.num_variants = 20

pipeline = GenerativePipeline(generation_configs=[config])
pipeline.run()
```

How the process works:
1. Select positions to mask (random, skipping `conserved_positions`).
2. Insert `<mask>` tokens at those positions and call the API.
3. For ESM2: decode logits with temperature/top-k/top-p sampling. For DSM: receive filled sequences directly.
4. Repeat for `num_iterations` rounds, re-masking different positions each time.
5. Return each variant with mutation metadata (`num_mutations`, `mutation_rate`).

### Co-folding (Boltz2, Chai-1)

Multi-molecule structure prediction where each pipeline sequence becomes one chain and `static_entities` adds ligands, cofactors, DNA/RNA, or extra protein chains:

```python
from biolm.pipeline import DataPipeline, FoldingEntity

pipeline = DataPipeline(sequences=protein_sequences)

pipeline.add_cofolding_prediction(
    model_name="boltz2",
    prediction_type="structure",
    sequence_chain_id="A",
    sequence_entity_type="protein",
    static_entities=[
        FoldingEntity(id="L", entity_type="ligand", smiles="c1ccccc1"),
        FoldingEntity(id="B", entity_type="protein", sequence="MKTAYIAK..."),
        FoldingEntity(id="D", entity_type="dna", sequence="ATGCGATCG"),
        FoldingEntity(id="X", entity_type="ligand", ccd="ATP"),
    ],
    params={"recycling_steps": 3, "sampling_steps": 20},
    batch_size=1,
)
```

| `FoldingEntity` field | Type | Description |
|---|---|---|
| `id` | `str` | Chain / molecule identifier |
| `entity_type` | `str` | `"protein"`, `"dna"`, `"rna"`, or `"ligand"` |
| `sequence` | `Optional[str]` | AA or nucleotide sequence |
| `smiles` | `Optional[str]` | SMILES string (for ligands) |
| `ccd` | `Optional[str]` | CCD code, e.g. `"ATP"`, `"HEM"` |

Structures and confidence scores are stored in the DuckDB `structures` table.

---

## Convenience Functions

One-liner alternatives to building a full pipeline object.

### `Embed()`

```python
from biolm.pipeline import Embed

# Returns a DataFrame with columns: sequence, sequence_id, embedding
df = Embed("esm2-8m", sequences=my_sequences)

# Specific layer
df = Embed("esm2-650m", sequences=my_sequences, layer=33)
```

Equivalent to creating a `SingleStepPipeline` with `action="encode"` and calling `get_embeddings_bulk()` — but in one call.

### `Predict()`

```python
from biolm.pipeline import Predict

df = Predict("temberture-regression", sequences=my_sequences, extractions="prediction")
```

---

## Context Manager

`DataPipeline` and `GenerativePipeline` implement `__enter__`/`__exit__` for automatic DuckDB cleanup:

```python
ds = DuckDBDataStore("project.duckdb")

with DataPipeline(sequences=my_sequences, datastore=ds) as pipeline:
    pipeline.add_prediction("temberture-regression",
        extractions="prediction", columns="tm")
    pipeline.run()
    df = pipeline.get_final_data()
# ds is closed automatically on exit — even if an exception is raised

# Async variant
async with DataPipeline(sequences=my_sequences, datastore=ds) as pipeline:
    ...
```

---

## Caching and Resume

All predictions are cached in DuckDB. Re-running the same sequences against the same model hits the cache — no API call is made.

```python
# Resume a failed or interrupted run
pipeline = DataPipeline(
    sequences=my_sequences,
    run_id="my_analysis_v1",
    resume=True,
)

# Trickle new sequences into an existing cache
pipeline = DataPipeline(
    sequences=old_sequences + new_sequences,
    datastore="project.duckdb",
)
# Cached sequences skip the API; only new ones are sent
```

To reconstruct and resume a pipeline after a kernel death:

```python
pipeline = DataPipeline.from_db("my_pipeline.duckdb")
pipeline.run(resume=True)
```

`from_db()` reads the stage graph from the `pipeline_definitions` table. Any `CustomFilter` or custom `embedding_extractor` callable must be re-attached manually after reconstruction.

### GenerativePipeline resume behavior

`GenerativePipeline` always re-runs the generation stage on resume — generation is stochastic and each run produces a fresh set of sequences. Downstream prediction and filter stages still benefit from the prediction cache: any sequence already scored in a previous run is not sent to the API again.

---

## Architecture

Between stages the pipeline passes a `WorkingSet` — a `frozenset[int]` of sequence IDs — rather than a DataFrame. All data lives in DuckDB. A DataFrame is materialized only at `get_final_data()` time via a single SQL JOIN. `1M sequence IDs ≈ 28 MB (frozenset[int]) vs 500 MB+ DataFrame`.

Independent stages run concurrently via `asyncio.gather()` (Kahn's algorithm topological sort). Declaring `depends_on=[]` (the default) places a stage at the earliest possible level.

```
→ Level 0: [predict_tm, predict_sol]  — asyncio.gather(), concurrent
→ Level 1: [ranking_filter]           — sequential, after both complete
```

Within each `PredictionStage`: `batch_size` (default 32) sequences per pipeline batch, `max_concurrent` (default 5) batches in flight, `max_connections` (default 10) HTTP connections. Raise `batch_size` and `max_concurrent` for fast scoring models; lower them for slow structure models.

For the full architecture deep-dive — DuckDB schema, performance patterns, design decisions, execution model — see [`PIPELINE_VISION_AND_ARCHITECTURE.md`](PIPELINE_VISION_AND_ARCHITECTURE.md).

---

## Pipeline Context

Share data between stages, backed by the DuckDB `pipeline_context` table and scoped per `run_id`:

```python
pipeline = DataPipeline(sequences=my_sequences)

# Write
pipeline.context.set("experiment", "thermostability_screen")
pipeline.context.set("config", {"target_tm": 65, "n_variants": 200})

# Read
pipeline.context.get("experiment")         # → "thermostability_screen"
pipeline.context.get("missing", "default") # → "default"

# Access predicted structures from upstream stages
struct = pipeline.context.get_structure(seq_id, "esmfold")
structs = pipeline.context.get_structures_for_ws(working_set, "esmfold")
```

Context values are JSON-serialized. Any JSON-serializable Python object can be stored.

---

## Data Exploration

After `pipeline.run()`:

```python
# Aggregate statistics
pipeline.explore()    # → dict: sequence counts, prediction stats, stage breakdown
pipeline.stats()      # → DataFrame: per-stage timing and counts
pipeline.summary()    # → DataFrame: stage input/output counts

# SQL query over the DuckDB database
pipeline.query("SELECT sequence, tm FROM sequences JOIN predictions ...")

# Visualization (requires matplotlib)
pipeline.plot(kind="funnel")         # Sequences in vs out per stage
pipeline.plot(kind="predictions")    # Prediction value distributions
pipeline.plot(kind="distributions")  # Cross-stage property distributions
```

---

## DuckDB DataStore — Direct Usage

The datastore can also be used independently:

```python
from biolm.pipeline import DuckDBDataStore

store = DuckDBDataStore("pipeline.duckdb", "data/")

# Add sequences (batch, deduplicated by hash)
seq_ids = store.add_sequences_batch(["MKLLIV", "ACDEFG"])

# Add a prediction
store.add_prediction(seq_id, "tm", "temberture-regression", 48.6)

# Bulk add predictions
store.add_predictions_bulk(seq_ids, pred_type="tm", model="temberture-regression",
                           values=[48.6, 52.1])

# Check cache before calling API
uncached = store.get_uncached_sequence_ids(seq_ids, cache_key, model_name)

# Query with SQL
df = store.query("SELECT * FROM predictions WHERE value > 60")

# Export
df = store.export_to_dataframe(include_predictions=True)
store.export_to_csv("results.csv")

# Context manager
with DuckDBDataStore("pipeline.duckdb") as store:
    ...
```

---

## Full Example: Multi-Model PETase Design

Generate candidates from three sources, score with three models, filter to the top 20 by solubility.

```python
from biolm.pipeline import GenerativePipeline, DirectGenerationConfig, DataPipeline
from biolm.pipeline.data import EmbeddingSpec
from biolm.pipeline.filters import (
    ValidAminoAcidFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    RankingFilter,
)

LCC = "SNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGG..."

# Step 1: Generate from multiple sources
configs = [
    DirectGenerationConfig(
        model_name="protein-mpnn",
        item_field="pdb",
        structure_path="lcc.pdb",
        params={"batch_size": 50, "temperature": 0.3},
    ),
    DirectGenerationConfig(
        model_name="zymctrl",
        item_field="ec_number",
        sequence="3.1.1.101",
        params={"temperature": 1.0, "max_length": 300},
    ),
    DirectGenerationConfig(
        model_name="dsm-650m-base",
        item_field="sequence",
        sequence=LCC,
        params={"num_sequences": 20, "temperature": 1.0},
    ),
]

gen_pipeline = GenerativePipeline(generation_configs=configs, deduplicate=True)
gen_pipeline.run()
generated = gen_pipeline.get_final_data()

# Step 2: Score all variants
pipeline = DataPipeline(sequences=[LCC] + generated["sequence"].tolist())

pipeline.add_filter(ValidAminoAcidFilter(), stage_name="valid_aa")
pipeline.add_filter(
    SequenceLengthFilter(min_length=100, max_length=500),
    stage_name="length_filter",
    depends_on=["valid_aa"],
)

# Three predictions run in parallel (explicit depends_on avoids auto-chaining)
pipeline.add_prediction(
    "temberture-regression",
    extractions="prediction",
    columns="tm",
    depends_on=["length_filter"],
)
pipeline.add_prediction(
    "biolmsol",
    extractions="solubility_score",
    columns="solubility",
    depends_on=["length_filter"],
)
pipeline.add_prediction(
    "esmc-300m",
    action="score",
    extractions="log_prob",
    depends_on=["length_filter"],
)

pipeline.add_filter(
    ThresholdFilter("tm", min_value=45),
    stage_name="tm_filter",
    depends_on=["predict_tm"],
)
pipeline.add_filter(
    RankingFilter("solubility", n=20, ascending=False),
    stage_name="top_solubility",
    depends_on=["tm_filter", "predict_solubility", "predict_log_prob"],
)

pipeline.run()
df = pipeline.get_final_data()
print(df[["sequence", "tm", "solubility", "log_prob"]].to_string())
```
