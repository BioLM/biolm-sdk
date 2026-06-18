"""DSM diffusion generation + 3-run resumable pipeline.

Flow
----
Run 1  (dsm-150m-base generation → esmc-300m LP filter → ESMFold)
  ├── GenerationStage : DirectGenerationConfig(dsm-150m-base, 200 sequences)
  │                     DSM params: item_field='sequence', num_sequences, temperature
  ├── PredictionStage : esmc-300m log-probability scoring (action='score')
  ├── FilterStage     : LP > LP_STRICT (0.50)   ← strict filter
  └── PredictionStage : esmfold structure + pLDDT

Run 2  (same DB — dsm-650m-base, 100 sequences, same strict filter)
  ├── GenerationStage : DirectGenerationConfig(dsm-650m-base, 100 sequences)
  │                     Cache: LP already stored for run-1 sequences
  ├── PredictionStage : esmc-300m LP (new sequences computed; run-1 results cached)
  ├── FilterStage     : LP > LP_STRICT (same threshold)
  └── PredictionStage : esmfold (run-1 survivors already cached)

Run 3  (same DB — looser filter, fold any new survivors)
  Reads ALL sequences + LP scores from shared DB, applies LP > LP_LOOSE (0.30).
  Sequences with cached ESMFold predictions (from runs 1+2) are skipped by the
  cache check in PredictionStage; only newly passing sequences are folded.

DSM API params (from /schema/dsm-150m-base/generate/):
  item_field   : 'sequence'    ← plain amino-acid string (can be empty for unconditional)
  num_sequences: 1-32          ← sequences per call
  temperature  : 0.1-2.0
  step_divisor : 100           ← diffusion steps (lower = slower, better quality)
  remasking    : 'random' | 'low_confidence' | 'low_logit' | 'dual'

Why this works
--------------
All three runs share the same DuckDB file.  `add_sequences_batch()` uses hash-based
deduplication, so the same sequence always gets the same `sequence_id`.  Prediction
results (LP + pLDDT) are stored keyed by (sequence_id, prediction_type, model_name)
so cache hits are automatic across runs.

Usage
-----
    export BIOLMAI_TOKEN=<your_token>
    python scripts/pipeline_mlm_resume.py

Outputs (outputs/mlm_resume/):
    run1_results.csv, run2_results.csv, run3_results.csv
    pipeline.duckdb   ← shared, persists across all three runs
"""

import asyncio
import os
from pathlib import Path

from biolm.pipeline import (
    DirectGenerationConfig,
    DuckDBDataStore,
    GenerativePipeline,
    ThresholdFilter,
    ValidAminoAcidFilter,
)
from biolm.pipeline.data import DataPipeline, PredictionStage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    raise RuntimeError("Set BIOLMAI_TOKEN env var before running")

OUTPUT_DIR = Path("outputs/mlm_resume")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared DuckDB — all three runs read/write the same file
DB_PATH = OUTPUT_DIR / "pipeline.duckdb"

# Seed / conditioning sequence for DSM.
#   - Unconditional (de novo) generation: leave DSM_SEED_SEQUENCE empty.
#     The pipeline builds a fully-masked input of length DSM_MAX_LENGTH:
#       "<mask>" * DSM_MAX_LENGTH
#     DSM then fills all positions through diffusion.
#   - Conditional (sequence redesign): set DSM_SEED_SEQUENCE to an amino-acid
#     string.  You may also embed '<mask>' tokens in the string to fix some
#     positions and let DSM redesign only the masked ones.
SEED_SEQUENCE = os.environ.get("DSM_SEED_SEQUENCE", "")

# Length of sequences to generate unconditionally (ignored when SEED_SEQUENCE
# is provided — length is taken from the seed instead).
DSM_MAX_LENGTH = int(os.environ.get("DSM_MAX_LENGTH", "80"))

# Resolved input for the API: mask tokens for de novo, seed AA string otherwise.
_DSM_INPUT = SEED_SEQUENCE if SEED_SEQUENCE else "<mask>" * DSM_MAX_LENGTH

# LP scoring model — esmc-300m score action returns *total* log-probability
# (sum over all residues, so values are strongly negative).
# Observed range for 80-residue de-novo DSM sequences (esmc-300m):
#   MIN ≈ -220,  MAX ≈ -153,  AVG ≈ -191
# (values are LESS negative than the old per-residue estimate suggested)
# LP_STRICT keeps the top ~15% (score ≥ -180 ≈ well above average of -191).
# LP_LOOSE re-opens to the top ~45% for run 3 so ESMFold sees more candidates.
LP_MODEL = "esmc-300m"
LP_PRED_TYPE = "lp_score"
LP_STRICT = -180.0  # run 1 + 2: keep sequences with log_prob ≥ -180  (~top 15%)
LP_LOOSE = -190.0  # run 3: relax to log_prob ≥ -190                  (~top 45%)


# ---------------------------------------------------------------------------
# Shared helper: attach LP + ESMFold stages to any GenerativePipeline
# ---------------------------------------------------------------------------
_VALID_AA_FILTER = ValidAminoAcidFilter()


def _add_downstream_stages(
    pipeline: GenerativePipeline,
    lp_threshold: float,
) -> None:
    """Add seq validation → LP scoring → LP filter → ESMFold to a GenerativePipeline."""

    # Remove sequences with non-standard amino acids before LP scoring.
    # Some DSM models (e.g. dsm-650m-base) can output mask tokens or rare
    # residues that esmc-300m's score endpoint rejects.
    pipeline.add_filter(
        filter_func=_VALID_AA_FILTER,
        stage_name="seq_validation",
        depends_on=["generation"],
    )

    pipeline.add_prediction(
        model_name=LP_MODEL,
        action="score",
        extractions="log_prob",
        columns=LP_PRED_TYPE,
        stage_name="lp_scoring",
        depends_on=["seq_validation"],
        batch_size=64,
        max_concurrent=8,
    )

    pipeline.add_filter(
        filter_func=ThresholdFilter(
            column=LP_PRED_TYPE,
            min_value=lp_threshold,
        ),
        stage_name="lp_filter",
        depends_on=["lp_scoring"],
    )

    pipeline.add_prediction(
        model_name="esmfold",
        action="predict",
        extractions="mean_plddt",
        columns="plddt",
        stage_name="esmfold",
        depends_on=["lp_filter"],
        batch_size=4,
        max_concurrent=2,
        skip_on_error=True,  # don't abort pipeline on per-batch timeouts
    )


# ---------------------------------------------------------------------------
# Run 1: dsm-150m-base generation, strict LP filter
# DSM = Discrete (masked) Sequence Model; uses diffusion over the mask space.
# API params (from schema):
#   item_field   : 'sequence'  (empty string → unconditional)
#   num_sequences: 1-32 per call
#   temperature  : 0.1-2.0
#   step_divisor : diffusion steps divisor (default 100)
#   remasking    : strategy ('random', 'low_confidence', 'low_logit', 'dual')
# DSM max is 32 sequences per call, so we use 32 per call; GenerativePipeline
# fires one call per config so we set num_sequences=32 and generate a few configs.
# ---------------------------------------------------------------------------
async def run1(datastore: DuckDBDataStore) -> None:
    print("\n" + "=" * 60)
    print(f"RUN 1: dsm-150m-base generation  (LP threshold = {LP_STRICT})")
    print("=" * 60)

    # Fire multiple configs to generate ~200 sequences total (6 × 32 = 192).
    # For unconditional (de novo) generation, _DSM_INPUT is a fully-masked
    # string of length DSM_MAX_LENGTH.  DSM fills all positions via diffusion.
    configs = [
        DirectGenerationConfig(
            model_name="dsm-150m-base",
            sequence=_DSM_INPUT,
            item_field="sequence",
            params={
                "num_sequences": 32,
                "temperature": 1.0,
                "step_divisor": 100,
                "remasking": "random",
            },
        )
        for _ in range(6)  # 6 calls × 32 = 192 sequences
    ]

    pipeline = GenerativePipeline(
        generation_configs=configs,
        deduplicate=True,
        datastore=datastore,
        run_id="dsm_run1",
        output_dir=str(OUTPUT_DIR),
        verbose=True,
    )
    _add_downstream_stages(pipeline, lp_threshold=LP_STRICT)

    results = await pipeline.run_async()

    print("\nRun 1 stage results:")
    for _name, result in results.items():
        print(f"  {result}")

    df_final = pipeline.get_final_data()
    df_final.to_csv(OUTPUT_DIR / "run1_results.csv", index=False)
    print(
        f"\nRun 1 (dsm-150m-base): {len(df_final)} sequences passed → run1_results.csv"
    )


# ---------------------------------------------------------------------------
# Run 2: dsm-650m-base (larger model, same interface), same strict LP filter
# ---------------------------------------------------------------------------
async def run2(datastore: DuckDBDataStore) -> None:
    print("\n" + "=" * 60)
    print(f"RUN 2: dsm-650m-base generation  (LP threshold = {LP_STRICT})")
    print("=" * 60)

    configs = [
        DirectGenerationConfig(
            model_name="dsm-650m-base",
            sequence=_DSM_INPUT,
            item_field="sequence",
            params={
                "num_sequences": 32,
                "temperature": 0.8,  # slightly lower temp for more focused sampling
                "step_divisor": 100,
                "remasking": "low_confidence",
            },
        )
        for _ in range(4)  # 4 calls × 32 = 128 sequences
    ]

    pipeline = GenerativePipeline(
        generation_configs=configs,
        deduplicate=True,
        datastore=datastore,  # SAME datastore → cache shared with run 1
        run_id="dsm_run2",
        output_dir=str(OUTPUT_DIR),
        verbose=True,
    )
    _add_downstream_stages(pipeline, lp_threshold=LP_STRICT)

    results = await pipeline.run_async()

    print("\nRun 2 stage results:")
    for _name, result in results.items():
        print(f"  {result}")

    df_final = pipeline.get_final_data()
    df_final.to_csv(OUTPUT_DIR / "run2_results.csv", index=False)
    print(
        f"\nRun 2 (dsm-650m-base): {len(df_final)} sequences passed → run2_results.csv"
    )


# ---------------------------------------------------------------------------
# Run 3: looser filter — reprocess ALL sequences in DB
# ---------------------------------------------------------------------------
async def run3(datastore: DuckDBDataStore) -> None:
    print("\n" + "=" * 60)
    print(f"RUN 3: looser LP filter ({LP_STRICT} → {LP_LOOSE})  — reprocess full DB")
    print("=" * 60)

    # Export every sequence in the datastore together with its LP score.
    # export_to_dataframe() issues a single CASE WHEN pivot query — efficient
    # even for large datasets.
    df_all = datastore.export_to_dataframe(
        include_sequences=True,
        include_generation_metadata=False,
    )

    if LP_PRED_TYPE not in df_all.columns:
        print(f"  Column '{LP_PRED_TYPE}' not found in export; skipping run 3.")
        return

    df_all_with_lp = df_all.dropna(subset=[LP_PRED_TYPE])
    n_total = len(df_all_with_lp)

    # Apply the looser filter
    df_passing = df_all_with_lp[df_all_with_lp[LP_PRED_TYPE] > LP_LOOSE].copy()
    print(
        f"  {n_total} sequences have LP scores; "
        f"{len(df_passing)} pass the looser threshold ({LP_LOOSE})"
    )

    if df_passing.empty:
        print("  No sequences pass the looser filter.")
        return

    # DataPipeline: fold passing sequences.
    # PredictionStage performs a vectorized cache check (get_uncached_sequence_ids)
    # so sequences already folded in runs 1+2 are skipped automatically.
    pipeline = DataPipeline(
        sequences=df_passing,
        datastore=datastore,  # SAME datastore
        run_id="mlm_run3",
        output_dir=str(OUTPUT_DIR),
        verbose=True,
    )
    pipeline.add_stage(
        PredictionStage(
            name="esmfold",
            model_name="esmfold",
            action="predict",
            extractions="mean_plddt",
            columns="plddt",
            batch_size=4,
            max_concurrent=2,
            skip_on_error=True,  # don't abort on per-batch timeouts
        )
    )

    results = await pipeline.run_async()

    print("\nRun 3 stage results:")
    for _name, result in results.items():
        print(f"  {result}")

    df_final = pipeline.get_final_data()
    df_final.to_csv(OUTPUT_DIR / "run3_results.csv", index=False)

    # Summary
    print(f"\nRun 3: {len(df_final)} sequences folded/cached → run3_results.csv")
    if "plddt" in df_final.columns:
        high_q = (df_final["plddt"] >= 70).sum()
        print(f"  pLDDT ≥ 70 (high confidence): {high_q}")


# ---------------------------------------------------------------------------
# Main: run all three sequentially on the shared datastore
# ---------------------------------------------------------------------------
async def main() -> None:
    # Open one DuckDBDataStore instance; share it across all three runs.
    # hash-based deduplication in add_sequences_batch() guarantees that
    # the same amino-acid sequence always maps to the same sequence_id.
    datastore = DuckDBDataStore(DB_PATH)

    try:
        await run1(datastore)
        await run2(datastore)
        await run3(datastore)

        # Final cross-run summary
        print("\n" + "=" * 60)
        print("FINAL DATASTORE SUMMARY")
        print("=" * 60)
        conn = datastore.conn

        total_seq = conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
        total_folded = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction_type='plddt'"
        ).fetchone()[0]
        total_lp = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction_type=?",
            [LP_PRED_TYPE],
        ).fetchone()[0]
        print(f"  Total sequences in DB  : {total_seq}")
        print(f"  Sequences with LP score: {total_lp}")
        print(f"  Sequences folded (pLDDT): {total_folded}")
        print(f"\nDB location: {DB_PATH}")
    finally:
        datastore.close()


if __name__ == "__main__":
    asyncio.run(main())
