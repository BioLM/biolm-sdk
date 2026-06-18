"""Multi-MPNN parallel generation with temperature scanning → Tm + solubility.

Flow
----
Generation (all 15 configs fired in parallel):
  ┌─ protein-mpnn              @ T=[0.1, 0.3, 0.5]  ──┐
  ├─ hyper-mpnn                @ T=[0.1, 0.3, 0.5]  ──┤
  ├─ global-label-membrane-mpnn@ T=[0.1, 0.3, 0.5]  ──┼─► 15-model fan-out → dedup
  ├─ soluble-mpnn              @ T=[0.1, 0.3, 0.5]  ──┤
  └─ ligand-mpnn               @ T=[0.1, 0.3, 0.5]  ──┘
          │
          ▼
  Downstream (parallel):
  ├─ temberture-regression  (thermal stability, Tm in °C)
  └─ biolmsol               (solubility probability 0-1)
          │
          ▼
  Filter: Tm > 50 °C  AND  solubility > 0.5
          │
          ▼
  Rank: top 100 by Tm (final design set)

Usage
-----
    export BIOLMAI_TOKEN=<your_token>
    export MPNN_STRUCTURE_PATH=/path/to/protein.pdb
    python scripts/pipeline_mpnn_multi.py

Outputs (outputs/mpnn_multi/):
    mpnn_designs.csv     – final ranked DataFrame
    pipeline.duckdb      – DuckDB cache (resumable)
"""

import asyncio
import os
from pathlib import Path

from biolm.pipeline import (
    DirectGenerationConfig,
    DuckDBDataStore,
    GenerativePipeline,
    RankingFilter,
    ThresholdFilter,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    raise RuntimeError("Set BIOLMAI_TOKEN env var before running")

STRUCTURE_PATH = os.environ.get("MPNN_STRUCTURE_PATH", "examples/protein.pdb")

OUTPUT_DIR = Path("outputs/mpnn_multi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sequences per (model, temperature) combination
N_PER_CONFIG = 100

# Downstream filter thresholds
TM_MIN = 50.0  # temberture-regression output is Tm in °C
SOLUBILITY_MIN = 0.5  # biolmsol output is 0-1 probability

# Final ranking: keep top N by Tm
TOP_N_FINAL = 100

# Models and temperatures to fan out across
# All MPNN slugs use hyphens (confirmed from biolm.ai/models catalogue)
MPNN_MODELS = [
    "protein-mpnn",
    "hyper-mpnn",
    "global-label-membrane-mpnn",  # global transmembrane label variant
    "soluble-mpnn",
    "ligand-mpnn",
]
TEMPERATURES = [0.1, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Build generation configs (5 models × 3 temperatures = 15 configs)
# ---------------------------------------------------------------------------
def build_generation_configs() -> list[DirectGenerationConfig]:
    configs = []
    for model in MPNN_MODELS:
        for temp in TEMPERATURES:
            configs.append(
                DirectGenerationConfig(
                    model_name=model,
                    structure_path=STRUCTURE_PATH,
                    item_field="pdb",  # all MPNN variants take 'pdb' items
                    params={
                        "batch_size": N_PER_CONFIG,  # MPNN uses 'batch_size', not 'num_sequences'
                        "temperature": temp,
                    },
                )
            )
    return configs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def main() -> None:
    db_path = OUTPUT_DIR / "pipeline.duckdb"
    datastore = DuckDBDataStore(db_path)

    try:
        generation_configs = build_generation_configs()

        print("\n=== Multi-MPNN Pipeline ===")
        print(f"  Structure      : {STRUCTURE_PATH}")
        print(f"  Models         : {', '.join(MPNN_MODELS)}")
        print(f"  Temperatures   : {TEMPERATURES}")
        print(f"  Total configs  : {len(generation_configs)}")
        print(f"  Max sequences  : {len(generation_configs) * N_PER_CONFIG} (before dedup)")
        print()

        pipeline = GenerativePipeline(
            generation_configs=generation_configs,
            deduplicate=True,
            datastore=datastore,
            run_id="mpnn_multi_v1",
            output_dir=str(OUTPUT_DIR),
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Stage 1a: temberture-regression — melting temperature prediction
        #   Predicts Tm (°C). Higher Tm → more thermally stable.
        # ------------------------------------------------------------------
        pipeline.add_prediction(
            model_name="temberture-regression",
            action="predict",
            extractions="prediction",
            columns="tm",
            stage_name="temberture",
            depends_on=["generation"],
            batch_size=32,
            max_concurrent=5,
        )

        # ------------------------------------------------------------------
        # Stage 1b: biolmsol — solubility prediction (parallel with temberture)
        #   Predicts solubility probability in [0, 1].
        # ------------------------------------------------------------------
        pipeline.add_prediction(
            model_name="biolmsol",
            action="predict",
            extractions="solubility_score",
            columns="solubility",
            stage_name="biolmsol",
            depends_on=["generation"],
            batch_size=32,
            max_concurrent=5,
        )

        # ------------------------------------------------------------------
        # Stage 2: Filter — Tm AND solubility threshold
        #   Both prediction stages must complete first; either dependency
        #   triggers this filter (both are satisfied once both run).
        # ------------------------------------------------------------------
        pipeline.add_filter(
            filter_func=ThresholdFilter(
                column="tm",
                min_value=TM_MIN,
            ),
            stage_name="filter_tm",
            depends_on=["temberture", "biolmsol"],  # wait for both
        )

        pipeline.add_filter(
            filter_func=ThresholdFilter(
                column="solubility",
                min_value=SOLUBILITY_MIN,
            ),
            stage_name="filter_sol",
            depends_on=["filter_tm"],
        )

        # ------------------------------------------------------------------
        # Stage 3: Rank — keep top 100 by Tm
        # ------------------------------------------------------------------
        pipeline.add_filter(
            filter_func=RankingFilter(
                column="tm",
                n=TOP_N_FINAL,
                ascending=False,  # highest Tm first
            ),
            stage_name="rank_top100",
            depends_on=["filter_sol"],
        )

        # ------------------------------------------------------------------
        # Run
        # ------------------------------------------------------------------
        stage_results = await pipeline.run_async()

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n=== Stage Summary ===")
        for _name, result in stage_results.items():
            print(f"  {result}")

        df_final = pipeline.get_final_data()
        out_csv = OUTPUT_DIR / "mpnn_designs.csv"
        df_final.to_csv(out_csv, index=False)
        print(f"\nTop {len(df_final)} designs saved → {out_csv}")

        if len(df_final) > 0:
            cols = [
                c
                for c in ["sequence", "model_name", "temperature", "tm", "solubility"]
                if c in df_final.columns
            ]
            best = df_final.nlargest(5, "tm")[cols]
            print("\nTop 5 by Tm:")
            print(best.to_string(index=False))
    finally:
        datastore.close()


if __name__ == "__main__":
    asyncio.run(main())
