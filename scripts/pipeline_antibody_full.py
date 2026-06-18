"""Full antibody design pipeline: AntiFold → IgBERT scoring → ESM-C scoring → ABodyBuilder3.

Flow
----
1. AntiFold (generate): PDB → redesigned H+L chain sequences at two temperatures
2. RankingFilter: top 20 by AntiFold global_score
3. IgBERT Paired (predict): log-probability scoring of paired H+L chains
4. ESM-C 300M (score): log-probability scoring of heavy chain separately
5. ESM-C 300M (score): log-probability scoring of light chain separately
6. RankingFilter: top 10 by IgBERT log_prob
7. ABodyBuilder3-pLDDT (predict): 3-D structure + per-residue pLDDT for top 10

API actions used (only 4 public actions exist: score, predict, generate, encode):
  - antifold/generate      → {sequences: [{heavy, light, global_score, ...}]}
  - igbert-paired/predict   → {log_prob: float}
  - esmc-300m/score         → {log_prob: float}
  - abodybuilder3-plddt/predict → {pdb: str, plddt: [[...]]}

Usage
-----
    export BIOLMAI_TOKEN=<your_token>
    python scripts/pipeline_antibody_full.py
"""

import asyncio
import os
import traceback
from pathlib import Path

from biolm.pipeline import (
    DirectGenerationConfig,
    DuckDBDataStore,
    GenerativePipeline,
    RankingFilter,
)
from biolm.pipeline.data import ExtractionSpec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    raise RuntimeError("Set BIOLMAI_TOKEN env var before running")

# Default: bundled test fixture (Fab with H + L chains from PDB 2FJG)
_HERE = Path(__file__).parent.parent
STRUCTURE_PATH = os.environ.get(
    "ANTIBODY_STRUCTURE_PATH",
    str(_HERE / "tests/fixtures/antibody_HL.pdb"),
)

OUTPUT_DIR = Path("outputs/antibody_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEAVY_CHAIN = os.environ.get("HEAVY_CHAIN", "H")
LIGHT_CHAIN = os.environ.get("LIGHT_CHAIN", "L")

# Generation: sequences per temperature
N_PER_TEMP = 10

# ---------------------------------------------------------------------------
# AntiFold generation configs — temperature diversity scan
# ---------------------------------------------------------------------------
def make_antifold_cfg(sampling_temp: float) -> DirectGenerationConfig:
    return DirectGenerationConfig(
        model_name="antifold",
        structure_path=STRUCTURE_PATH,
        item_field="pdb",
        params={
            "heavy_chain": HEAVY_CHAIN,
            "light_chain": LIGHT_CHAIN,
            "num_seq_per_target": N_PER_TEMP,
            "sampling_temp": sampling_temp,
        },
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def main() -> None:
    db_path = OUTPUT_DIR / "pipeline.duckdb"
    datastore = DuckDBDataStore(db_path)

    try:
        pipeline = GenerativePipeline(
            generation_configs=[
                make_antifold_cfg(0.2),  # conservative
                make_antifold_cfg(0.5),  # moderate diversity
            ],
            deduplicate=True,
            datastore=datastore,
            run_id="antibody_full_v1",
            output_dir=str(OUTPUT_DIR),
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Stage 1: Keep top 20 by AntiFold global_score (lower = better)
        # ------------------------------------------------------------------
        pipeline.add_filter(
            filter_func=RankingFilter(
                column="global_score",
                n=20,
                ascending=True,
            ),
            stage_name="filter_top20_antifold",
            depends_on=["generation"],
        )

        # ------------------------------------------------------------------
        # Stage 2: IgBERT paired scoring (log-probability of paired H+L)
        #   igbert-paired/predict → {log_prob: float}
        #   item_columns maps API fields to DataFrame columns from AntiFold
        # ------------------------------------------------------------------
        pipeline.add_prediction(
            model_name="igbert-paired",
            action="predict",
            extractions="log_prob",
            columns="igbert_log_prob",
            stage_name="igbert_score",
            depends_on=["filter_top20_antifold"],
            batch_size=8,
            max_concurrent=4,
            item_columns={"heavy": "heavy_chain", "light": "light_chain"},
        )

        # ------------------------------------------------------------------
        # Stage 3a: ESM-C scoring of heavy chain
        #   esmc-300m/score → {log_prob: float}
        # ------------------------------------------------------------------
        pipeline.add_prediction(
            model_name="esmc-300m",
            action="score",
            extractions="log_prob",
            columns="esmc_heavy_log_prob",
            stage_name="esmc_heavy",
            depends_on=["filter_top20_antifold"],
            batch_size=8,
            max_concurrent=4,
            item_columns={"sequence": "heavy_chain"},
        )

        # ------------------------------------------------------------------
        # Stage 3b: ESM-C scoring of light chain
        #   esmc-300m/score → {log_prob: float}
        # ------------------------------------------------------------------
        pipeline.add_prediction(
            model_name="esmc-300m",
            action="score",
            extractions="log_prob",
            columns="esmc_light_log_prob",
            stage_name="esmc_light",
            depends_on=["filter_top20_antifold"],
            batch_size=8,
            max_concurrent=4,
            item_columns={"sequence": "light_chain"},
        )

        # ------------------------------------------------------------------
        # Stage 4: Filter top 10 by IgBERT log_prob (higher = better)
        # ------------------------------------------------------------------
        pipeline.add_filter(
            filter_func=RankingFilter(
                column="igbert_log_prob",
                n=10,
                ascending=False,  # higher log_prob = better
            ),
            stage_name="filter_top10",
            depends_on=["igbert_score", "esmc_heavy", "esmc_light"],
        )

        # ------------------------------------------------------------------
        # Stage 5: ABodyBuilder3 structure prediction for top 10
        #   abodybuilder3-plddt/predict → {pdb: str, plddt: [[...]]}
        # ------------------------------------------------------------------
        pipeline.add_prediction(
            model_name="abodybuilder3-plddt",
            action="predict",
            extractions=[ExtractionSpec("plddt", reduction="mean")],
            columns={"plddt": "abb3_plddt"},
            stage_name="abodybuilder3",
            depends_on=["filter_top10"],
            batch_size=1,
            max_concurrent=2,
            item_columns={"H": "heavy_chain", "L": "light_chain"},
            params={"plddt": True},
        )

        # ------------------------------------------------------------------
        # Run
        # ------------------------------------------------------------------
        print("\n=== Full Antibody Pipeline ===")
        print(f"  Structure      : {STRUCTURE_PATH}")
        print(f"  Heavy/Light    : chains {HEAVY_CHAIN}/{LIGHT_CHAIN}")
        print(f"  Generated      : {2 * N_PER_TEMP} max (2 temps × {N_PER_TEMP})")
        print(f"  Pipeline       : AntiFold → top20 → IgBERT + ESM-C → top10 → ABodyBuilder3")
        print()

        stage_results = await pipeline.run_async()

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n=== Stage Summary ===")
        for _name, result in stage_results.items():
            print(f"  {result}")

        df_final = pipeline.get_final_data()

        # Show top 10 results
        score_cols = [c for c in ["heavy_chain", "light_chain", "global_score",
                                   "igbert_log_prob", "esmc_heavy_log_prob",
                                   "esmc_light_log_prob", "abb3_plddt"] if c in df_final.columns]
        if not df_final.empty:
            print(f"\nTop {len(df_final)} designs:")
            # Truncate long chain sequences for display
            display_df = df_final[score_cols].copy()
            for col in ["heavy_chain", "light_chain"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].str[:20] + "..."
            print(display_df.to_string(index=True))

        out_csv = OUTPUT_DIR / "antibody_designs.csv"
        df_final.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df_final)} final designs → {out_csv}")
    finally:
        datastore.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        traceback.print_exc()
