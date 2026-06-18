"""Antibody design pipeline: AntiFold → AntiFold score filter → IgBERT embeddings → ABodyBuilder3.

Flow
----
1. AntiFold (antifold): structure-conditioned generation of redesigned H+L
   antibody sequences at three temperatures (diversity scan).
   AntiFold params (confirmed from API schema):
     item_field  = 'pdb'
     heavy_chain : chain ID of the heavy chain in the PDB (e.g. 'H')
     light_chain : chain ID of the light chain in the PDB (e.g. 'L')
     num_seq_per_target : number of sequences to generate
     sampling_temp : sampling temperature (NOT 'temperature')
     regions : which regions to redesign (default CDR1+CDR2+CDR3)

2. AntiFold global_score filter via RankingFilter (top-N by score)
3. IgBERT paired embeddings (igbert-paired, action='encode')
4. ABodyBuilder3-pLDDT (abodybuilder3-plddt): 3-D structure + per-residue pLDDT
   Note: abodybuilder3-plddt takes {'H': heavy_seq, 'L': light_seq} items —
   the heavy_chain / light_chain columns populated by AntiFold are used here.

AntiFold response format (handled by GenerationStage._extract_sequences):
  [{"sequences": [{"heavy": "...", "light": "...", "score": ..., ...}]}]
  → stored as sequence = "heavy:light", with heavy_chain / light_chain columns.

Usage
-----
    export BIOLMAI_TOKEN=<your_token>
    # PDB must contain heavy chain 'H' and light chain 'L'
    export ANTIBODY_STRUCTURE_PATH=/path/to/antibody.pdb
    python scripts/pipeline_antibody_antifold.py

Outputs (in outputs/antibody_antifold/):
    antibody_designs.csv      – final DataFrame with scores/embeddings
    pipeline.duckdb           – DuckDB database (cache, resume)
"""

import asyncio
import os
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

# Path to an antibody PDB containing heavy (H) and light (L) chains.
_HERE = Path(__file__).parent.parent
STRUCTURE_PATH = os.environ.get(
    "ANTIBODY_STRUCTURE_PATH",
    str(_HERE / "tests/fixtures/antibody_HL.pdb"),
)

OUTPUT_DIR = Path("outputs/antibody_antifold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AntiFold chain IDs in the PDB (must match actual chain letters in the file)
HEAVY_CHAIN = os.environ.get("HEAVY_CHAIN", "H")
LIGHT_CHAIN = os.environ.get("LIGHT_CHAIN", "L")

# Sequences to generate per temperature
N_PER_TEMP = 150

# Final cut: keep only top N by AntiFold global_score
TOP_N = 10


# ---------------------------------------------------------------------------
# Generation configs — temperature diversity scan
# AntiFold API params (from /schema/antifold/generate/):
#   item_field         : 'pdb'
#   heavy_chain        : chain letter of H chain in PDB
#   light_chain        : chain letter of L chain in PDB
#   num_seq_per_target : sequences to generate
#   sampling_temp      : sampling temperature
#   regions            : defaults to ["CDR1", "CDR2", "CDR3"] — redesign CDRs only
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
                make_antifold_cfg(0.2),  # conservative — close to wild-type
                make_antifold_cfg(0.5),  # moderate diversity
                make_antifold_cfg(1.0),  # high diversity
            ],
            deduplicate=True,
            datastore=datastore,
            run_id="antibody_antifold_v1",
            output_dir=str(OUTPUT_DIR),
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Stage 1: Keep top N by AntiFold global_score
        #   global_score is a sequence-recovery / log-probability metric
        #   returned by AntiFold for each generated design.
        # ------------------------------------------------------------------
        pipeline.add_filter(
            filter_func=RankingFilter(
                column="global_score",
                n=TOP_N,
                ascending=True,  # AntiFold global_score: lower = better recovery
            ),
            stage_name="filter_top10",
            depends_on=["generation"],
        )

        # ------------------------------------------------------------------
        # Stage 2: IgBERT paired embeddings
        #   igbert accepts {'heavy': h_seq, 'light': l_seq} items for paired mode.
        #   AntiFold stores individual chains in 'heavy_chain' / 'light_chain'
        #   columns, so we map those to the 'heavy' / 'light' API fields.
        # ------------------------------------------------------------------
        pipeline.add_prediction(
            model_name="igbert-paired",
            action="predict",
            extractions="log_prob",
            columns="igbert_log_prob",
            stage_name="igbert",
            depends_on=["filter_top10"],
            batch_size=8,
            max_concurrent=4,
            item_columns={"heavy": "heavy_chain", "light": "light_chain"},
        )

        # ------------------------------------------------------------------
        # Stage 3: ABodyBuilder3-pLDDT — 3-D structure + per-residue pLDDT
        #   Runs in parallel with IgBERT (both depend on filter_top10 only).
        #   abodybuilder3-plddt takes {'H': heavy_seq, 'L': light_seq} per item
        #   and accepts exactly 1 item per API call (batch_size=1 is enforced
        #   server-side).  AntiFold populates 'heavy_chain' and 'light_chain'
        #   columns that we map to the 'H' and 'L' API fields via item_columns.
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
            params={"plddt": True},  # Request per-residue pLDDT in response
        )

        # ------------------------------------------------------------------
        # Run
        # ------------------------------------------------------------------
        print("\n=== Antibody AntiFold Pipeline ===")
        print(f"  Structure      : {STRUCTURE_PATH}")
        print(f"  Heavy/Light    : chains {HEAVY_CHAIN}/{LIGHT_CHAIN}")
        print(f"  Generated      : {3 * N_PER_TEMP} max (3 temps × {N_PER_TEMP})")
        print(f"  Final cut      : top {TOP_N} by AntiFold global_score")
        print()

        stage_results = await pipeline.run_async()

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n=== Stage Summary ===")
        for _name, result in stage_results.items():
            print(f"  {result}")

        df_final = pipeline.get_final_data()

        out_csv = OUTPUT_DIR / "antibody_designs.csv"
        df_final.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df_final)} final designs → {out_csv}")
    finally:
        datastore.close()


if __name__ == "__main__":
    asyncio.run(main())
