"""Explore-after-run demo: multi-model predictions, filtering, SQL queries, resume, and plots.

Demonstrates the core value proposition of BioLM pipelines: run once, explore many times.

Flow:
  1. 30 antimicrobial peptides
  2. Parallel predictions: temberture-regression (Tm) + biolmsol (solubility)
  3. ThresholdFilter on Tm, RankingFilter on solubility
  4. summary(), explore(), stats()
  5. SQL queries via query()
  6. Reopen from_db() — zero API calls on cached data
  7. plot("funnel"), plot("distributions")
  8. Export CSV

Usage:
    export BIOLMAI_TOKEN=<your_token>
    python scripts/demo_explore_pipeline.py
"""

import asyncio
import os
import sys
from pathlib import Path

from biolm.pipeline import (
    DataPipeline,
    DuckDBDataStore,
    RankingFilter,
    ThresholdFilter,
    ValidAminoAcidFilter,
)

TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    print("ERROR: BIOLMAI_TOKEN not set.")
    sys.exit(1)

OUTPUT_DIR = Path("outputs/explore_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 30 antimicrobial peptides — diverse lengths, charges, hydrophobicities
PEPTIDES = [
    "GIGKFLHSAKKFGKAFVGEIMNS",        # Magainin 2
    "GIGKFLHSAGKFGKAFVGEIMKS",        # Magainin 1 analog
    "GLFDIIKKIAESF",                   # Aurein 1.2
    "GLFDIVKKVVGALGSL",               # Aurein 2.2
    "FLPLILRKIVTAL",                   # Citropin 1.1
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # LL-37
    "RLFDKIRQVIRKF",                   # Indolicidin analog
    "KWKLFKKIPKFLHLAKKF",             # Cecropin-melittin hybrid
    "ACYCRIPACIAGERRYGTCIYQGRLWAFCC", # HBD-1 analog
    "DHYNCVSSGGQCLYSACPIFTKIQGTCYRGKAKCCK",  # HNP-1 analog
    "RWKIFKKIEKVGRNVRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin B
    "GIGAVLKVLTTGLPALISWIKRKRQQ",     # Melittin
    "VDKGSYLPRPTPPRPIYNRN",           # Apidaecin
    "GKPRPYSPRPTSHPRPIRV",            # Drosocin
    "KLAKLAKKLAKLAK",                  # (KLA)3
    "LKLLKKLLKLLKKL",                  # Designed AMP
    "RRWWRRWWRR",                      # Synthetic arginine-rich
    "KWKWKWKWKW",                      # KW repeat
    "GIKKFLGSIWKFIKAFVKEIMN",         # MSI-78 (pexiganan)
    "CKVWGKLCRTRGCTTTHCRRH",          # Protegrin analog
    "RRLCRIVVIRVCR",                   # Tachyplesin analog
    "GFCWYVNAAAHCGKRFNRVCYRN",       # Plectasin analog
    "RRWQWR",                          # Lactoferricin fragment
    "RWRWRW",                          # RW repeat
    "FKRIVQRIKDFL",                    # LL-37 fragment
    "KFLKKAKKFGK",                     # Magainin fragment
    "GIGKFLHSAK",                      # Magainin N-term
    "KWKLFKKI",                        # Short cecropin
    "RLFDKIRQ",                        # Indolicidin short
]


async def main():
    db_path = OUTPUT_DIR / "pipeline.duckdb"
    ds = DuckDBDataStore(db_path)

    try:
        # ── Run 1: predictions + filtering ──────────────────────────────
        pipeline = DataPipeline(
            sequences=PEPTIDES,
            datastore=ds,
            run_id="explore_demo_v1",
            output_dir=str(OUTPUT_DIR),
            verbose=True,
        )

        # Validate amino acids
        pipeline.add_filter(ValidAminoAcidFilter(verbose=True), stage_name="validate")

        # Parallel predictions: Tm and solubility
        pipeline.add_prediction(
            "temberture-regression",
            action="predict",
            extractions="prediction",
            columns="melting_temperature",
            stage_name="predict_tm",
            depends_on=["validate"],
            batch_size=16,
        )
        pipeline.add_prediction(
            "biolmsol",
            action="predict",
            extractions="solubility_score",
            columns="solubility",
            stage_name="predict_sol",
            depends_on=["validate"],
            batch_size=16,
        )

        # Filters (sequential after predictions)
        pipeline.add_filter(
            ThresholdFilter(
                column="melting_temperature",
                min_value=40.0,
            ),
            stage_name="filter_tm",
            depends_on=["predict_tm", "predict_sol"],
        )
        pipeline.add_filter(
            RankingFilter(
                column="solubility",
                n=15,
                ascending=False,
            ),
            stage_name="filter_sol_top15",
        )

        print("\n" + "=" * 60)
        print("  EXPLORE DEMO: Multi-Model Predict → Filter → Explore")
        print("=" * 60)

        stage_results = await pipeline.run_async()

        # ── Explore the results ─────────────────────────────────────────
        print("\n\n" + "=" * 60)
        print("  EXPLORING RESULTS")
        print("=" * 60)

        # 1. Summary table
        print("\n── summary() ──")
        print(pipeline.summary().to_string(index=False))

        # 2. Explore dict
        print("\n── explore() ──")
        info = pipeline.explore()
        for k, v in info.items():
            print(f"  {k}: {v}")

        # 3. Stats
        print("\n── stats() ──")
        print(pipeline.stats().to_string(index=False))

        # 4. SQL queries
        print("\n── query(): top 10 by Tm ──")
        top_tm = pipeline.query("""
            SELECT s.sequence, p.value AS melting_temperature
            FROM sequences s
            JOIN predictions p ON s.sequence_id = p.sequence_id
            WHERE p.prediction_type = 'melting_temperature'
            ORDER BY p.value DESC
            LIMIT 10
        """)
        print(top_tm.to_string(index=False))

        print("\n── query(): Tm vs solubility ──")
        tm_vs_sol = pipeline.query("""
            SELECT
                s.sequence,
                tm.value AS tm,
                sol.value AS solubility
            FROM sequences s
            JOIN predictions tm ON s.sequence_id = tm.sequence_id
                AND tm.prediction_type = 'melting_temperature'
            JOIN predictions sol ON s.sequence_id = sol.sequence_id
                AND sol.prediction_type = 'solubility'
            ORDER BY tm.value DESC
            LIMIT 10
        """)
        print(tm_vs_sol.to_string(index=False))

        # 5. Final data
        df_final = pipeline.get_final_data()
        print(f"\n{len(df_final)} sequences survived all filters")

        # ── Run 2: reopen from same DB — zero API calls ────────────────
        print("\n\n" + "=" * 60)
        print("  RESUME: Reopening from DB (zero API calls expected)")
        print("=" * 60)

        ds2 = DuckDBDataStore(db_path)
        pipeline2 = DataPipeline(
            sequences=PEPTIDES,
            datastore=ds2,
            run_id="explore_demo_v2",
            output_dir=str(OUTPUT_DIR),
            verbose=True,
        )
        pipeline2.add_filter(ValidAminoAcidFilter(verbose=True), stage_name="validate")
        pipeline2.add_prediction(
            "temberture-regression", action="predict",
            extractions="prediction",
            columns="melting_temperature",
            stage_name="predict_tm", depends_on=["validate"], batch_size=16,
        )
        pipeline2.add_prediction(
            "biolmsol", action="predict", extractions="solubility_score",
            columns="solubility",
            stage_name="predict_sol", depends_on=["validate"], batch_size=16,
        )
        pipeline2.add_filter(
            ThresholdFilter(column="melting_temperature", min_value=40.0),
            stage_name="filter_tm", depends_on=["predict_tm", "predict_sol"],
        )
        pipeline2.add_filter(
            RankingFilter(column="solubility", n=15, ascending=False),
            stage_name="filter_sol_top15",
        )

        await pipeline2.run_async()
        print("\nCached run complete — no API calls needed!")
        print(pipeline2.summary().to_string(index=False))
        ds2.close()

        # ── Plots ──────────────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive for scripts

            print("\n── plot('funnel') ──")
            pipeline.plot("funnel")

            print("\n── plot('distributions') ──")
            pipeline.plot("distributions")
        except ImportError:
            print("\nmatplotlib not installed — skipping plots")

        # ── Export ─────────────────────────────────────────────────────
        out_csv = OUTPUT_DIR / "explore_results.csv"
        df_final.to_csv(out_csv, index=False)
        print(f"\nExported {len(df_final)} sequences → {out_csv}")

    finally:
        ds.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
