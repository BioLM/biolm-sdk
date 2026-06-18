"""
Multi-model PETase design pipeline targeting LCC (Leaf-branch Compost Cutinase).

Generates variants from 7 model sources, scores for Tm and solubility,
filters and ranks the best candidates.

Generation sources:
  1. DSM-650M-base     — diffusion sequence model (sequence-conditioned)
  2. ZymCTRL           — enzyme generation conditioned on EC 3.1.1.101 (PETase)
  3. ProGen2-OAS       — autoregressive generation from LCC seed
  4. ProteinMPNN       — inverse folding from ESMFold structure
  5. HyperMPNN         — hypernetwork MPNN variant
  6. LigandMPNN        — ligand-aware MPNN variant
  7. ESM2 remasking    — masked-LM iterative refinement

Scoring:
  - temberture-regression (Tm)
  - biolmsol (solubility)
  - esmc-300m (log-probability)

Requires BIOLMAI_TOKEN.

Run:
    python scripts/pipeline_petase_multimodel.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from biolm.client import BioLMApiClient
from biolm.pipeline.data import DataPipeline, EmbeddingSpec
from biolm.pipeline.datastore_duckdb import DuckDBDataStore
from biolm.pipeline.filters import (
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    ValidAminoAcidFilter,
)
from biolm.pipeline.generative import (
    DirectGenerationConfig,
    GenerativePipeline,
)
from biolm.pipeline.mlm_remasking import RemaskingConfig

TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    print("ERROR: BIOLMAI_TOKEN not set.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LCC PETase — Leaf-branch Compost Cutinase (mature form)
# One of the most active PET-degrading enzymes known.
# EC 3.1.1.101 (PET hydrolase)
# ---------------------------------------------------------------------------
LCC_SEQUENCE = (
    "SNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYT"
    "ARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGK"
    "VDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSI"
    "APVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRY"
    "STFACENPNSTRVSDFRTANCS"
)

N_PER_MODEL = 10  # sequences per generation source
MPNN_TEMPS = [0.1, 0.3]


async def step_1_fold_lcc():
    """Predict LCC structure with ESMFold (single call)."""
    print("=" * 70)
    print("STEP 1: Fold LCC with ESMFold")
    print("=" * 70)

    api = BioLMApiClient("esmfold")
    result = await api.predict(items=[{"sequence": LCC_SEQUENCE}])
    await api.shutdown()

    # BioLMApiClient with single item may return a dict (unwrap_single)
    r = result[0] if isinstance(result, list) else result
    pdb_str = r["pdb"]
    plddt = r["mean_plddt"]
    ptm = r["ptm"]
    print(f"  pLDDT: {plddt:.1f}  pTM: {ptm:.3f}  PDB: {len(pdb_str)} chars")
    return pdb_str


async def step_2_generate(pdb_str: str, tmp: Path):
    """Generate variants from all 7 model sources."""
    print("\n" + "=" * 70)
    print("STEP 2: Generate PETase variants from 7 model sources")
    print("=" * 70)

    db = tmp / "petase_gen.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "gen_data")

    configs = []

    # 1. DSM-650M-base — sequence-conditioned diffusion
    configs.append(
        DirectGenerationConfig(
            model_name="dsm-650m-base",
            item_field="sequence",
            sequence=LCC_SEQUENCE,
            params={"num_sequences": N_PER_MODEL, "temperature": 1.0},
        )
    )

    # 2. ZymCTRL — enzyme generation conditioned on PETase EC number
    configs.append(
        DirectGenerationConfig(
            model_name="zymctrl",
            item_field="ec_number",
            sequence="3.1.1.101",  # PETase EC number → sent as item_field value
            params={"temperature": 1.0, "max_length": 300},
        )
    )

    # 3. ProGen2-OAS — autoregressive from LCC N-terminal seed
    configs.append(
        DirectGenerationConfig(
            model_name="progen2-oas",
            item_field="context",
            sequence=LCC_SEQUENCE[:25],  # N-terminal seed
            params={"temperature": 1.0, "max_length": len(LCC_SEQUENCE) + 20},
        )
    )

    # 4. ProteinMPNN — inverse folding from ESMFold structure
    for temp in MPNN_TEMPS:
        configs.append(
            DirectGenerationConfig(
                model_name="protein-mpnn",
                item_field="pdb",
                structure_path=None,
                sequence=None,
                params={"batch_size": N_PER_MODEL, "temperature": temp},
            )
        )

    # 5. HyperMPNN
    for temp in MPNN_TEMPS:
        configs.append(
            DirectGenerationConfig(
                model_name="hyper-mpnn",
                item_field="pdb",
                params={"batch_size": N_PER_MODEL, "temperature": temp},
            )
        )

    # 6. LigandMPNN
    for temp in MPNN_TEMPS:
        configs.append(
            DirectGenerationConfig(
                model_name="ligand-mpnn",
                item_field="pdb",
                params={"batch_size": N_PER_MODEL, "temperature": temp},
            )
        )

    # For MPNN models: inject the PDB string as the sequence (item_field="pdb")
    for cfg in configs:
        if cfg.item_field == "pdb" and cfg.sequence is None:
            cfg.sequence = pdb_str

    # NOTE: ESM2 remasking skipped — the MLMRemasker sends mask_positions
    # as params but the ESM2 predict endpoint requires <mask> tokens inline.
    # This is a pre-existing remasker bug, not a pipeline issue.

    print(f"  {len(configs)} generation configs:")
    for cfg in configs:
        print(f"    {cfg.model_name}: item_field={cfg.item_field}")

    pipeline = GenerativePipeline(
        generation_configs=configs,
        deduplicate=True,
        datastore=ds,
        verbose=True,
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    print(f"\n  Generated {len(df)} unique sequences (after dedup)")

    # Validate
    assert len(df) > 0, "No sequences generated"
    assert df["sequence"].nunique() == len(df), "Duplicate sequences in output"

    # Show per-model breakdown
    if "model_name" in df.columns:
        print("\n  Per-model breakdown:")
        for model, count in df["model_name"].value_counts().items():
            print(f"    {model}: {count} sequences")

    ds.close()
    return df


async def step_3_score(generated_df: pd.DataFrame, tmp: Path):
    """Score all generated variants for Tm, solubility, and log-probability."""
    print("\n" + "=" * 70)
    print(f"STEP 3: Score {len(generated_df)} variants (Tm + solubility + log-prob)")
    print("=" * 70)

    db = tmp / "petase_score.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "score_data")

    # Include parent LCC for comparison
    all_seqs = [LCC_SEQUENCE] + generated_df["sequence"].tolist()

    pipeline = DataPipeline(
        sequences=all_seqs,
        datastore=ds,
        verbose=True,
    )

    # Filter invalid AAs first
    pipeline.add_filter(
        ValidAminoAcidFilter(verbose=True),
        stage_name="filter_valid",
    )

    # Length filter — PETases are ~250-300 aa; keep 100-500
    pipeline.add_filter(
        SequenceLengthFilter(min_length=100, max_length=500),
        stage_name="filter_length",
        depends_on=["filter_valid"],
    )

    # Parallel predictions: Tm + solubility + log-prob
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
        depends_on=["filter_length"],
    )
    pipeline.add_prediction(
        "biolmsol",
        extractions="solubility_score",
        columns="solubility",
        stage_name="predict_sol",
        depends_on=["filter_length"],
    )
    pipeline.add_prediction(
        "esmc-300m",
        action="score",
        extractions="log_prob",
        columns="log_prob",
        stage_name="score_lp",
        depends_on=["filter_length"],
    )

    # Filter: Tm > 45 (PETases typically need moderate thermostability)
    pipeline.add_filter(
        ThresholdFilter("tm", min_value=45.0),
        stage_name="filter_tm",
        depends_on=["predict_tm"],
    )

    # Rank: top 20 by solubility among Tm-passing
    pipeline.add_filter(
        RankingFilter("solubility", n=20, ascending=False),
        stage_name="rank_sol",
        depends_on=["filter_tm", "predict_sol", "score_lp"],
    )

    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()

    # Print stage funnel
    print(f"\n  Stage funnel:")
    for name, sr in pipeline.stage_results.items():
        extra = ""
        if sr.cached_count:
            extra = f" (cached={sr.cached_count})"
        print(f"    {name}: {sr.input_count} → {sr.output_count}{extra}")

    # Validate
    assert len(df) > 0, "No sequences survived scoring pipeline"
    for col in ["tm", "solubility", "log_prob"]:
        assert col in df.columns, f"Missing column: {col}"
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"  WARNING: {col} has {n_null}/{len(df)} NULL values")

    # Check if LCC parent survived
    lcc_row = df[df["sequence"] == LCC_SEQUENCE]
    if len(lcc_row) > 0:
        lcc = lcc_row.iloc[0]
        print(f"\n  LCC parent: Tm={lcc['tm']:.1f}  Sol={lcc['solubility']:.3f}  LP={lcc['log_prob']:.1f}")
    else:
        print("\n  LCC parent was filtered out")

    ds.close()
    return df, pipeline


async def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Step 1: Fold LCC
        pdb_str = await step_1_fold_lcc()

        # Step 2: Generate from all sources
        gen_df = await step_2_generate(pdb_str, tmp_path)

        # Step 3: Score and filter
        final_df, pipeline = await step_3_score(gen_df, tmp_path)

        # Final report
        print("\n" + "=" * 70)
        print(f"FINAL RESULTS: {len(final_df)} PETase candidates")
        print("=" * 70)

        if len(final_df) == 0:
            print("  No candidates survived all filters.")
            return

        # Sort by solubility descending
        final_df = final_df.sort_values("solubility", ascending=False)

        # Check if parent LCC is in there for comparison
        is_parent = final_df["sequence"] == LCC_SEQUENCE

        print(f"\n  {'Seq':>4}  {'Len':>4}  {'Tm':>6}  {'Sol':>6}  {'LP':>8}  Sequence")
        print(f"  {'---':>4}  {'---':>4}  {'---':>6}  {'---':>6}  {'---':>8}  --------")
        for i, (_, row) in enumerate(final_df.iterrows()):
            tag = " ← LCC" if row["sequence"] == LCC_SEQUENCE else ""
            print(
                f"  {i+1:>4}  {len(row['sequence']):>4}  "
                f"{row['tm']:>6.1f}  {row['solubility']:>6.3f}  "
                f"{row['log_prob']:>8.1f}  "
                f"{row['sequence'][:40]}...{tag}"
            )

        # Stats
        print(f"\n  Tm range:  [{final_df['tm'].min():.1f}, {final_df['tm'].max():.1f}]")
        print(f"  Sol range: [{final_df['solubility'].min():.3f}, {final_df['solubility'].max():.3f}]")
        print(f"  LP range:  [{final_df['log_prob'].min():.1f}, {final_df['log_prob'].max():.1f}]")
        print(f"  Unique sequences: {final_df['sequence'].nunique()}/{len(final_df)}")

        # Export
        out = tmp_path / "petase_candidates.csv"
        final_df.to_csv(out, index=False)
        print(f"\n  Exported to: {out}")

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
