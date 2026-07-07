"""
Advanced Pipeline Examples

Demonstrates new features:
1. Multiple predictions at once (parallel)
2. Flattened sampling parameters
3. RankingFilter for top-N selection
4. Resample flag control
5. MLM remasking
"""

import sys



import numpy as np
import pandas as pd

from biolm.pipeline import (
    CONSERVATIVE_CONFIG,
    MODERATE_CONFIG,
    DataPipeline,
    DiversitySamplingFilter,
    MLMRemasker,
    RemaskingConfig,
)


def example_1_parallel_predictions():
    """Example 1: Run multiple models in parallel at the same level."""
    print("=" * 60)
    print("Example 1: Parallel Predictions")
    print("=" * 60)

    sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKTAYIDSAQ"]

    pipeline = DataPipeline(sequences=sequences, verbose=False)

    # Add multiple predictions at once - they run in parallel!
    pipeline.add_predictions(
        [
            {"model_name": "temberture-regression", "extractions": "prediction", "columns": "tm"},
            {"model_name": "biolmsol", "extractions": "solubility_score", "columns": "solubility"},
        ]
    )

    print(f"Added {len(pipeline.stages)} stages")
    print("All will run in parallel (no dependencies)")

    # Check execution order
    levels = pipeline._resolve_dependencies()
    print(f"\nExecution plan: {len(levels)} level(s)")
    for i, level in enumerate(levels):
        print(f"  Level {i+1}: {[s.name for s in level]}")


def example_2_flattened_sampling_params():
    """Example 2: Flattened sampling parameters (tabular, not nested)."""
    print("\n" + "=" * 60)
    print("Example 2: Flattened Sampling Parameters")
    print("=" * 60)

    from biolm.pipeline import DataStore

    store = DataStore("example_sampling.db", "example_sampling_data")

    try:
        # Add sequence with generation metadata
        seq_id = store.add_sequence("MKTAYIAKQRQ")

        # All sampling params are flattened columns (not nested JSON!)
        store.add_generation_metadata(
            seq_id,
            model_name="proteinmpnn",
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            num_return_sequences=10,
            do_sample=True,
            repetition_penalty=1.2,
            max_length=500,
        )

        # Export - all params are separate columns
        df = store.export_to_dataframe(
            include_sequences=True, include_generation_metadata=True
        )

        print("\nColumns in exported DataFrame:")
        print(df.columns.tolist())

        print("\nGeneration metadata (flattened):")
        print(df[["sequence", "gen_model_name", "gen_temperature", "gen_top_k", "gen_top_p"]].head())

        # Can easily filter by any sampling parameter!
        df_high_temp = df[df["gen_temperature"] > 0.8]
        df_topk_50 = df[df["gen_top_k"] == 50]

        print(f"\nSequences with temperature > 0.8: {len(df_high_temp)}")
        print(f"Sequences with top_k=50: {len(df_topk_50)}")
    finally:
        store.close()


def example_3_ranking_filter():
    """Example 3: RankingFilter for top-N selection."""
    print("\n" + "=" * 60)
    print("Example 3: Ranking Filter")
    print("=" * 60)

    # Create synthetic data
    df = pd.DataFrame(
        {
            "sequence": [f"SEQ{i}" for i in range(100)],
            "tm": np.random.uniform(40, 80, 100),
            "plddt": np.random.uniform(60, 95, 100),
        }
    )

    print(f"Starting with {len(df)} sequences")

    # Top 10 by Tm
    from biolm.pipeline.filters import RankingFilter

    filter1 = RankingFilter("tm", n=10, ascending=False)
    df_top10 = filter1(df)

    print("\nTop 10 by Tm:")
    print(df_top10[["sequence", "tm"]].sort_values("tm", ascending=False))

    # Bottom 5 by Tm
    filter2 = RankingFilter("tm", n=5, ascending=True)
    df_bottom5 = filter2(df)

    print("\nBottom 5 by Tm:")
    print(df_bottom5[["sequence", "tm"]].sort_values("tm"))

    # Top 10% by pLDDT
    filter3 = RankingFilter(
        "plddt", method="percentile", percentile=10, ascending=False
    )
    df_top_percentile = filter3(df)

    print(f"\nTop 10% by pLDDT: {len(df_top_percentile)} sequences")
    print(
        f"pLDDT range: {df_top_percentile['plddt'].min():.1f} - {df_top_percentile['plddt'].max():.1f}"
    )


def example_4_resample_flag():
    """Example 4: Resample flag for incremental sampling."""
    print("\n" + "=" * 60)
    print("Example 4: Resample Flag")
    print("=" * 60)

    # Initial dataset
    df = pd.DataFrame(
        {
            "sequence": [f"SEQ{i}" for i in range(50)],
            "tm": np.random.uniform(40, 80, 50),
        }
    )

    print(f"Initial dataset: {len(df)} sequences")

    # Sample 10 with resample=False
    filter_obj = DiversitySamplingFilter(
        n_samples=10,
        method="top",
        score_column="tm",
        resample=False,  # Don't resample!
    )

    df_sampled = filter_obj(df)
    sampled_seqs = set(df_sampled["sequence"])

    print(f"\nFirst sampling: {len(df_sampled)} sequences")
    print(f"Top sequence: {df_sampled['sequence'].iloc[0]}")

    # Add new sequences
    df_new = pd.DataFrame(
        {
            "sequence": [f"NEW{i}" for i in range(30)],
            "tm": np.random.uniform(40, 80, 30),
        }
    )

    df_expanded = pd.concat([df_sampled, df_new], ignore_index=True)

    print(f"\nAdded {len(df_new)} new sequences")
    print(f"Total: {len(df_expanded)} sequences")

    # Sample again with resample=False
    # Should keep old samples + add new ones up to n_samples
    df_sampled2 = filter_obj(df_expanded)

    print(f"\nSecond sampling: {len(df_sampled2)} sequences")

    # Check if original samples are preserved
    preserved = sum(1 for seq in sampled_seqs if seq in set(df_sampled2["sequence"]))
    print(f"Preserved from first sampling: {preserved}/{len(sampled_seqs)}")

    new_added = sum(1 for seq in df_sampled2["sequence"] if seq.startswith("NEW"))
    print(f"New sequences added: {new_added}")

    print("\n✓ With resample=False, we only shift the distribution")
    print("  Old samples are kept, new ones added to fill up to n_samples")


async def _example_5_mlm_remasking_async():
    parent_sequence = "MKTAYIAKQRQGHQAMAEIKQ"

    # Conservative remasking (10% masked)
    print("\nConservative remasking (10% masked, low temp):")
    remasker_conservative = MLMRemasker(CONSERVATIVE_CONFIG)
    variants = await remasker_conservative.generate_variants(parent_sequence, num_variants=5)
    for i, (var_seq, metadata) in enumerate(variants):
        print(
            f"  Variant {i+1}: {metadata['num_mutations']} mutations ({metadata['mutation_rate']:.1%})"
        )

    # Moderate remasking (15% masked)
    print("\nModerate remasking (15% masked, medium temp):")
    remasker_moderate = MLMRemasker(MODERATE_CONFIG)
    variants = await remasker_moderate.generate_variants(parent_sequence, num_variants=5)
    for i, (var_seq, metadata) in enumerate(variants):
        print(
            f"  Variant {i+1}: {metadata['num_mutations']} mutations ({metadata['mutation_rate']:.1%})"
        )

    # Custom remasking with conserved positions
    print("\nCustom remasking with conserved positions:")
    custom_config = RemaskingConfig(
        mask_fraction=0.2,
        num_iterations=3,
        conserved_positions=[0, 1, 2],  # Keep first 3 positions
        mask_strategy="blocks",  # Mask in blocks
        block_size=3,
    )
    remasker_custom = MLMRemasker(custom_config)
    variants = await remasker_custom.generate_variants(parent_sequence, num_variants=3)
    print(f"\nGenerated {len(variants)} variants with conserved positions")
    for i, (var_seq, metadata) in enumerate(variants):
        preserved = var_seq[:3] == parent_sequence[:3]
        print(f"  Variant {i+1}: First 3 positions preserved: {preserved}")


def example_5_mlm_remasking():
    """Example 5: MLM remasking for variant generation."""
    import asyncio

    print("\n" + "=" * 60)
    print("Example 5: MLM Remasking")
    print("=" * 60)
    asyncio.run(_example_5_mlm_remasking_async())


def example_6_full_pipeline_with_new_features():
    """Example 6: Full pipeline using all new features."""
    print("\n" + "=" * 60)
    print("Example 6: Full Pipeline with New Features")
    print("=" * 60)

    sequences = [f"MKTAYIAKQRQ{i}" for i in range(20)]

    pipeline = DataPipeline(
        sequences=sequences, output_dir="advanced_pipeline_output", verbose=False
    )

    # Stage 1: Multiple predictions in parallel
    print("\n1. Adding parallel predictions...")
    pipeline.add_predictions([
        {"model_name": "temberture-regression", "extractions": "prediction", "columns": "tm"},
        {"model_name": "biolmsol", "extractions": "solubility_score", "columns": "solubility"},
    ])

    # Stage 2: Rank and select top 10 by Tm (depends on predictions)
    print("2. Adding ranking filter...")
    from biolm.pipeline.filters import RankingFilter

    pipeline.add_filter(
        RankingFilter("tm", n=10, ascending=False),
        stage_name="rank_by_tm",
        depends_on=["predict_tm"],  # Wait for Tm prediction
    )

    # Stage 3: Sample 5 for further analysis (depends on ranking)
    print("3. Adding diversity sampling...")
    pipeline.add_filter(
        DiversitySamplingFilter(
            n_samples=5,
            method="spread",
            score_column="tm",
            resample=False,  # Don't resample if run again
        ),
        stage_name="diversity_sample",
        depends_on=["rank_by_tm"],
    )

    print("\n4. Execution plan:")
    levels = pipeline._resolve_dependencies()
    for i, level in enumerate(levels):
        print(f"   Level {i+1}: {[s.name for s in level]}")

    print("\n✓ Pipeline configured successfully!")
    print(f"   Total stages: {len(pipeline.stages)}")


if __name__ == "__main__":
    print("BioLM Pipeline - Advanced Examples\n")

    try:
        example_1_parallel_predictions()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_2_flattened_sampling_params()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_3_ranking_filter()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_4_resample_flag()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_5_mlm_remasking()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    try:
        example_6_full_pipeline_with_new_features()
    except Exception as e:
        print(f"Example 6 failed: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
