"""
Simple Pipeline Example

This example demonstrates basic usage of the pipeline system.
All examples make live API calls — requires BIOLMAI_TOKEN in environment.
"""

import traceback

import os
from pathlib import Path

from biolm.pipeline import (
    DataPipeline,
    DirectGenerationConfig,
    GenerativePipeline,
    Predict,
)
from biolm.pipeline.filters import (
    DiversitySamplingFilter,
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
)
from biolm.pipeline.visualization import PipelinePlotter

# PDB fixtures bundled with the SDK
_HERE = Path(__file__).parent.parent
SAMPLE_PDB = str(_HERE / "tests/fixtures/sample.pdb")        # single chain
ANTIBODY_PDB = str(_HERE / "tests/fixtures/multi_model.pdb") # H + L chains

SEQUENCES = [
    "MKTAYIAKQRQGHQAMAEIKQ",
    "MKLAVIDSAQRQGHQAMAEIKQ",
    "MKTAYIDSAQRQGHQAMAEIKQ",
    "MKTAYIAKQRQGHQAMAEI",
    "MKTAYIAKQRQGHQAMAEIKQGHQAMAEIKQ",
    "MSILVTRPSPAGEELVSRLR",
    "MENDELMENDELMENDEL",
    "ACDEFGHIKLMNPQRSTVWY",
    "MKTAYIAKQRQGHQAMAEIKQLVSRLR",
    "MSILVTRPSPAGEEL",
    "MKLAVIDSAQRQGHQAMAEIKQLVSR",
    "MKTAYIDSAQRQGHQAMAEIKQVSRL",
    "ACDEFGHIKLMNPQRSTVWYACDE",
    "MSILVTRPSPAGEELVSRLRTLGQ",
    "MENDELMENDELMENDELMEND",
]


def example_1_quick_prediction():
    """Example 1: Quick single-step prediction — top 10 by Tm."""
    print("=" * 60)
    print("Example 1: Quick Prediction (top 10 by Tm)")
    print("=" * 60)

    df = Predict(
        "temberture-regression",
        sequences=SEQUENCES,
        extractions="prediction",
        columns="tm",
        verbose=False,
    )
    df_sorted = df.sort_values("tm", ascending=False).reset_index(drop=True)
    print(f"\n{len(df)} sequences predicted. Top 10 by Tm:")
    print(df_sorted[["sequence", "tm"]].head(10).to_string(index=True))


def example_2_data_pipeline():
    """Example 2: Data pipeline with length filter → Tm prediction → Tm filter."""
    print("\n" + "=" * 60)
    print("Example 2: Data Pipeline (length filter → Tm → threshold)")
    print("=" * 60)

    pipeline = DataPipeline(sequences=SEQUENCES)
    pipeline.add_filter(
        SequenceLengthFilter(min_length=20, max_length=30), stage_name="length_filter"
    )
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="tm_prediction",
    )
    pipeline.add_filter(ThresholdFilter("tm", min_value=49), stage_name="tm_filter")

    print("\nRunning pipeline...")
    pipeline.run()

    df = pipeline.get_final_data()
    df_sorted = df.sort_values("tm", ascending=False).reset_index(drop=True)
    print(f"\nSurvivors ({len(df)} sequences), sorted by Tm:")
    print(df_sorted[["sequence", "tm"]].head(10).to_string(index=True))

    print("\nPipeline summary:")
    print(pipeline.summary())


def example_3_generative_pipeline():
    """Example 3: Sequence-conditioned generation — progen2-oas → Tm → top 10.

    progen2-oas generates antibody sequences from a seed context string.
    For PDB-input models (protein-mpnn, antifold, ligand-mpnn) see:
      scripts/pipeline_antibody_antifold.py  — antifold H+L chain design
      scripts/pipeline_mpnn_multi.py         — protein-mpnn multi-temperature scan
    Those use DirectGenerationConfig(structure_path=..., item_field='pdb').
    """
    print("\n" + "=" * 60)
    print("Example 3: Generative Pipeline (progen2-oas → Tm → top 10)")
    print("=" * 60)

    config = DirectGenerationConfig(
        model_name="progen2-oas",
        sequence="M",           # seed context — progen2-oas extends from here
        item_field="context",
        params={"temperature": 0.7, "top_p": 0.9, "num_samples": 3, "max_length": 50},
    )

    pipeline = GenerativePipeline(generation_configs=[config], deduplicate=True)
    pipeline.add_prediction(
        "temberture-regression", extractions="prediction", columns="tm"
    )
    pipeline.add_filter(
        RankingFilter("tm", n=10, ascending=False), stage_name="top10_by_tm"
    )

    print("\nRunning generative pipeline...")
    pipeline.run()

    df = pipeline.get_final_data()
    df_sorted = df.sort_values("tm", ascending=False).reset_index(drop=True)
    cols = [c for c in ["sequence", "tm"] if c in df_sorted.columns]
    print(f"\nTop 10 generated sequences by Tm (from {len(df)} survivors):")
    print(df_sorted[cols].head(10).to_string(index=True))

    print("\nPipeline summary:")
    print(pipeline.summary())


def example_4_visualization():
    """Example 4: Pipeline visualization — Tm funnel + distribution plots."""
    print("\n" + "=" * 60)
    print("Example 4: Visualization (Tm funnel + distribution)")
    print("=" * 60)

    pipeline = DataPipeline(sequences=SEQUENCES)
    pipeline.add_prediction(
        "temberture-regression", extractions="prediction", columns="tm"
    )
    pipeline.add_filter(ThresholdFilter("tm", min_value=49), stage_name="tm_filter")

    pipeline.run()

    df = pipeline.get_final_data()
    print(f"\nAll {len(df)} sequences with Tm ≥ 49:")
    print(df[["sequence", "tm"]].sort_values("tm", ascending=False).head(10).to_string(index=True))

    print("\nGenerating visualizations...")
    plotter = PipelinePlotter(pipeline)
    plotter.plot_funnel(save_path="pipeline_funnel.png")
    plotter.plot_distribution("tm", save_path="tm_distribution.png")
    print("Saved: pipeline_funnel.png, tm_distribution.png")


def example_5_datastore_usage():
    """Example 5: Direct DataStore usage."""
    print("\n" + "=" * 60)
    print("Example 5: DataStore (direct usage)")
    print("=" * 60)

    from biolm.pipeline import DataStore

    store = DataStore("example.db", "example_data")

    try:
        seqs = SEQUENCES[:5]
        ids = [store.add_sequence(s) for s in seqs]
        print(f"Added {len(ids)} sequences (ids {ids[0]}–{ids[-1]})")

        fake_tm = [48.5, 50.6, 47.2, 49.8, 51.3]
        for sid, tm in zip(ids, fake_tm):
            store.add_prediction(sid, "melting_temperature", "temberture-regression", tm)

        df = store.export_to_dataframe()
        print(f"\nExported DataFrame ({len(df)} rows):")
        print(df[["sequence", "melting_temperature"]].sort_values("melting_temperature", ascending=False).head(10).to_string(index=True))

        print("\nDataStore stats:")
        print(store.get_stats())
    finally:
        store.close()


if __name__ == "__main__":
    print("BioLM Pipeline Examples\n")

    for label, fn in [
        ("Example 1", example_1_quick_prediction),
        ("Example 2", example_2_data_pipeline),
        ("Example 3", example_3_generative_pipeline),
        ("Example 4", example_4_visualization),
        ("Example 5", example_5_datastore_usage),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"\n{label} failed: {e}")
            traceback.print_exc()

    print("\nAll examples completed!")
