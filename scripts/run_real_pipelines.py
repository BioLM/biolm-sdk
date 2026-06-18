"""
Real-API integration tests for the pipeline system.

NO MOCKS — every API call hits the live BioLM endpoints.
Requires BIOLMAI_TOKEN in environment.

Avoids structure prediction models (esmfold, alphafold2) to keep costs down.
Uses fast property models: temberture-regression, biolmsol.

Real API response formats:
  temberture-regression: {"prediction": 48.6}   → extractions="prediction"
  biolmsol:              {"solubility_score": 0.368} → extractions="solubility_score"

Run:
    python scripts/test_real_pipelines.py
"""

import asyncio
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd

from biolm.pipeline.base import PipelineContext, WorkingSet
from biolm.pipeline.data import DataPipeline
from biolm.pipeline.datastore_duckdb import DuckDBDataStore
from biolm.pipeline.filters import (
    CustomFilter,
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    ValidAminoAcidFilter,
)

# ---------------------------------------------------------------------------
TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    print("ERROR: BIOLMAI_TOKEN not set.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Test sequences
# ---------------------------------------------------------------------------

# Thermostable proteins
THERMOSTABLE = [
    "MKILILLGAEKGIGKSTIAKLLAQKFGKIVIETNEKDEDAKSIAEQLGKPFDSVSQLDTIAPQVLAQLLREELSSIMTQIPDVSIVVLDSQGAAITQSALENIRPDYIIVNKMDLKKKDQFAPAAILEQKIRDYFPELDAQSKELEDLFQKAGVEVISQAESFIAQYISQRKDLP",
    "MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAMIEKTISGQDALPNISDAERIFAELLTGLAAAQPGFPLAQLKTFVDQEFAQIKHVLHGISGGVGGVATITAPKKVTLLGRDSFEVAVALMK",
    "MKVLKAGISFTLGSGVILGAFIFFVLVKDNPKLTTGELTLQTAVEMAPQPIAGLSHEIQGVGYEITDDMIEPVTLGSGLVNPVGQILMGGIDGGISALPNLEKFNKTIEGLPDKLKPTFTTIEDAMKLYAAYGG",
]

# Mesophilic
MESOPHILIC = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLK",
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGAAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFH",
]

# Antibody variable regions
ANTIBODY_VH = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSRYTMSWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDDHYSLDYWGQGTLVTVSS",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARVTSANWFDPWGQGTLVTVSS",
    "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVSNIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDGNYYGSGFAYWGQGTLVTVSS",
]
ANTIBODY_VL = [
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
    "SYELTQPPSVSSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIYDNTNRPSGVPDRFSGSKSGTSASLAISGLRSEDEADYYCATWDDSLSGYVFGGGTKLTVL",
    "DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNNKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPLTFGGGTKVEIK",
]

BAD_SEQS = ["MKTXAYIAKQRQISFVKSHFS", "EVQLVES*GGGLVQPGGSLRL"]

ALL_SINGLE = THERMOSTABLE + MESOPHILIC


def _check(df, name, min_rows=1, required_cols=None, no_all_null=True):
    print(f"\n  [{name}] shape={df.shape}")
    if len(df) < min_rows:
        raise AssertionError(f"FAIL: {name} has {len(df)} rows, expected >= {min_rows}")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise AssertionError(f"FAIL: missing columns {missing}. Has: {list(df.columns)}")
    if no_all_null and len(df) > 0:
        for col in df.columns:
            if df[col].isna().all():
                raise AssertionError(f"FAIL: column '{col}' is ALL NULL ({len(df)} rows)")
    if "sequence_id" in df.columns:
        n_uniq = df["sequence_id"].nunique()
        if n_uniq != len(df):
            raise AssertionError(f"FAIL: {len(df)} rows but {n_uniq} unique seq IDs")
    print(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        if col in ("sequence_id", "sequence", "length", "hash", "created_at"):
            continue
        n_notna = df[col].notna().sum()
        suffix = ""
        if df[col].dtype in ("float64", "int64", "float32"):
            vals = df[col].dropna()
            if len(vals):
                suffix = f"  [{vals.min():.2f}, {vals.max():.2f}] mean={vals.mean():.2f}"
        print(f"    {col}: {n_notna}/{len(df)} non-null{suffix}")
    if "sequence" in df.columns:
        print(f"  Unique sequences: {df['sequence'].nunique()}/{len(df)}")
    print(f"  OK")


# ===========================================================================
# Test 1: Basic Tm prediction
# ===========================================================================
async def test_1_basic_tm(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 1: Basic Tm prediction (temberture-regression)")
    print("=" * 70)

    db = tmp_path / "t1.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t1_data")

    pipeline = DataPipeline(sequences=ALL_SINGLE, datastore=ds, verbose=True)
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "Test 1", min_rows=5, required_cols=["sequence", "tm"])
    assert df["tm"].notna().all(), f"Some Tm values NULL"

    print(f"\n  Tm predictions:")
    for _, row in df.iterrows():
        seq_short = row["sequence"][:35] + "..."
        print(f"    {seq_short}  Tm={row['tm']:.1f}")

    ds.close()


# ===========================================================================
# Test 2: Parallel Tm + biolmsol, then filter
# ===========================================================================
async def test_2_parallel_plus_filter(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 2: Parallel (Tm + biolmsol) + filter")
    print("=" * 70)

    db = tmp_path / "t2.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t2_data")

    pipeline = DataPipeline(sequences=ALL_SINGLE, datastore=ds, verbose=True)
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
    )
    pipeline.add_prediction(
        "biolmsol",
        extractions="solubility_score",
        columns="solubility",
        stage_name="predict_sol",
    )
    # Keep top 3 by Tm
    pipeline.add_filter(
        RankingFilter("tm", n=3, ascending=False),
        stage_name="rank_tm",
        depends_on=["predict_tm", "predict_sol"],
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "Test 2", min_rows=1, required_cols=["tm", "solubility"])
    assert len(df) <= 3, f"Ranking let {len(df)} > 3"
    assert df["tm"].notna().all() and df["solubility"].notna().all()

    print(f"\n  Top {len(df)} by Tm:")
    for _, row in df.iterrows():
        seq_short = row["sequence"][:30] + "..."
        print(f"    {seq_short}  Tm={row['tm']:.1f}  Sol={row['solubility']:.3f}")

    ds.close()


# ===========================================================================
# Test 3: ValidAA + bad seqs
# ===========================================================================
async def test_3_valid_aa(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 3: ValidAminoAcidFilter removes bad sequences")
    print("=" * 70)

    all_seqs = ALL_SINGLE + BAD_SEQS
    db = tmp_path / "t3.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t3_data")

    pipeline = DataPipeline(sequences=all_seqs, datastore=ds, verbose=True)
    pipeline.add_filter(ValidAminoAcidFilter(verbose=True), stage_name="filter_valid")
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
        depends_on=["filter_valid"],
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "Test 3", min_rows=len(ALL_SINGLE), required_cols=["tm"])
    for bad in BAD_SEQS:
        assert bad not in df["sequence"].values, f"Bad seq survived: {bad[:20]}..."
    print(f"  Bad sequences removed, {len(df)} remain")

    ds.close()


# ===========================================================================
# Test 4: Cache + resume
# ===========================================================================
async def test_4_cache_resume(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 4: Cache + resume (run twice)")
    print("=" * 70)

    db = tmp_path / "t4.duckdb"
    data_dir = tmp_path / "t4_data"
    run_id = "cache_001"

    # Run 1
    ds1 = DuckDBDataStore(db_path=db, data_dir=data_dir)
    p1 = DataPipeline(sequences=ALL_SINGLE[:3], datastore=ds1, run_id=run_id, verbose=True)
    p1.add_prediction("temberture-regression", extractions="prediction", columns="tm", stage_name="predict_tm")
    t0 = time.time()
    await p1.run_async(enable_streaming=False)
    t1 = time.time() - t0
    df1 = p1.get_final_data()
    ds1.close()
    print(f"\n  Run 1: {t1:.1f}s, {len(df1)} rows")

    # Run 2 — resume
    ds2 = DuckDBDataStore(db_path=db, data_dir=data_dir)
    p2 = DataPipeline(sequences=ALL_SINGLE[:3], datastore=ds2, run_id=run_id, resume=True, verbose=True)
    p2.add_prediction("temberture-regression", extractions="prediction", columns="tm", stage_name="predict_tm")
    t0 = time.time()
    await p2.run_async(enable_streaming=False)
    t2 = time.time() - t0
    df2 = p2.get_final_data()
    ds2.close()
    print(f"  Run 2 (resume): {t2:.1f}s, {len(df2)} rows")

    assert len(df2) == len(df1), f"Resume changed output: {len(df2)} vs {len(df1)}"
    assert t2 < t1 * 0.5 or t2 < 2.0, f"Resume not faster: {t2:.1f}s vs {t1:.1f}s"
    print(f"  Speedup: {t1/max(t2, 0.001):.0f}x")


# ===========================================================================
# Test 5: Trickle new seqs
# ===========================================================================
async def test_5_trickle(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 5: Trickle new sequences (cache reuse)")
    print("=" * 70)

    db = tmp_path / "t5.duckdb"
    data_dir = tmp_path / "t5_data"

    # Run 1: 3 seqs
    ds1 = DuckDBDataStore(db_path=db, data_dir=data_dir)
    p1 = DataPipeline(sequences=ALL_SINGLE[:3], datastore=ds1, run_id="tr1", verbose=True)
    p1.add_prediction("temberture-regression", extractions="prediction", columns="tm", stage_name="predict_tm")
    await p1.run_async(enable_streaming=False)
    ds1.close()

    # Run 2: all 5
    ds2 = DuckDBDataStore(db_path=db, data_dir=data_dir)
    p2 = DataPipeline(sequences=ALL_SINGLE, datastore=ds2, run_id="tr2", verbose=True)
    p2.add_prediction("temberture-regression", extractions="prediction", columns="tm", stage_name="predict_tm")
    await p2.run_async(enable_streaming=False)

    df = p2.get_final_data()
    _check(df, "Test 5", min_rows=5, required_cols=["tm"])

    sr = p2.stage_results.get("predict_tm")
    if sr:
        print(f"  Cache hit: {sr.cached_count} cached, {sr.computed_count} new")
        assert sr.cached_count >= 3, f"Expected >= 3 cached, got {sr.cached_count}"

    ds2.close()


# ===========================================================================
# Test 6: Multi-column antibody H+L
# ===========================================================================
async def test_6_multi_col_antibody(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 6: Multi-column antibody (heavy + light)")
    print("=" * 70)

    df_input = pd.DataFrame({"heavy_chain": ANTIBODY_VH, "light_chain": ANTIBODY_VL})
    db = tmp_path / "t6.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t6_data")

    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=True,
    )
    # temberture-regression takes a single 'sequence' field.
    # Use item_columns to send just the heavy chain for Tm prediction.
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
        item_columns={"sequence": "heavy_chain"},
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "Test 6", min_rows=3, required_cols=["heavy_chain", "light_chain", "tm"])

    # Verify columns in DuckDB
    db_cols = {r[0] for r in ds.conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='sequences'"
    ).fetchall()}
    assert "heavy_chain" in db_cols and "light_chain" in db_cols

    # export_to_dataframe
    df_exp = ds.export_to_dataframe(include_predictions=True)
    assert "heavy_chain" in df_exp.columns and "light_chain" in df_exp.columns
    _check(df_exp, "Test 6 export", min_rows=3, required_cols=["heavy_chain", "light_chain", "tm"])

    print(f"\n  Results:")
    for _, row in df.iterrows():
        vh = row["heavy_chain"][:20] + "..."
        vl = row["light_chain"][:20] + "..."
        print(f"    VH={vh}  VL={vl}  Tm={row['tm']:.1f}")

    ds.close()


# ===========================================================================
# Test 7: Multi-column dedup
# ===========================================================================
async def test_7_multi_col_dedup(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 7: Multi-column dedup (4 rows → 3 unique)")
    print("=" * 70)

    df_input = pd.DataFrame({
        "heavy_chain": [ANTIBODY_VH[0], ANTIBODY_VH[0], ANTIBODY_VH[0], ANTIBODY_VH[1]],
        "light_chain": [ANTIBODY_VL[0], ANTIBODY_VL[1], ANTIBODY_VL[0], ANTIBODY_VL[0]],
    })

    db = tmp_path / "t7.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t7_data")

    pipeline = DataPipeline(
        sequences=df_input, input_columns=["heavy_chain", "light_chain"],
        datastore=ds, verbose=True,
    )
    # Use item_columns to send heavy_chain as the 'sequence' to the API
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
        item_columns={"sequence": "heavy_chain"},
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    assert len(df) == 3, f"Expected 3, got {len(df)}"
    assert df["sequence_id"].nunique() == 3
    _check(df, "Test 7", min_rows=3, required_cols=["tm", "heavy_chain", "light_chain"])
    print(f"  4 input → {len(df)} unique")

    ds.close()


# ===========================================================================
# Test 7b: Tm on heavy AND light separately (parallel), zipped back together
# ===========================================================================
async def test_7b_parallel_per_chain_tm(tmp_path):
    """Predict Tm on heavy_chain and light_chain separately, in parallel.
    Both predictions should appear as separate columns in the final output."""
    print("\n" + "=" * 70)
    print("TEST 7b: Parallel per-chain Tm (heavy + light separately)")
    print("=" * 70)

    df_input = pd.DataFrame({
        "heavy_chain": ANTIBODY_VH,
        "light_chain": ANTIBODY_VL,
    })
    print(f"  Input: {len(df_input)} antibody pairs")

    db = tmp_path / "t7b.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t7b_data")

    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=True,
    )

    # Predict Tm on heavy chain
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm_heavy",
        stage_name="predict_tm_heavy",
        item_columns={"sequence": "heavy_chain"},
        depends_on=[],
    )
    # Predict Tm on light chain (runs in parallel — same dependency level)
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm_light",
        stage_name="predict_tm_light",
        item_columns={"sequence": "light_chain"},
        depends_on=[],
    )
    # Filter: keep pairs where BOTH chains have Tm > 45
    pipeline.add_filter(
        ThresholdFilter("tm_heavy", min_value=45.0),
        stage_name="filter_tm_heavy",
        depends_on=["predict_tm_heavy"],
    )
    pipeline.add_filter(
        ThresholdFilter("tm_light", min_value=45.0),
        stage_name="filter_tm_light",
        depends_on=["predict_tm_light"],
    )

    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "Test 7b", min_rows=1, required_cols=[
        "heavy_chain", "light_chain", "tm_heavy", "tm_light"
    ])

    # Both Tm columns should be non-null
    assert df["tm_heavy"].notna().all(), "tm_heavy has NULLs"
    assert df["tm_light"].notna().all(), "tm_light has NULLs"
    # Both should pass their threshold
    assert (df["tm_heavy"] >= 45.0).all(), "Some tm_heavy < 45"
    assert (df["tm_light"] >= 45.0).all(), "Some tm_light < 45"

    print(f"\n  Per-chain Tm results ({len(df)} pairs):")
    for _, row in df.iterrows():
        vh = row["heavy_chain"][:20] + "..."
        vl = row["light_chain"][:20] + "..."
        print(f"    VH={vh}  Tm_H={row['tm_heavy']:.1f}  |  VL={vl}  Tm_L={row['tm_light']:.1f}")

    # Verify export also has both columns
    df_exp = ds.export_to_dataframe(include_predictions=True)
    assert "tm_heavy" in df_exp.columns, "tm_heavy missing from export"
    assert "tm_light" in df_exp.columns, "tm_light missing from export"
    print(f"  export_to_dataframe has both tm_heavy and tm_light ✓")

    ds.close()


# ===========================================================================
# Test 8: Chained filter funnel
# ===========================================================================
async def test_8_chained_filters(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 8: Chained filter funnel (ValidAA → Length → Tm → Top-2)")
    print("=" * 70)

    all_seqs = ALL_SINGLE + BAD_SEQS
    db = tmp_path / "t8.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t8_data")

    pipeline = DataPipeline(sequences=all_seqs, datastore=ds, verbose=True)
    pipeline.add_filter(ValidAminoAcidFilter(verbose=True), stage_name="f_valid")
    pipeline.add_filter(SequenceLengthFilter(min_length=100), stage_name="f_length", depends_on=["f_valid"])
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
        depends_on=["f_length"],
    )
    pipeline.add_filter(RankingFilter("tm", n=2, ascending=False), stage_name="rank_top2", depends_on=["predict_tm"])
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "Test 8", min_rows=1, required_cols=["tm"])
    assert len(df) <= 2
    for bad in BAD_SEQS:
        assert bad not in df["sequence"].values
    assert all(len(s) >= 100 for s in df["sequence"])

    print(f"\n  Funnel:")
    for name, sr in pipeline.stage_results.items():
        print(f"    {name}: {sr.input_count} → {sr.output_count}")

    ds.close()


# ===========================================================================
# Test 9: Context
# ===========================================================================
async def test_9_context(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 9: Pipeline context")
    print("=" * 70)

    db = tmp_path / "t9.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t9_data")

    pipeline = DataPipeline(sequences=ALL_SINGLE[:2], datastore=ds, verbose=True)
    pipeline.context.set("experiment", "thermo_screen")
    pipeline.context.set("config", {"model": "temberture-regression"})
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "Test 9", min_rows=2, required_cols=["tm"])
    assert pipeline.context.get("experiment") == "thermo_screen"
    assert pipeline.context.get("config")["model"] == "temberture-regression"
    assert pipeline.context.get("missing", "default") == "default"

    ctx2 = PipelineContext(ds, "other_run")
    assert ctx2.get("experiment") is None
    print("  Context persists and is isolated")

    ds.close()


# ===========================================================================
# Test 10: Streaming
# ===========================================================================
async def test_10_streaming(tmp_path):
    print("\n" + "=" * 70)
    print("TEST 10: Streaming mode (predict → filter)")
    print("=" * 70)

    db = tmp_path / "t10.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "t10_data")

    pipeline = DataPipeline(sequences=ALL_SINGLE, datastore=ds, verbose=True)
    pipeline.add_prediction(
        "temberture-regression",
        extractions="prediction",
        columns="tm",
        stage_name="predict_tm",
    )
    pipeline.add_filter(RankingFilter("tm", n=3, ascending=False), stage_name="rank_top3", depends_on=["predict_tm"])
    await pipeline.run_async(enable_streaming=True)

    df = pipeline.get_final_data()
    _check(df, "Test 10", min_rows=1, required_cols=["tm"])
    assert len(df) <= 3

    ds.close()


# ===========================================================================
async def main():
    results = {}
    tests = [
        ("Test 1: Basic Tm", test_1_basic_tm),
        ("Test 2: Parallel + filter", test_2_parallel_plus_filter),
        ("Test 3: ValidAA filter", test_3_valid_aa),
        ("Test 4: Cache + resume", test_4_cache_resume),
        ("Test 5: Trickle new seqs", test_5_trickle),
        ("Test 6: Multi-col antibody", test_6_multi_col_antibody),
        ("Test 7: Multi-col dedup", test_7_multi_col_dedup),
        ("Test 7b: Parallel per-chain Tm", test_7b_parallel_per_chain_tm),
        ("Test 8: Chained filters", test_8_chained_filters),
        ("Test 9: Context", test_9_context),
        ("Test 10: Streaming", test_10_streaming),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, func in tests:
            try:
                await func(tmp_path)
                results[name] = "PASS"
            except Exception as e:
                results[name] = f"FAIL: {e}"
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, status in results.items():
        icon = "+" if status == "PASS" else "X"
        print(f"  [{icon}] {name}: {status}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} tests passed!")
    else:
        failed = sum(1 for s in results.values() if s != "PASS")
        print(f"\n{failed}/{len(results)} tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
