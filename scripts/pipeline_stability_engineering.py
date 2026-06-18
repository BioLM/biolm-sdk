"""Nanobody stability engineering: ESMFold → SPURS + ThermoMPNN + TEMPRO.

Demonstrates a stability-focused protein engineering workflow:
  1. ESMFold: Fold nanobody to get 3-D structure
  2. SPURS:   Structure-aware ddG prediction per mutation (GNN + transformer)
  3. ThermoMPNN: Structure-based ddG prediction per mutation (MPNN)
  4. TEMPRO:  Absolute Tm prediction for each variant sequence (nanobody-specific)

Scans all 19 substitutions at each CDR3 position, ranks by consensus
stabilization (negative ddG from both SPURS and ThermoMPNN, high Tm from TEMPRO).

Usage:
    export BIOLMAI_TOKEN=<your_token>
    python scripts/pipeline_stability_engineering.py
"""

import asyncio
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from biolm.client import BioLMApiClient
from biolm.pipeline import DataPipeline, DuckDBDataStore

TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    print("ERROR: BIOLMAI_TOKEN not set.")
    sys.exit(1)

OUTPUT_DIR = Path("outputs/stability_engineering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Target: anti-GFP nanobody (cAbGFP4, PDB 3OGO chain A)
# 126 aa — within TEMPRO's 100-160 range
# ---------------------------------------------------------------------------
NANOBODY_SEQ = (
    "QVQLVESGGALVQPGGSLRLSCAASGFPVNRYSMRWYRQAPGKEREWVAGMSSAGDRSSYE"
    "DSVKGRFTISRDDARNTVYLQMNSLKPEDTAVYYCNVNVGFEYWGQGTQVTVSS"
)

# CDR3 region (IMGT definition for nanobodies): approx positions 97-113
# We'll scan a focused window — CDR3 loop is the primary diversity/engineering region
CDR3_START = 97   # 1-indexed
CDR3_END = 113

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Step 1: Fold
# ---------------------------------------------------------------------------
async def step_fold() -> str:
    """Fold nanobody with ESMFold."""
    print("=" * 70)
    print("STEP 1: Fold nanobody with ESMFold")
    print("=" * 70)

    api = BioLMApiClient("esmfold")
    result = await api.predict(items=[{"sequence": NANOBODY_SEQ}])
    await api.shutdown()

    r = result[0] if isinstance(result, list) else result
    pdb_str = r["pdb"]
    plddt = r["mean_plddt"]
    ptm = r["ptm"]
    print(f"  pLDDT: {plddt:.1f}  pTM: {ptm:.3f}  PDB: {len(pdb_str)} chars")
    print(f"  Sequence length: {len(NANOBODY_SEQ)} aa")
    print(f"  Scanning CDR3: positions {CDR3_START}-{CDR3_END}")
    return pdb_str


# ---------------------------------------------------------------------------
# Step 2: Generate mutations
# ---------------------------------------------------------------------------
def generate_mutations() -> list[str]:
    """Generate all single-point mutations in CDR3."""
    mutations = []
    for pos in range(CDR3_START, CDR3_END + 1):
        wt = NANOBODY_SEQ[pos - 1]  # 1-indexed → 0-indexed
        for mt in CANONICAL_AA:
            if mt != wt:
                mutations.append(f"{wt}{pos}{mt}")
    return mutations


# ---------------------------------------------------------------------------
# Step 3: Score with SPURS (batched, batch_size=4)
# ---------------------------------------------------------------------------
async def step_spurs(pdb_str: str, mutations: list[str]) -> dict[str, float]:
    """Score each mutation with SPURS (structure-aware ddG)."""
    print(f"\n{'=' * 70}")
    print(f"STEP 2: SPURS ddG prediction ({len(mutations)} mutations)")
    print("=" * 70)

    api = BioLMApiClient("spurs")
    spurs_ddg: dict[str, float] = {}

    # SPURS accepts batch_size=4 items, each with its own mutations list
    # But all share the same PDB + sequence. Send 1 mutation per item.
    batch_size = 4
    for i in range(0, len(mutations), batch_size):
        batch = mutations[i : i + batch_size]
        items = [
            {
                "sequence": NANOBODY_SEQ,
                "pdb": pdb_str,
                "chain_id": "A",
                "mutations": [mut],
            }
            for mut in batch
        ]
        try:
            results = await api.predict(items=items)
            if not isinstance(results, list):
                results = [results]
            for mut, res in zip(batch, results):
                if isinstance(res, dict) and "ddG" in res:
                    spurs_ddg[mut] = res["ddG"]
                elif isinstance(res, dict) and "error" in res:
                    print(f"  SPURS error for {mut}: {str(res['error'])[:60]}")
        except Exception as e:
            print(f"  SPURS batch error: {e}")

        if (i + batch_size) % 40 == 0 or i + batch_size >= len(mutations):
            print(f"  Processed {min(i + batch_size, len(mutations))}/{len(mutations)}")

    await api.shutdown()
    print(f"  Got {len(spurs_ddg)} SPURS ddG values")
    return spurs_ddg


# ---------------------------------------------------------------------------
# Step 4: Score with ThermoMPNN (batch_size=1, needs PDB)
# ---------------------------------------------------------------------------
async def step_thermompnn(pdb_str: str, mutations: list[str]) -> dict[str, float]:
    """Score each mutation with ThermoMPNN (structure-based ddG)."""
    print(f"\n{'=' * 70}")
    print(f"STEP 3: ThermoMPNN ddG prediction ({len(mutations)} mutations)")
    print("=" * 70)

    api = BioLMApiClient("thermompnn")
    thermo_ddg: dict[str, float] = {}

    # ThermoMPNN: batch_size=1, one mutation per call (multi-mutation returns None
    # due to unwrap_single in the client). Use concurrent tasks for speed.
    sem = asyncio.Semaphore(4)

    async def _score_one(mut: str) -> tuple[str, float | None]:
        async with sem:
            try:
                result = await api.predict(
                    items=[{"pdb": pdb_str, "mutations": [mut]}],
                    params={"chain": "A"},
                )
                r = result[0] if isinstance(result, list) else result
                if isinstance(r, dict) and "ddg" in r:
                    return mut, r["ddg"]
            except Exception:
                pass
            return mut, None

    tasks = [_score_one(mut) for mut in mutations]
    for i in range(0, len(tasks), 40):
        batch = tasks[i : i + 40]
        results = await asyncio.gather(*batch)
        for mut, ddg in results:
            if ddg is not None:
                thermo_ddg[mut] = ddg
        print(f"  Processed {min(i + 40, len(mutations))}/{len(mutations)}")

    await api.shutdown()
    print(f"  Got {len(thermo_ddg)} ThermoMPNN ddG values")
    return thermo_ddg


# ---------------------------------------------------------------------------
# Step 5: Score variant sequences with TEMPRO (Tm via pipeline)
# ---------------------------------------------------------------------------
async def step_tempro(variant_seqs: list[str], variant_labels: list[str]) -> dict[str, float]:
    """Score each variant's absolute Tm with TEMPRO-650M."""
    print(f"\n{'=' * 70}")
    print(f"STEP 4: TEMPRO-650M Tm prediction ({len(variant_seqs)} variants)")
    print("=" * 70)

    db_path = OUTPUT_DIR / "tempro.duckdb"
    ds = DuckDBDataStore(db_path)

    pipeline = DataPipeline(
        sequences=variant_seqs,
        datastore=ds,
        run_id="stability_tempro",
        verbose=True,
    )

    pipeline.add_prediction(
        model_name="tempro-650m",
        action="predict",
        extractions="tm",
        columns="tm",
        stage_name="tempro",
        batch_size=8,
        max_concurrent=4,
    )

    await pipeline.run_async()
    df = pipeline.get_final_data()

    # Map back to mutation labels
    tempro_tm: dict[str, float] = {}
    for label, seq in zip(variant_labels, variant_seqs):
        row = df[df["sequence"] == seq]
        if not row.empty and pd.notna(row.iloc[0].get("tm")):
            tempro_tm[label] = row.iloc[0]["tm"]

    ds.close()
    print(f"  Got {len(tempro_tm)} TEMPRO Tm values")
    return tempro_tm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # Step 1: Fold
    pdb_str = await step_fold()

    # Generate mutations
    mutations = generate_mutations()
    n_positions = CDR3_END - CDR3_START + 1
    print(f"\n  {len(mutations)} mutations across {n_positions} CDR3 positions")

    # Build variant sequences for TEMPRO
    variant_seqs = []
    variant_labels = []
    for mut in mutations:
        wt, pos_str, mt = mut[0], mut[1:-1], mut[-1]
        pos = int(pos_str) - 1  # 0-indexed
        var_seq = NANOBODY_SEQ[:pos] + mt + NANOBODY_SEQ[pos + 1 :]
        variant_seqs.append(var_seq)
        variant_labels.append(mut)

    # Steps 2-4: Score with all three models
    # Run SPURS and ThermoMPNN concurrently (both need structure, independent)
    spurs_task = step_spurs(pdb_str, mutations)
    thermo_task = step_thermompnn(pdb_str, mutations)
    spurs_ddg, thermo_ddg = await asyncio.gather(spurs_task, thermo_task)

    # TEMPRO: score variant sequences (needs pipeline, run after structure models)
    # Include wild-type for reference
    all_seqs = [NANOBODY_SEQ] + variant_seqs
    all_labels = ["WT"] + variant_labels
    tempro_tm = await step_tempro(all_seqs, all_labels)

    wt_tm = tempro_tm.get("WT", float("nan"))

    # ------------------------------------------------------------------
    # Combine results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RESULTS: Stability Landscape")
    print("=" * 70)

    rows = []
    for mut in mutations:
        wt_aa, pos_str, mt_aa = mut[0], mut[1:-1], mut[-1]
        pos = int(pos_str)
        rows.append({
            "mutation": mut,
            "position": pos,
            "wt_aa": wt_aa,
            "mt_aa": mt_aa,
            "spurs_ddg": spurs_ddg.get(mut),
            "thermompnn_ddg": thermo_ddg.get(mut),
            "tempro_tm": tempro_tm.get(mut),
        })

    df = pd.DataFrame(rows)

    # Add derived columns
    df["delta_tm"] = df["tempro_tm"] - wt_tm
    # Consensus: average of the two ddG predictors (lower = more stabilizing)
    df["consensus_ddg"] = df[["spurs_ddg", "thermompnn_ddg"]].mean(axis=1)

    print(f"\n  Wild-type Tm (TEMPRO): {wt_tm:.1f} C")
    print(f"  Mutations scored: {len(df)}")
    print(f"  SPURS coverage:      {df['spurs_ddg'].notna().sum()}/{len(df)}")
    print(f"  ThermoMPNN coverage: {df['thermompnn_ddg'].notna().sum()}/{len(df)}")
    print(f"  TEMPRO coverage:     {df['tempro_tm'].notna().sum()}/{len(df)}")

    # ------------------------------------------------------------------
    # Top stabilizing mutations (consensus ddG < 0 = stabilizing)
    # ------------------------------------------------------------------
    stabilizing = df[df["consensus_ddg"] < 0].sort_values("consensus_ddg")
    print(f"\n  Stabilizing mutations (consensus ddG < 0): {len(stabilizing)}")

    if len(stabilizing) > 0:
        print(f"\n  {'Mut':>6}  {'SPURS':>7}  {'Thermo':>7}  {'Consens':>7}  {'Tm':>6}  {'dTm':>6}")
        print(f"  {'---':>6}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>6}  {'---':>6}")
        for _, row in stabilizing.head(20).iterrows():
            spurs = f"{row['spurs_ddg']:>7.3f}" if pd.notna(row["spurs_ddg"]) else "    N/A"
            thermo = f"{row['thermompnn_ddg']:>7.3f}" if pd.notna(row["thermompnn_ddg"]) else "    N/A"
            cons = f"{row['consensus_ddg']:>7.3f}" if pd.notna(row["consensus_ddg"]) else "    N/A"
            tm = f"{row['tempro_tm']:>6.1f}" if pd.notna(row["tempro_tm"]) else "   N/A"
            dtm = f"{row['delta_tm']:>6.1f}" if pd.notna(row["delta_tm"]) else "   N/A"
            print(f"  {row['mutation']:>6}  {spurs}  {thermo}  {cons}  {tm}  {dtm}")

    # ------------------------------------------------------------------
    # Most destabilizing
    # ------------------------------------------------------------------
    destabilizing = df.sort_values("consensus_ddg", ascending=False)
    print(f"\n  Most destabilizing (top 10):")
    print(f"  {'Mut':>6}  {'SPURS':>7}  {'Thermo':>7}  {'Consens':>7}  {'Tm':>6}  {'dTm':>6}")
    print(f"  {'---':>6}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>6}  {'---':>6}")
    for _, row in destabilizing.head(10).iterrows():
        spurs = f"{row['spurs_ddg']:>7.3f}" if pd.notna(row["spurs_ddg"]) else "    N/A"
        thermo = f"{row['thermompnn_ddg']:>7.3f}" if pd.notna(row["thermompnn_ddg"]) else "    N/A"
        cons = f"{row['consensus_ddg']:>7.3f}" if pd.notna(row["consensus_ddg"]) else "    N/A"
        tm = f"{row['tempro_tm']:>6.1f}" if pd.notna(row["tempro_tm"]) else "   N/A"
        dtm = f"{row['delta_tm']:>6.1f}" if pd.notna(row["delta_tm"]) else "   N/A"
        print(f"  {row['mutation']:>6}  {spurs}  {thermo}  {cons}  {tm}  {dtm}")

    # ------------------------------------------------------------------
    # Per-position summary
    # ------------------------------------------------------------------
    print(f"\n  Per-position average consensus ddG:")
    pos_summary = df.groupby("position").agg(
        mean_ddg=("consensus_ddg", "mean"),
        min_ddg=("consensus_ddg", "min"),
        best_mut=("consensus_ddg", "idxmin"),
    ).reset_index()
    pos_summary["best_mut"] = pos_summary["best_mut"].map(lambda idx: df.loc[idx, "mutation"])
    pos_summary = pos_summary.sort_values("mean_ddg")

    print(f"  {'Pos':>5}  {'WT':>3}  {'Mean ddG':>9}  {'Best ddG':>9}  {'Best Mut':>9}")
    for _, row in pos_summary.iterrows():
        pos = int(row["position"])
        wt_aa = NANOBODY_SEQ[pos - 1]
        print(f"  {pos:>5}  {wt_aa:>3}  {row['mean_ddg']:>9.3f}  {row['min_ddg']:>9.3f}  {row['best_mut']:>9}")

    # ------------------------------------------------------------------
    # Correlation between predictors
    # ------------------------------------------------------------------
    both = df.dropna(subset=["spurs_ddg", "thermompnn_ddg"])
    if len(both) > 5:
        corr = both["spurs_ddg"].corr(both["thermompnn_ddg"])
        print(f"\n  SPURS vs ThermoMPNN correlation: r={corr:.3f} (n={len(both)})")

    both_tm = df.dropna(subset=["consensus_ddg", "delta_tm"])
    if len(both_tm) > 5:
        corr_tm = both_tm["consensus_ddg"].corr(both_tm["delta_tm"])
        print(f"  Consensus ddG vs delta-Tm correlation: r={corr_tm:.3f} (n={len(both_tm)})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_csv = OUTPUT_DIR / "stability_landscape.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved {len(df)} mutations → {out_csv}")

    print(f"\n{'=' * 70}")
    print("Pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
