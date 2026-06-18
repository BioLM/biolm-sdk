"""Peptide embedding pipeline: ESM2 + ESM-C multi-model embeddings with PCA and filtering.

Demonstrates:
  1. Multi-model embedding (ESM2-8m, ESM2-650M, ESM-C 300M, ESM-C 600M)
  2. Embedding-based PCA (per-model and concatenated)
  3. DiversitySamplingFilter on embedding-derived scores
  4. get_embeddings_concat() for fused multi-model representations

Peptide set: 30 diverse antimicrobial peptides (AMPs) of varying length and charge.

Usage:
    export BIOLMAI_TOKEN=<your_token>
    python scripts/pipeline_peptide_embeddings.py
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from biolm.pipeline import (
    DataPipeline,
    DuckDBDataStore,
    EmbeddingSpec,
    RankingFilter,
    ValidAminoAcidFilter,
)

TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    print("ERROR: BIOLMAI_TOKEN not set.")
    sys.exit(1)

OUTPUT_DIR = Path("outputs/peptide_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 30 antimicrobial peptides — diverse lengths, charges, hydrophobicities
# ---------------------------------------------------------------------------
PEPTIDES = [
    # Magainins / frog-derived
    "GIGKFLHSAKKFGKAFVGEIMNS",        # Magainin 2
    "GIGKFLHSAGKFGKAFVGEIMKS",        # Magainin 1 analog
    "GLFDIIKKIAESF",                   # Aurein 1.2
    "GLFDIVKKVVGALGSL",               # Aurein 2.2
    "FLPLILRKIVTAL",                   # Citropin 1.1
    # Human defensins / cathelicidins
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # LL-37
    "RLFDKIRQVIRKF",                   # Indolicidin analog
    "KWKLFKKIPKFLHLAKKF",             # Cecropin-melittin hybrid
    "ACYCRIPACIAGERRYGTCIYQGRLWAFCC", # HBD-1 analog
    "DHYNCVSSGGQCLYSACPIFTKIQGTCYRGKAKCCK",  # HNP-1 analog
    # Insect-derived
    "RWKIFKKIEKVGRNVRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin B
    "GIGAVLKVLTTGLPALISWIKRKRQQ",     # Melittin
    "VDKGSYLPRPTPPRPIYNRN",           # Apidaecin
    "GKPRPYSPRPTSHPRPIRV",            # Drosocin
    # Synthetic / designed
    "KLAKLAKKLAKLAK",                  # (KLA)₃ — designed helix
    "LKLLKKLLKLLKKL",                  # Designed AMP
    "RRWWRRWWRR",                      # Synthetic arginine-rich
    "KWKWKWKWKW",                      # KW repeat
    "GIKKFLGSIWKFIKAFVKEIMN",         # MSI-78 (pexiganan)
    # Cyclic / constrained
    "CKVWGKLCRTRGCTTTHCRRH",          # Protegrin analog
    "RRLCRIVVIRVCR",                   # Tachyplesin analog
    "GFCWYVNAAAHCGKRFNRVCYRN",       # Plectasin analog
    # Short peptides
    "RRWQWR",                          # Lactoferricin fragment
    "RWRWRW",                          # RW repeat
    "FKRIVQRIKDFL",                    # LL-37 fragment
    "KFLKKAKKFGK",                     # Magainin fragment
    "GIGKFLHSAK",                      # Magainin N-term
    "KWKLFKKI",                        # Short cecropin
    "RLFDKIRQ",                        # Indolicidin short
]

# Models to embed — each returns different representation dimension
EMBEDDING_MODELS = [
    ("esm2-8m",   "embeddings", 6),    # key, default layer
    ("esm2-650m", "embeddings", 33),
    ("esmc-300m", "embeddings", 29),
    ("esmc-600m", "embeddings", 35),
]


async def main():
    db_path = OUTPUT_DIR / "pipeline.duckdb"
    ds = DuckDBDataStore(db_path)

    try:
        pipeline = DataPipeline(
            sequences=PEPTIDES,
            datastore=ds,
            run_id="peptide_emb_v1",
            output_dir=str(OUTPUT_DIR),
            verbose=True,
        )

        # ------------------------------------------------------------------
        # Stage 0: Validate amino acids
        # ------------------------------------------------------------------
        pipeline.add_filter(
            ValidAminoAcidFilter(verbose=True),
            stage_name="filter_valid",
        )

        # ------------------------------------------------------------------
        # Stage 1-4: Embed with all 4 models (parallel — all depend on filter_valid)
        # ------------------------------------------------------------------
        for model_name, key, layer in EMBEDDING_MODELS:
            pipeline.add_prediction(
                model_name=model_name,
                action="encode",
                embedding_extractor=EmbeddingSpec(key=key, layer=layer),
                stage_name=f"embed_{model_name.replace('-', '_')}",
                depends_on=["filter_valid"],
                batch_size=8,
                max_concurrent=4,
            )

        # ------------------------------------------------------------------
        # Run
        # ------------------------------------------------------------------
        print("\n=== Peptide Embedding Pipeline ===")
        print(f"  Peptides: {len(PEPTIDES)}")
        print(f"  Models:   {', '.join(m for m, _, _ in EMBEDDING_MODELS)}")
        print()

        stage_results = await pipeline.run_async()

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n=== Stage Summary ===")
        for name, result in stage_results.items():
            print(f"  {result}")

        df_final = pipeline.get_final_data()
        print(f"\n{len(df_final)} sequences with embeddings from all models")

        if len(df_final) == 0:
            print("No sequences survived — check stage results above.")
            return

        seq_ids = df_final["sequence_id"].tolist()

        # ------------------------------------------------------------------
        # Per-model embedding retrieval + stats
        # ------------------------------------------------------------------
        print("\n=== Embedding Dimensions ===")
        model_embeddings: dict[str, dict[int, np.ndarray]] = {}
        for model_name, _, _ in EMBEDDING_MODELS:
            emb_map = ds.get_embeddings_bulk(seq_ids, model_name=model_name)
            model_embeddings[model_name] = emb_map
            if emb_map:
                dim = next(iter(emb_map.values())).shape[0]
                print(f"  {model_name}: {len(emb_map)} seqs × {dim} dims")
            else:
                print(f"  {model_name}: no embeddings found")

        # ------------------------------------------------------------------
        # Concatenated embeddings
        # ------------------------------------------------------------------
        model_names = [m for m, _, _ in EMBEDDING_MODELS]
        concat_map = ds.get_embeddings_concat(seq_ids, model_names)
        if concat_map:
            concat_dim = next(iter(concat_map.values())).shape[0]
            print(f"\n  Concatenated ({'+'.join(model_names)}): {len(concat_map)} seqs × {concat_dim} dims")
        else:
            print("\n  Concatenated: no sequences with ALL model embeddings")

        # ------------------------------------------------------------------
        # PCA — per model
        # ------------------------------------------------------------------
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("\nsklearn not installed — skipping PCA")
            return

        print("\n=== PCA (per model) ===")
        for model_name, _, _ in EMBEDDING_MODELS:
            emb_map = model_embeddings[model_name]
            if len(emb_map) < 3:
                print(f"  {model_name}: too few embeddings for PCA")
                continue
            # Align with df_final ordering
            ids_with_emb = [sid for sid in seq_ids if sid in emb_map]
            mat = np.stack([emb_map[sid] for sid in ids_with_emb])
            pca = PCA(n_components=min(3, len(ids_with_emb)))
            coords = pca.fit_transform(mat)
            var = pca.explained_variance_ratio_
            print(f"  {model_name}: PC1={var[0]:.1%}  PC2={var[1]:.1%}"
                  + (f"  PC3={var[2]:.1%}" if len(var) > 2 else ""))
            # Store PC1/PC2 as columns for downstream filtering
            pc_df = pd.DataFrame({
                "sequence_id": ids_with_emb,
                f"{model_name}_pc1": coords[:, 0],
                f"{model_name}_pc2": coords[:, 1],
            })
            df_final = df_final.merge(pc_df, on="sequence_id", how="left")

        # ------------------------------------------------------------------
        # PCA — concatenated
        # ------------------------------------------------------------------
        if len(concat_map) >= 3:
            print("\n=== PCA (concatenated) ===")
            ids_concat = [sid for sid in seq_ids if sid in concat_map]
            mat_concat = np.stack([concat_map[sid] for sid in ids_concat])
            pca_concat = PCA(n_components=min(3, len(ids_concat)))
            coords_concat = pca_concat.fit_transform(mat_concat)
            var_c = pca_concat.explained_variance_ratio_
            print(f"  Concat: PC1={var_c[0]:.1%}  PC2={var_c[1]:.1%}"
                  + (f"  PC3={var_c[2]:.1%}" if len(var_c) > 2 else ""))

            # Store concat PCA scores
            pc_concat_df = pd.DataFrame({
                "sequence_id": ids_concat,
                "concat_pc1": coords_concat[:, 0],
                "concat_pc2": coords_concat[:, 1],
            })
            df_final = df_final.merge(pc_concat_df, on="sequence_id", how="left")

            # ------------------------------------------------------------------
            # Embedding-based diversity: euclidean distance from centroid
            # ------------------------------------------------------------------
            centroid = mat_concat.mean(axis=0)
            dists = np.linalg.norm(mat_concat - centroid, axis=1)
            dist_df = pd.DataFrame({
                "sequence_id": ids_concat,
                "dist_from_centroid": dists,
            })
            df_final = df_final.merge(dist_df, on="sequence_id", how="left")

            # Rank by distance from centroid (most diverse = furthest)
            ranked = df_final.dropna(subset=["dist_from_centroid"]).sort_values(
                "dist_from_centroid", ascending=False
            )
            print("\n=== Top 10 Most Diverse (by centroid distance) ===")
            print(f"  {'#':>3}  {'Len':>4}  {'Dist':>8}  {'PC1':>8}  {'PC2':>8}  Sequence")
            for i, (_, row) in enumerate(ranked.head(10).iterrows()):
                print(f"  {i+1:>3}  {len(row['sequence']):>4}  "
                      f"{row['dist_from_centroid']:>8.2f}  "
                      f"{row.get('concat_pc1', float('nan')):>8.2f}  "
                      f"{row.get('concat_pc2', float('nan')):>8.2f}  "
                      f"{row['sequence'][:40]}")

        # ------------------------------------------------------------------
        # Pairwise cosine similarity matrix (concat embeddings)
        # ------------------------------------------------------------------
        if len(concat_map) >= 3:
            from sklearn.metrics.pairwise import cosine_similarity
            cos_sim = cosine_similarity(mat_concat)
            # Mask diagonal
            np.fill_diagonal(cos_sim, np.nan)
            mean_sim = np.nanmean(cos_sim, axis=1)
            print(f"\n=== Pairwise Cosine Similarity (concat) ===")
            print(f"  Mean: {np.nanmean(cos_sim):.4f}")
            print(f"  Min:  {np.nanmin(cos_sim):.4f}")
            print(f"  Max:  {np.nanmax(cos_sim):.4f}")

            # Most unique peptide (lowest mean cosine similarity)
            most_unique_idx = np.argmin(mean_sim)
            most_unique_sid = ids_concat[most_unique_idx]
            most_unique_row = df_final[df_final["sequence_id"] == most_unique_sid].iloc[0]
            print(f"\n  Most unique: {most_unique_row['sequence'][:40]}... "
                  f"(mean cos_sim={mean_sim[most_unique_idx]:.4f})")

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        out_csv = OUTPUT_DIR / "peptide_embeddings.csv"
        # Don't save embedding arrays to CSV — just scores and PCA coords
        save_cols = [c for c in df_final.columns
                     if not c.startswith("embedding") and c != "hash"]
        df_final[save_cols].to_csv(out_csv, index=False)
        print(f"\nSaved {len(df_final)} peptides → {out_csv}")

        # Save concat embedding matrix as npz for downstream use
        if concat_map:
            ids_ordered = [sid for sid in seq_ids if sid in concat_map]
            mat = np.stack([concat_map[sid] for sid in ids_ordered])
            npz_path = OUTPUT_DIR / "concat_embeddings.npz"
            np.savez_compressed(npz_path, embeddings=mat, sequence_ids=np.array(ids_ordered))
            print(f"Saved concat embeddings ({mat.shape}) → {npz_path}")

        print("\nDone!")
    finally:
        ds.close()


if __name__ == "__main__":
    asyncio.run(main())
