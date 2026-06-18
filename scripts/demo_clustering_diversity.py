"""Clustering + diversity demo: ESM2 embeddings → K-means → diversity sampling → PCA plot.

Demonstrates:
  1. ESM2-8m embeddings on 50 sequences from 3 protein families
  2. ClusteringStage with K-means (k=3)
  3. Cluster size + silhouette score
  4. DiversitySamplingFilter to select diverse representatives
  5. PCA visualization colored by cluster

Usage:
    export BIOLMAI_TOKEN=<your_token>
    python scripts/demo_clustering_diversity.py
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np

from biolm.pipeline import (
    DataPipeline,
    DiversitySamplingFilter,
    DuckDBDataStore,
    EmbeddingSpec,
    ValidAminoAcidFilter,
)

TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    print("ERROR: BIOLMAI_TOKEN not set.")
    sys.exit(1)

OUTPUT_DIR = Path("outputs/clustering_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 50 sequences from 3 families: AMPs, defensins, designed helices
SEQUENCES = [
    # ── Family 1: Antimicrobial peptides (magainins, cecropins) ──
    "GIGKFLHSAKKFGKAFVGEIMNS",        # Magainin 2
    "GIGKFLHSAGKFGKAFVGEIMKS",        # Magainin 1 analog
    "GLFDIIKKIAESF",                   # Aurein 1.2
    "GLFDIVKKVVGALGSL",               # Aurein 2.2
    "FLPLILRKIVTAL",                   # Citropin 1.1
    "GIKKFLGSIWKFIKAFVKEIMN",         # MSI-78 (pexiganan)
    "GIGAVLKVLTTGLPALISWIKRKRQQ",     # Melittin
    "RWKIFKKIEKVGRNVRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin B
    "KWKLFKKIPKFLHLAKKF",             # Cecropin-melittin hybrid
    "GIGKFLHSAK",                      # Magainin N-term
    "GIKKFLGSIWKFIK",                  # MSI-78 fragment
    "GIGKFLHSAKKFGK",                  # Magainin 2 N-half
    "GLFDIVKKVVGAL",                   # Aurein fragment
    "GLFDIIKKIAESFLPK",               # Aurein 1.2 extended
    "GIGAVLKVLTTGLPALISW",            # Melittin fragment
    "RWKIFKKIEKVGRNVRD",              # Cecropin A fragment
    # ── Family 2: Defensin-like (cysteine-rich, beta-sheet) ──
    "ACYCRIPACIAGERRYGTCIYQGRLWAFCC", # HBD-1 analog
    "DHYNCVSSGGQCLYSACPIFTKIQGTCYRGKAKCCK",  # HNP-1 analog
    "CKVWGKLCRTRGCTTTHCRRH",          # Protegrin analog
    "GFCWYVNAAAHCGKRFNRVCYRN",       # Plectasin analog
    "RRLCRIVVIRVCR",                   # Tachyplesin analog
    "RCICTTRGCRKKCIHDRR",             # Defensin variant A
    "ACYCRIPACFAGERRYGTCIY",          # HBD fragment
    "DHYNCVSSGGQCLYSACPIFTK",         # HNP fragment
    "GFCWYVNAAAHCGKRFN",             # Plectasin fragment
    "CKVWGKLCRTRGCTTTH",              # Protegrin fragment
    "ACYCRIPACIAGERRY",               # HBD short
    "RRLCRIVVIRVCRGK",                # Tachyplesin extended
    "RCICTTRGCRKKCIH",                # Defensin variant B
    "DHYNCVSSGGQCLYSAC",              # HNP short
    "GFCWYVNAAAHCGK",                 # Plectasin short
    "CKVWGKLCRTRGC",                   # Protegrin short
    # ── Family 3: Designed helical peptides (alpha-helix, amphipathic) ──
    "KLAKLAKKLAKLAK",                  # (KLA)3
    "LKLLKKLLKLLKKL",                  # Designed AMP
    "RRWWRRWWRR",                      # Synthetic arginine-rich
    "KWKWKWKWKW",                      # KW repeat
    "RRWQWR",                          # Lactoferricin fragment
    "RWRWRW",                          # RW repeat
    "KFLKKAKKFGK",                     # Magainin fragment
    "KWKLFKKI",                        # Short cecropin
    "RLFDKIRQ",                        # Indolicidin short
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # LL-37
    "RLFDKIRQVIRKF",                   # Indolicidin analog
    "FKRIVQRIKDFL",                    # LL-37 fragment
    "KLAKLAKKLAKLAKKLA",              # (KLA)4 extended
    "LKLLKKLLKLLKKLLK",               # Designed AMP extended
    "RRWWRRWWRRWW",                    # Arg-rich extended
    "KWKWKWKWKWKW",                    # KW repeat extended
]


async def main():
    db_path = OUTPUT_DIR / "pipeline.duckdb"
    ds = DuckDBDataStore(db_path)

    try:
        pipeline = DataPipeline(
            sequences=SEQUENCES,
            datastore=ds,
            run_id="cluster_demo_v1",
            output_dir=str(OUTPUT_DIR),
            verbose=True,
        )

        # 1. Validate amino acids
        pipeline.add_filter(ValidAminoAcidFilter(verbose=True), stage_name="validate")

        # 2. Embed with ESM2-8m (fast)
        pipeline.add_prediction(
            "esm2-8m",
            action="encode",
            embedding_extractor=EmbeddingSpec(key="embeddings", layer=6),
            stage_name="embed_esm2",
            depends_on=["validate"],
            batch_size=16,
            max_concurrent=4,
        )

        # 3. Cluster into 3 groups
        pipeline.add_clustering(
            method="kmeans",
            n_clusters=3,
            similarity_metric="embedding",
            embedding_model="esm2-8m",
            stage_name="cluster_k3",
            depends_on=["embed_esm2"],
        )

        # 4. Diversity sampling — pick representative subset
        pipeline.add_filter(
            DiversitySamplingFilter(
                n_samples=20,
                method="random",
            ),
            stage_name="diversity_select",
        )

        print("\n" + "=" * 60)
        print("  CLUSTERING DEMO: Embed → Cluster → Diversity Sample")
        print("=" * 60)

        stage_results = await pipeline.run_async()

        # ── Results ─────────────────────────────────────────────────────
        print("\n\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)

        print("\n── summary() ──")
        print(pipeline.summary().to_string(index=False))

        df_final = pipeline.get_final_data()
        print(f"\n{len(df_final)} sequences selected as diverse representatives")

        # Cluster stats (from pre-filter data)
        cluster_data = pipeline.query("""
            SELECT p.value AS cluster_id, COUNT(*) AS n_sequences
            FROM predictions p
            WHERE p.prediction_type = 'cluster_k3'
            GROUP BY p.value
            ORDER BY p.value
        """)
        print("\n── Cluster sizes ──")
        print(cluster_data.to_string(index=False))

        # ── PCA visualization ──────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score

            # Get all embeddings + cluster labels
            all_seq_ids = ds.conn.execute(
                "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = 'esm2-8m'"
            ).df()["sequence_id"].tolist()

            emb_map = ds.get_embeddings_bulk(all_seq_ids, model_name="esm2-8m")
            if len(emb_map) >= 3:
                ordered_ids = [sid for sid in all_seq_ids if sid in emb_map]
                mat = np.stack([emb_map[sid] for sid in ordered_ids])

                # Get cluster labels
                cluster_df = ds.conn.execute("""
                    SELECT sequence_id, CAST(value AS INTEGER) AS cluster_id
                    FROM predictions
                    WHERE prediction_type = 'cluster_k3'
                """).df()
                cluster_map = dict(zip(
                    cluster_df["sequence_id"], cluster_df["cluster_id"]
                ))
                labels = np.array([cluster_map.get(sid, -1) for sid in ordered_ids])

                # Silhouette score
                valid = labels >= 0
                if valid.sum() >= 3 and len(set(labels[valid])) >= 2:
                    sil = silhouette_score(mat[valid], labels[valid])
                    print(f"\n── Silhouette score: {sil:.3f} ──")

                # PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(mat)

                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=labels, cmap="Set1", alpha=0.7, s=60, edgecolors="black", linewidth=0.5,
                )
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                ax.set_title("Peptide Clusters (ESM2-8m embeddings, K-means k=3)")
                plt.colorbar(scatter, ax=ax, label="Cluster")
                plt.tight_layout()

                plot_path = OUTPUT_DIR / "cluster_pca.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                print(f"\nSaved PCA plot → {plot_path}")
                plt.close()

                # Highlight diverse representatives
                diverse_ids = set(df_final["sequence_id"].tolist())
                is_diverse = np.array([sid in diverse_ids for sid in ordered_ids])

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(
                    coords[~is_diverse, 0], coords[~is_diverse, 1],
                    c="lightgray", alpha=0.4, s=40, label="All",
                )
                ax.scatter(
                    coords[is_diverse, 0], coords[is_diverse, 1],
                    c=labels[is_diverse], cmap="Set1", alpha=0.9, s=80,
                    edgecolors="black", linewidth=1, label="Diverse subset",
                )
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                ax.set_title("Diversity-Sampled Representatives (20 of 50)")
                ax.legend()
                plt.tight_layout()

                plot_path2 = OUTPUT_DIR / "diversity_selection.png"
                plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
                print(f"Saved diversity plot → {plot_path2}")
                plt.close()

        except ImportError as e:
            print(f"\nSkipping plots ({e})")

        # ── Export ─────────────────────────────────────────────────────
        out_csv = OUTPUT_DIR / "diverse_peptides.csv"
        save_cols = [c for c in df_final.columns if c != "hash"]
        df_final[save_cols].to_csv(out_csv, index=False)
        print(f"\nExported {len(df_final)} diverse peptides → {out_csv}")

    finally:
        ds.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
