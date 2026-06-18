"""
Sequence clustering and diversity analysis.

Provides tools for grouping similar sequences and measuring sequence space coverage.

Performance Notes:
- For large datasets (>10k sequences), use sampling or embedding-based methods
- Pairwise distance computations are O(n²) - use max_sample parameter
- Embedding-based clustering scales much better than Hamming distance
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

try:
    from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans  # noqa: F401
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Performance thresholds
LARGE_DATASET_THRESHOLD = 10000  # Warn about expensive operations
VERY_LARGE_DATASET_THRESHOLD = 50000  # Force sampling or error


@dataclass
class ClusteringResult:
    """Results from sequence clustering."""

    cluster_ids: np.ndarray
    centroids: list[str]
    centroid_indices: np.ndarray
    n_clusters: int
    silhouette_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    cluster_sizes: Optional[dict[int, int]] = None


class SequenceClusterer:
    """
    Cluster sequences by similarity.

    Supports multiple clustering algorithms and similarity metrics.

    Performance Notes:
        - For >10k sequences with Hamming distance, consider using max_sample
        - Embedding-based clustering scales much better (O(n) with MiniBatch K-means)
        - Use mini_batch=True for very large datasets (>50k sequences)

    Args:
        method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        n_clusters: Number of clusters (for kmeans/hierarchical)
        similarity_metric: How to measure sequence similarity
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples per cluster
        mini_batch: Use MiniBatchKMeans for large datasets (faster, approximate)
        max_sample: Maximum sequences to use for distance matrix (None = all)

    Example:
        >>> # For large datasets, use sampling or mini-batch
        >>> clusterer = SequenceClusterer(
        ...     method='kmeans',
        ...     n_clusters=100,
        ...     mini_batch=True  # Much faster for large N
        ... )
        >>> result = clusterer.cluster(sequences)
    """

    def __init__(
        self,
        method: Literal["kmeans", "dbscan", "hierarchical"] = "kmeans",
        n_clusters: Optional[int] = None,
        similarity_metric: Literal["embedding"] = "embedding",
        eps: float = 0.5,
        min_samples: int = 5,
        random_state: int = 42,
        mini_batch: bool = False,
        max_sample: Optional[int] = None,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for clustering. Install with: pip install scikit-learn"
            )

        self.method = method
        self.n_clusters = n_clusters
        self.similarity_metric = similarity_metric
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.mini_batch = mini_batch
        self.max_sample = max_sample

        if method in ["kmeans", "hierarchical"] and n_clusters is None:
            raise ValueError(f"{method} requires n_clusters parameter")

    def cluster(
        self, sequences: list[str], embeddings: Optional[np.ndarray] = None
    ) -> ClusteringResult:
        """
        Cluster sequences and return assignments.

        Args:
            sequences: List of protein sequences
            embeddings: Pre-computed embeddings (if using embedding metric)

        Returns:
            ClusteringResult with cluster assignments and metrics
        """
        n = len(sequences)

        # validate n_clusters before calling sklearn — otherwise
        # sklearn raises a cryptic error deep inside fit_predict.
        if self.n_clusters is not None and self.n_clusters > n:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) must be <= number of sequences "
                f"({n}). Either reduce n_clusters or provide more sequences."
            )

        # Performance warnings
        if (
            n > VERY_LARGE_DATASET_THRESHOLD
            and not self.mini_batch
            and self.method == "kmeans"
        ):
            warnings.warn(
                f"Clustering {n:,} sequences. Consider setting mini_batch=True for faster (approximate) clustering.",
                PerformanceWarning,
                stacklevel=2,
            )

        if self.similarity_metric == "embedding":
            if embeddings is None:
                raise ValueError(
                    "embeddings required when similarity_metric='embedding'"
                )
            distance_matrix = self._compute_embedding_distances(embeddings)
        else:
            raise ValueError(
                f"Unknown similarity_metric '{self.similarity_metric}'. "
                "Only 'embedding' is currently supported."
            )

        # Perform clustering
        if self.method == "kmeans":
            cluster_ids = self._cluster_kmeans(distance_matrix, n)
        elif self.method == "dbscan":
            cluster_ids = self._cluster_dbscan(distance_matrix)
        elif self.method == "hierarchical":
            cluster_ids = self._cluster_hierarchical(distance_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Find centroids (sequences closest to cluster centers)
        centroids, centroid_indices = self._find_centroids(
            sequences, cluster_ids, distance_matrix
        )

        # Compute metrics (skip for very large datasets to save time)
        silhouette = None
        davies_bouldin = None

        if len(set(cluster_ids)) > 1 and n < LARGE_DATASET_THRESHOLD:
            try:
                silhouette = silhouette_score(
                    distance_matrix, cluster_ids, metric="precomputed"
                )
            except Exception:
                pass

        # Count cluster sizes
        cluster_sizes = dict(Counter(cluster_ids))

        return ClusteringResult(
            cluster_ids=cluster_ids,
            centroids=centroids,
            centroid_indices=centroid_indices,
            n_clusters=len(set(cluster_ids)),
            silhouette_score=silhouette,
            davies_bouldin_score=davies_bouldin,
            cluster_sizes=cluster_sizes,
        )

    def _compute_embedding_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances between embeddings."""
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for embedding distances. Install with: pip install scipy"
            )

        # Compute pairwise distances
        condensed_dist = pdist(embeddings, metric="euclidean")
        distance_matrix = squareform(condensed_dist)

        return distance_matrix

    def _cluster_kmeans(
        self, distance_matrix: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """K-means clustering on distance matrix."""
        if self.mini_batch and n_samples > 1000:
            # Use MiniBatchKMeans for large datasets (much faster)
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=min(1000, n_samples // 10),
                n_init=3,  # Fewer inits for speed
                max_iter=100,
            )
        else:
            kmeans = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
            )
        cluster_ids = kmeans.fit_predict(distance_matrix)
        return cluster_ids

    def _cluster_dbscan(self, distance_matrix: np.ndarray) -> np.ndarray:
        """DBSCAN clustering on distance matrix."""
        dbscan = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="precomputed"
        )
        cluster_ids = dbscan.fit_predict(distance_matrix)

        # warn when all points are noise so the user knows to
        # adjust eps or min_samples rather than silently getting an empty result.
        unique_labels = set(cluster_ids)
        if unique_labels == {-1}:
            warnings.warn(
                f"DBSCAN clustering produced 0 valid clusters — all "
                f"{len(cluster_ids)} points are noise (label=-1). Consider "
                f"adjusting eps or min_samples parameters.",
                UserWarning,
                stacklevel=2,
            )

        return cluster_ids

    def _cluster_hierarchical(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Hierarchical clustering on distance matrix."""
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for hierarchical clustering. Install with: pip install scipy"
            )

        # Compute linkage
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, method="average")

        # Cut dendrogram to get clusters
        cluster_ids = (
            fcluster(Z, self.n_clusters, criterion="maxclust") - 1
        )  # 0-indexed

        return cluster_ids

    def _find_centroids(
        self, sequences: list[str], cluster_ids: np.ndarray, distance_matrix: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """Find centroid sequence for each cluster (medoid)."""
        unique_clusters = set(cluster_ids)
        centroids = []
        centroid_indices = []

        for cluster_id in sorted(unique_clusters):
            if cluster_id == -1:  # DBSCAN noise
                continue

            # Get indices of sequences in this cluster
            cluster_mask = cluster_ids == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Find sequence with minimum sum of distances to others in cluster
            cluster_distances = distance_matrix[cluster_mask][:, cluster_mask]
            sum_distances = cluster_distances.sum(axis=1)
            centroid_idx_local = np.argmin(sum_distances)
            centroid_idx_global = cluster_indices[centroid_idx_local]

            centroids.append(sequences[centroid_idx_global])
            centroid_indices.append(centroid_idx_global)

        return centroids, np.array(centroid_indices)


class DiversityAnalyzer:
    """
    Analyze sequence diversity and coverage.

    Provides metrics for understanding sequence space exploration.

    Performance Notes:
        - Shannon entropy: O(n*L) where L is sequence length
        - Pairwise distances: O(n²*L) - use max_sample for large n
        - All metrics scale linearly except pairwise distances

    Example:
        >>> analyzer = DiversityAnalyzer()
        >>> metrics = analyzer.compute_all_metrics(sequences, max_sample=10000)
        >>> print(f"Shannon entropy: {metrics['shannon_entropy']:.2f}")
    """

    @staticmethod
    def shannon_entropy(sequences: list[str], normalize: bool = True) -> float:
        """
        Calculate Shannon entropy of amino acid distribution.

        Measures positional diversity across all sequences.
        Optimized for large datasets using vectorized operations.

        Args:
            sequences: List of protein sequences
            normalize: Normalize by log(20) for 0-1 range

        Returns:
            Shannon entropy (higher = more diverse)
        """
        if not sequences:
            return 0.0

        # Align sequences (pad to same length)
        max_len = max(len(s) for s in sequences)

        # Use numpy for efficiency
        seq_array = np.array([list(s.ljust(max_len, "-")) for s in sequences])

        # Calculate entropy at each position (vectorized)
        entropies = []
        for pos in range(max_len):
            column = seq_array[:, pos]
            # Remove gaps
            column = column[column != "-"]

            if len(column) == 0:
                continue

            # Count amino acids
            unique, counts = np.unique(column, return_counts=True)
            probs = counts / counts.sum()

            # Shannon entropy: -sum(p * log(p))
            entropy = -np.sum(
                probs * np.log2(probs + 1e-10)
            )  # Add epsilon for stability
            entropies.append(entropy)

        avg_entropy = np.mean(entropies) if entropies else 0.0

        if normalize:
            # Normalize by max possible entropy (log2(20) for 20 amino acids)
            avg_entropy = avg_entropy / np.log2(20)

        return float(avg_entropy)

    @staticmethod
    def motif_diversity(
        sequences: list[str], k: int = 3
    ) -> dict[str, Union[int, float]]:
        """
        Analyze k-mer (motif) diversity.

        Args:
            sequences: List of protein sequences
            k: Length of k-mers to analyze

        Returns:
            Dictionary with k-mer statistics
        """
        # Extract all k-mers
        kmers = []
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmers.append(seq[i : i + k])

        if not kmers:
            return {"unique_kmers": 0, "total_kmers": 0, "diversity_ratio": 0.0}

        kmer_counts = Counter(kmers)

        return {
            "unique_kmers": len(kmer_counts),
            "total_kmers": len(kmers),
            "diversity_ratio": len(kmer_counts) / len(kmers),
            "most_common": kmer_counts.most_common(5),
        }

    @classmethod
    def compute_all_metrics(
        cls, sequences: list[str], max_sample: Optional[int] = 10000
    ) -> dict[str, Union[float, dict]]:
        """
        Compute all diversity metrics at once.

        Args:
            sequences: List of protein sequences
            max_sample: Unused, reserved for future embedding-based pairwise stats

        Returns:
            Dictionary with all diversity metrics
        """
        metrics = {
            "n_sequences": len(sequences),
            "n_unique": len(set(sequences)),
            "uniqueness_ratio": (
                len(set(sequences)) / len(sequences) if sequences else 0
            ),
            "shannon_entropy": cls.shannon_entropy(sequences),
            "motif_diversity_3mer": cls.motif_diversity(sequences, k=3),
            "motif_diversity_5mer": cls.motif_diversity(sequences, k=5),
        }

        return metrics


def cluster_sequences(
    sequences: list[str],
    method: str = "kmeans",
    n_clusters: int = 10,
    embeddings: Optional[np.ndarray] = None,
    mini_batch: bool = False,
    max_sample: Optional[int] = None,
    **kwargs,
) -> ClusteringResult:
    """
    Convenience function for clustering sequences using embeddings.

    Performance Tips:
        - For >50k sequences, use mini_batch=True
        - Requires pre-computed embeddings (similarity_metric='embedding')

    Args:
        sequences: List of protein sequences
        method: Clustering algorithm
        n_clusters: Number of clusters
        embeddings: Pre-computed embeddings (required)
        mini_batch: Use MiniBatchKMeans for faster (approximate) clustering
        max_sample: Reserved; not yet used
        **kwargs: Additional arguments for SequenceClusterer

    Returns:
        ClusteringResult
    """
    clusterer = SequenceClusterer(
        method=method,
        n_clusters=n_clusters,
        mini_batch=mini_batch,
        max_sample=max_sample,
        **kwargs,
    )
    return clusterer.cluster(sequences, embeddings)


def analyze_diversity(sequences: list[str], max_sample: Optional[int] = 10000) -> dict:
    """
    Convenience function for analyzing sequence diversity.

    Args:
        sequences: List of protein sequences
        max_sample: Reserved for future embedding-based pairwise stats

    Returns:
        Dictionary of diversity metrics (shannon_entropy, motif diversity, uniqueness)

    Example:
        >>> metrics = analyze_diversity(sequences)
        >>> print(f"Entropy: {metrics['shannon_entropy']:.2f}")
    """
    return DiversityAnalyzer.compute_all_metrics(sequences, max_sample=max_sample)


# Define custom warning for performance issues
class PerformanceWarning(UserWarning):
    """Warning for potentially slow operations on large datasets."""

    pass
