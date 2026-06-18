"""
Visualization utilities for pipeline results.

Provides plotting functions for:
- Pipeline funnel diagrams
- Prediction distributions
- Embedding visualizations (PCA, UMAP)
- Correlation matrices
- Temperature scan results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def plot_pipeline_funnel(
    stage_results: dict[str, Any],
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
):
    """
    Plot pipeline funnel showing sequence counts through stages.

    Args:
        stage_results: Dict mapping stage names to StageResult objects
        figsize: Figure size
        save_path: Optional path to save figure

    Example:
        >>> plot_pipeline_funnel(pipeline.stage_results)
    """
    # matplotlib imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from None

    fig, ax = plt.subplots(figsize=figsize)

    stages = list(stage_results.keys())
    # Defensive access — StageResult may be missing input_count/output_count
    input_counts = [getattr(r, 'input_count', None) or 0 for r in stage_results.values()]
    output_counts = [getattr(r, 'output_count', None) or 0 for r in stage_results.values()]

    x = np.arange(len(stages))
    width = 0.35

    ax.bar(x - width / 2, input_counts, width, label="Input", alpha=0.8)
    ax.bar(x + width / 2, output_counts, width, label="Output", alpha=0.8)

    ax.set_xlabel("Stage")
    ax.set_ylabel("Sequence Count")
    ax.set_title("Pipeline Funnel")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 50,
    figsize: tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Plot distribution of a column.

    Args:
        df: DataFrame
        column: Column name to plot
        bins: Number of bins
        figsize: Figure size
        title: Optional title
        xlabel: Optional x-axis label
        save_path: Optional path to save figure

    Example:
        >>> plot_distribution(df, 'tm', title='Tm Distribution')
    """
    # matplotlib imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from None

    fig, ax = plt.subplots(figsize=figsize)

    data = df[column].dropna()

    # Guard against all-NaN column producing a silent empty plot
    if data.empty:
        raise ValueError(
            f"No valid (non-NaN) data in column '{column}'. "
            f"Cannot plot distribution."
        )

    ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Distribution of {column}")

    # Add statistics
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
    ax.axvline(
        median_val, color="blue", linestyle="--", label=f"Median: {median_val:.2f}"
    )
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    alpha: float = 0.6,
):
    """
    Plot scatter plot of two columns.

    Args:
        df: DataFrame
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Optional column for coloring points
        figsize: Figure size
        title: Optional title
        save_path: Optional path to save figure
        alpha: Point transparency

    Example:
        >>> plot_scatter(df, 'tm', 'plddt', color_col='temperature')
    """
    # matplotlib imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from None

    fig, ax = plt.subplots(figsize=figsize)

    if color_col and color_col in df.columns:
        scatter = ax.scatter(
            df[x_col], df[y_col], c=df[color_col], alpha=alpha, cmap="viridis"
        )
        plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=alpha)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs {x_col}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
):
    """
    Plot correlation matrix heatmap.

    Args:
        df: DataFrame
        columns: Optional list of columns to include (defaults to all numeric)
        figsize: Figure size
        save_path: Optional path to save figure

    Example:
        >>> plot_correlation_matrix(df, columns=['tm', 'plddt', 'solubility'])
    """
    # matplotlib/seaborn imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "matplotlib and seaborn are required for visualization. "
            "Install them with: pip install matplotlib seaborn"
        ) from e

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_temperature_scan(
    df: pd.DataFrame,
    metric_col: str,
    temperature_col: str = "temperature",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
):
    """
    Plot results of temperature scanning.

    Args:
        df: DataFrame with temperature and metric columns
        metric_col: Column with metric to plot
        temperature_col: Column with temperature values
        figsize: Figure size
        save_path: Optional path to save figure

    Example:
        >>> plot_temperature_scan(df, 'tm', temperature_col='temperature')
    """
    # matplotlib/seaborn imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "matplotlib and seaborn are required for visualization. "
            "Install them with: pip install matplotlib seaborn"
        ) from e

    # Guard against missing temperature column before accessing it
    if temperature_col not in df.columns:
        raise ValueError(
            f"Column '{temperature_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Box plot
    df_clean = df[[temperature_col, metric_col]].dropna()
    df_clean[temperature_col] = df_clean[temperature_col].astype(str)

    sns.boxplot(data=df_clean, x=temperature_col, y=metric_col, ax=ax1)
    ax1.set_title(f"{metric_col} by Temperature")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(metric_col)

    # Violin plot
    sns.violinplot(data=df_clean, x=temperature_col, y=metric_col, ax=ax2)
    ax2.set_title(f"{metric_col} Distribution by Temperature")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(metric_col)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_components: int = 2,
    figsize: tuple[int, int] = (10, 8),
    title: str = "PCA of Embeddings",
    save_path: Optional[Path] = None,
):
    """
    Plot PCA of embeddings.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        labels: Optional array of labels for coloring
        n_components: Number of PCA components (2 or 3)
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> embeddings = np.array([...])  # Load from datastore
        >>> plot_embedding_pca(embeddings, labels=df['temperature'])
    """
    # matplotlib imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from None

    from sklearn.decomposition import PCA

    # Perform PCA
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)

        if labels is not None:
            scatter = ax.scatter(
                embeddings_pca[:, 0],
                embeddings_pca[:, 1],
                c=labels,
                alpha=0.6,
                cmap="viridis",
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.6)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title)

    elif n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if labels is not None:
            scatter = ax.scatter(
                embeddings_pca[:, 0],
                embeddings_pca[:, 1],
                embeddings_pca[:, 2],
                c=labels,
                alpha=0.6,
                cmap="viridis",
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(
                embeddings_pca[:, 0],
                embeddings_pca[:, 1],
                embeddings_pca[:, 2],
                alpha=0.6,
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return pca


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    figsize: tuple[int, int] = (10, 8),
    title: str = "UMAP of Embeddings",
    save_path: Optional[Path] = None,
):
    """
    Plot UMAP of embeddings.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        labels: Optional array of labels for coloring
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure

    Example:
        >>> embeddings = np.array([...])  # Load from datastore
        >>> plot_embedding_umap(embeddings, labels=df['temperature'])
    """
    # matplotlib imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from None

    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None

    # Perform UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(
            embeddings_umap[:, 0],
            embeddings_umap[:, 1],
            c=labels,
            alpha=0.6,
            cmap="viridis",
        )
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], alpha=0.6)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return reducer


def plot_sequence_diversity(
    df: pd.DataFrame,
    reference_sequence: Optional[str] = None,
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
):
    """
    Plot sequence diversity metrics.

    Args:
        df: DataFrame with 'sequence' column
        reference_sequence: Optional reference sequence for Hamming distance
        figsize: Figure size
        save_path: Optional path to save figure

    Example:
        >>> plot_sequence_diversity(df, reference_sequence='MKTAYIAKQRQ')
    """
    # matplotlib imported lazily to avoid blocking imports when not installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Length distribution
    lengths = df["sequence"].str.len()
    axes[0].hist(lengths, bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sequence Length Distribution")
    axes[0].axvline(
        lengths.mean(), color="red", linestyle="--", label=f"Mean: {lengths.mean():.1f}"
    )
    axes[0].legend()

    # Hamming distance distribution (if reference provided)
    if reference_sequence:
        from biolm.pipeline.filters import HammingDistanceFilter

        distances = df["sequence"].apply(
            lambda seq: HammingDistanceFilter.hamming_distance(seq, reference_sequence)
        )

        axes[1].hist(distances, bins=50, alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("Hamming Distance to Reference")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Sequence Diversity")
        axes[1].axvline(
            distances.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {distances.mean():.1f}",
        )
        axes[1].legend()
    else:
        # Unique residue frequency
        axes[1].text(
            0.5,
            0.5,
            "No reference sequence provided",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


class PipelinePlotter:
    """
    Convenience class for plotting pipeline results.

    Args:
        pipeline: Pipeline instance
        df: Optional DataFrame (uses pipeline.get_final_data() if not provided)

    Example:
        >>> plotter = PipelinePlotter(pipeline)
        >>> plotter.plot_funnel()
        >>> plotter.plot_distribution('tm')
    """

    def __init__(self, pipeline, df: Optional[pd.DataFrame] = None):
        self.pipeline = pipeline
        self.df = df if df is not None else pipeline.get_final_data()

    def plot_funnel(self, **kwargs):
        """Plot pipeline funnel."""
        plot_pipeline_funnel(self.pipeline.stage_results, **kwargs)

    def plot_distribution(self, column: str, **kwargs):
        """Plot distribution of a column."""
        plot_distribution(self.df, column, **kwargs)

    def plot_scatter(self, x_col: str, y_col: str, **kwargs):
        """Plot scatter plot."""
        plot_scatter(self.df, x_col, y_col, **kwargs)

    def plot_correlation_matrix(self, **kwargs):
        """Plot correlation matrix."""
        plot_correlation_matrix(self.df, **kwargs)

    def plot_temperature_scan(self, metric_col: str, **kwargs):
        """Plot temperature scan results."""
        plot_temperature_scan(self.df, metric_col, **kwargs)

    def plot_diversity(self, reference_sequence: Optional[str] = None, **kwargs):
        """Plot sequence diversity."""
        plot_sequence_diversity(self.df, reference_sequence, **kwargs)

    def plot_predictions(self, columns: Optional[list[str]] = None, **kwargs):
        """Plot distributions of all numeric prediction columns in a multi-panel layout.

        Args:
            columns: Specific columns to plot. If None, auto-detects all numeric
                columns except sequence_id, length, and hash.
            **kwargs: Forwarded to matplotlib (e.g. figsize, bins).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install matplotlib"
            ) from None

        if columns is None:
            exclude = {"sequence_id", "sequence", "length", "hash"}
            columns = [
                c for c in self.df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        if not columns:
            print("No numeric prediction columns found to plot.")
            return

        n = len(columns)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        figsize = kwargs.pop("figsize", (5 * ncols, 4 * nrows))
        bins = kwargs.pop("bins", 30)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for i, col in enumerate(columns):
            ax = axes[i // ncols][i % ncols]
            data = self.df[col].dropna()
            if data.empty:
                ax.set_title(f"{col} (no data)")
                continue
            ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")
            ax.set_title(col)
            ax.set_ylabel("Count")
            mean_val = data.mean()
            ax.axvline(mean_val, color="red", linestyle="--", alpha=0.7,
                       label=f"Mean: {mean_val:.2f}")
            ax.legend(fontsize=8)

        # Hide unused axes
        for i in range(n, nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        plt.suptitle("Prediction Distributions", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_distributions(self, columns: Optional[list[str]] = None, **kwargs):
        """Alias for plot_predictions — plot distributions of prediction columns."""
        return self.plot_predictions(columns=columns, **kwargs)
