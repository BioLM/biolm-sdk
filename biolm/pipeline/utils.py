"""
Utility functions for pipeline operations.

Also includes structure file conversion utilities (CIF ↔ PDB) for use with
structure-conditioned generative models (AntiFold, HyperMPNN, etc.).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def load_sequences_from_file(
    file_path: Union[str, Path], format: Optional[str] = None
) -> list[str]:
    """
    Load sequences from a file.

    Supports:
    - FASTA (.fasta, .fa, .faa)
    - CSV (.csv) with 'sequence' column
    - Plain text (one sequence per line)

    Args:
        file_path: Path to file
        format: Optional format override ('fasta', 'csv', 'txt')

    Returns:
        List of sequences

    Example:
        >>> sequences = load_sequences_from_file('sequences.fasta')
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect format
    if format is None:
        if path.suffix in [".fasta", ".fa", ".faa"]:
            format = "fasta"
        elif path.suffix == ".csv":
            format = "csv"
        else:
            format = "txt"

    if format == "fasta":
        return load_fasta(path)
    elif format == "csv":
        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            raise ValueError("CSV must have 'sequence' column")
        return df["sequence"].tolist()
    elif format == "txt":
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_fasta(file_path: Union[str, Path]) -> list[str]:
    """
    Load sequences from FASTA file.

    Args:
        file_path: Path to FASTA file

    Returns:
        List of sequences
    """
    sequences = []
    current_seq = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append("".join(current_seq))

    return sequences


def write_fasta(
    sequences: Union[list[str], pd.DataFrame],
    file_path: Union[str, Path],
    headers: Optional[list[str]] = None,
    sequence_column: str = "sequence",
    header_column: Optional[str] = None,
):
    """
    Write sequences to FASTA file.

    Args:
        sequences: List of sequences or DataFrame
        file_path: Output file path
        headers: Optional list of headers (one per sequence)
        sequence_column: Column name for sequences (if DataFrame)
        header_column: Column name for headers (if DataFrame)

    Example:
        >>> write_fasta(sequences, 'output.fasta')
        >>> write_fasta(df, 'output.fasta', header_column='id')
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(sequences, pd.DataFrame):
        seqs = sequences[sequence_column].tolist()
        if header_column and header_column in sequences.columns:
            headers = sequences[header_column].tolist()
        else:
            headers = [f"seq_{i}" for i in range(len(seqs))]
    else:
        seqs = sequences
        if headers is None:
            headers = [f"seq_{i}" for i in range(len(seqs))]

    with open(path, "w") as f:
        for header, seq in zip(headers, seqs):
            f.write(f">{header}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i:i+80]}\n")


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity (fraction of identical positions).

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Sequence identity (0.0 to 1.0)
    """
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0

    identical = sum(c1 == c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
    return identical / min_len


def compute_hamming_distance(seq1: str, seq2: str, normalize: bool = False) -> float:
    """
    Compute Hamming distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence
        normalize: If True, return normalized distance (0-1)

    Returns:
        Hamming distance
    """
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        distance = sum(c1 != c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
        distance += abs(len(seq1) - len(seq2))
    else:
        distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

    if normalize:
        max_len = max(len(seq1), len(seq2))
        return distance / max_len if max_len > 0 else 0.0
    return distance


def deduplicate_sequences(
    sequences: Union[list[str], pd.DataFrame], sequence_column: str = "sequence"
) -> Union[list[str], pd.DataFrame]:
    """
    Remove duplicate sequences.

    Args:
        sequences: List of sequences or DataFrame
        sequence_column: Column name if DataFrame

    Returns:
        Deduplicated sequences (same type as input)
    """
    if isinstance(sequences, list):
        seen = set()
        result = []
        for seq in sequences:
            if seq not in seen:
                seen.add(seq)
                result.append(seq)
        return result
    else:
        return sequences.drop_duplicates(subset=[sequence_column]).reset_index(
            drop=True
        )


def hash_sequence(sequence: str) -> str:
    """Generate SHA256 hash of a sequence."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:32]


def validate_sequence(sequence: str, alphabet: str = "protein") -> bool:
    """
    Validate that a sequence contains only valid amino acids.

    Args:
        sequence: Sequence to validate
        alphabet: 'protein' or 'dna'

    Returns:
        True if valid, False otherwise
    """
    if alphabet == "protein":
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    elif alphabet == "dna":
        valid_chars = set("ACGT")
    else:
        raise ValueError(f"Unknown alphabet: {alphabet}")

    return all(c in valid_chars for c in sequence.upper())


def split_sequences_by_length(
    sequences: list[str], boundaries: list[int]
) -> dict[str, list[str]]:
    """
    Split sequences into bins by length.

    Args:
        sequences: List of sequences
        boundaries: List of length boundaries [b1, b2, ...]
                   Creates bins: <b1, b1-b2, b2-b3, ..., >bn

    Returns:
        Dict mapping bin names to sequence lists

    Example:
        >>> bins = split_sequences_by_length(seqs, [100, 200, 300])
        >>> # Returns: {'<100': [...], '100-200': [...], '200-300': [...], '>300': [...]}
    """
    bins = {}

    # Create bin names
    sorted_boundaries = sorted(boundaries)
    bin_names = [f"<{sorted_boundaries[0]}"]
    for i in range(len(sorted_boundaries) - 1):
        bin_names.append(f"{sorted_boundaries[i]}-{sorted_boundaries[i+1]}")
    bin_names.append(f">{sorted_boundaries[-1]}")

    # Initialize bins
    for name in bin_names:
        bins[name] = []

    # Assign sequences to bins
    for seq in sequences:
        length = len(seq)
        assigned = False

        for i, boundary in enumerate(sorted_boundaries):
            if length < boundary:
                bins[bin_names[i]].append(seq)
                assigned = True
                break

        if not assigned:
            bins[bin_names[-1]].append(seq)

    return bins


def sample_sequences(
    sequences: Union[list[str], pd.DataFrame],
    n: int,
    method: str = "random",
    score_column: Optional[str] = None,
    random_seed: Optional[int] = 42,
) -> Union[list[str], pd.DataFrame]:
    """
    Sample n sequences from a collection.

    Args:
        sequences: List or DataFrame
        n: Number to sample
        method: 'random', 'top', or 'spread'
        score_column: Column for 'top' or 'spread' methods (DataFrame only)
        random_seed: Random seed

    Returns:
        Sampled sequences (same type as input)
    """
    if isinstance(sequences, list):
        if method == "random":
            import random

            random.seed(random_seed)
            return random.sample(sequences, min(n, len(sequences)))
        else:
            raise ValueError("Only 'random' method supported for list input")
    else:
        df = sequences

        if len(df) <= n:
            return df.copy()

        if method == "random":
            return df.sample(n=n, random_state=random_seed)

        elif method == "top":
            if score_column is None:
                raise ValueError("score_column required for 'top' method")
            return df.nlargest(n, score_column)

        elif method == "spread":
            if score_column:
                df_sorted = df.sort_values(score_column)
            else:
                df_sorted = df

            indices = np.linspace(0, len(df_sorted) - 1, n, dtype=int)
            return df_sorted.iloc[indices].copy()

        else:
            raise ValueError(f"Unknown method: {method}")


def merge_prediction_results(
    df: pd.DataFrame,
    predictions: dict[str, list[float]],
    prediction_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Merge prediction results into DataFrame.

    Args:
        df: Base DataFrame
        predictions: Dict mapping prediction names to value lists
        prediction_names: Optional list of column names

    Returns:
        DataFrame with predictions added
    """
    df = df.copy()

    for name, values in predictions.items():
        if len(values) != len(df):
            raise ValueError(f"Length mismatch: {len(values)} != {len(df)}")
        df[name] = values

    return df


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of DataFrame columns.

    Returns:
        Summary DataFrame with statistics
    """
    summary_data = []

    for col in df.columns:
        dtype = df[col].dtype

        row = {
            "Column": col,
            "Type": str(dtype),
            "Non-Null": df[col].notna().sum(),
            "Null": df[col].isna().sum(),
            "Unique": df[col].nunique(),
        }

        if np.issubdtype(dtype, np.number):
            row["Mean"] = df[col].mean()
            row["Std"] = df[col].std()
            row["Min"] = df[col].min()
            row["Max"] = df[col].max()

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def create_run_summary(pipeline) -> dict[str, Any]:
    """
    Create a summary dict of pipeline run.

    Args:
        pipeline: Pipeline instance

    Returns:
        Summary dict
    """
    summary = {
        "run_id": pipeline.run_id,
        "pipeline_type": pipeline.pipeline_type,
        "status": pipeline.status,
        "num_stages": len(pipeline.stages),
        "stage_names": [s.name for s in pipeline.stages],
        "stage_results": {},
    }

    for stage_name, result in pipeline.stage_results.items():
        summary["stage_results"][stage_name] = {
            "input_count": result.input_count,
            "output_count": result.output_count,
            "cached_count": result.cached_count,
            "computed_count": result.computed_count,
            "filtered_count": result.filtered_count,
            "elapsed_time": result.elapsed_time,
        }

    if pipeline.start_time and pipeline.end_time:
        summary["total_time"] = pipeline.end_time - pipeline.start_time

    return summary


def export_run_summary(pipeline, output_path: Union[str, Path]):
    """
    Export pipeline run summary to JSON.

    Args:
        pipeline: Pipeline instance
        output_path: Output file path
    """
    summary = create_run_summary(pipeline)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported run summary to {path}")


# ---------------------------------------------------------------------------
# Structure file conversion utilities
# ---------------------------------------------------------------------------


def cif_to_pdb(cif_path: str, output_path: Optional[str] = None) -> str:
    """Convert a CIF structure file to PDB format.

    Uses gemmi if available, falls back to biopython.

    Args:
        cif_path: Path to input CIF (.cif / .mmcif) file.
        output_path: Destination PDB path. Defaults to same name with .pdb extension.

    Returns:
        Path to written PDB file.

    Raises:
        ImportError: If neither gemmi nor biopython is installed.
    """
    if output_path is None:
        output_path = os.path.splitext(cif_path)[0] + ".pdb"

    try:
        import gemmi  # type: ignore

        structure = gemmi.read_structure(cif_path)
        structure.write_pdb(output_path)
        return output_path
    except ImportError:
        pass

    try:
        from Bio.PDB import PDBIO, MMCIFParser  # type: ignore

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path)
        return output_path
    except ImportError:
        pass

    raise ImportError(
        "CIF→PDB conversion requires gemmi or biopython.\n"
        "  pip install gemmi     # preferred\n"
        "  pip install biopython # fallback"
    )


def pdb_to_cif(pdb_path: str, output_path: Optional[str] = None) -> str:
    """Convert a PDB structure file to CIF format.

    Uses gemmi if available, falls back to biopython.

    Args:
        pdb_path: Path to input PDB (.pdb / .ent) file.
        output_path: Destination CIF path. Defaults to same name with .cif extension.

    Returns:
        Path to written CIF file.

    Raises:
        ImportError: If neither gemmi nor biopython is installed.
    """
    if output_path is None:
        output_path = os.path.splitext(pdb_path)[0] + ".cif"

    try:
        import gemmi  # type: ignore

        structure = gemmi.read_structure(pdb_path)
        structure.make_mmcif_document().write_file(output_path)
        return output_path
    except ImportError:
        pass

    try:
        from Bio.PDB import MMCIFIO, PDBParser  # type: ignore

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", pdb_path)
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_path)
        return output_path
    except ImportError:
        pass

    raise ImportError(
        "PDB→CIF conversion requires gemmi or biopython.\n"
        "  pip install gemmi     # preferred\n"
        "  pip install biopython # fallback"
    )


def load_structure_string(path: str) -> tuple[str, str]:
    """Load a structure file and return (format, content_string).

    Args:
        path: Path to a .pdb, .ent, .cif, or .mmcif file.

    Returns:
        Tuple of (format_str, content_str) where format_str is 'pdb' or 'cif'.

    Raises:
        ValueError: If file extension is not recognised.
        FileNotFoundError: If file does not exist.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pdb", ".ent"):
        fmt = "pdb"
    elif ext in (".cif", ".mmcif"):
        fmt = "cif"
    else:
        raise ValueError(
            f"Unrecognised structure file extension '{ext}'. "
            "Expected: .pdb, .ent, .cif, .mmcif"
        )
    with open(path) as fh:
        content = fh.read()
    return fmt, content
