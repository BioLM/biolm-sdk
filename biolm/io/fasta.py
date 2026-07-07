"""FASTA format input/output utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import IO, Any, Dict, List, Union


def _detect_sequence_type(sequence: str) -> str:
    """Detect sequence type (DNA, RNA, or protein).
    
    Args:
        sequence: Sequence string
        
    Returns:
        Sequence type: "dna", "rna", "aa", or "unknown"
    """
    seq_upper = sequence.upper().replace(":", "")  # Remove colons from paired sequences
    if all(c in "ATCGN" for c in seq_upper):
        return "dna"
    elif all(c in "AUCGN" for c in seq_upper):
        return "rna"
    elif all(c in "ACDEFGHIKLMNPQRSTVWY*" for c in seq_upper):
        return "aa"
    else:
        return "unknown"


def _parse_fasta_header(header: str) -> tuple[str, Dict[str, Any]]:
    """Parse FASTA header line into ID and metadata.
    
    FASTA headers start with '>' and may contain metadata separated by spaces
    or pipes. Common formats:
    - >seq_id
    - >seq_id description
    - >seq_id|metadata1|metadata2
    
    Args:
        header: Header line (with or without '>')
        
    Returns:
        Tuple of (sequence_id, metadata_dict)
    """
    # Remove leading '>' if present
    header = header.lstrip(">").strip()
    
    if not header:
        return "sequence", {}
    
    # Try to parse pipe-separated metadata
    parts = header.split("|")
    seq_id = parts[0].strip()
    metadata = {}
    
    # If there are additional parts, add them to metadata
    if len(parts) > 1:
        for i, part in enumerate(parts[1:], 1):
            metadata[f"metadata_{i}"] = part.strip()
    
    # If no pipes, check for space-separated description
    if "|" not in header and " " in header:
        parts = header.split(" ", 1)
        seq_id = parts[0].strip()
        if len(parts) > 1:
            metadata["description"] = parts[1].strip()
    
    return seq_id, metadata


def load_fasta(file_path: Union[str, Path, IO]) -> List[Dict[str, Any]]:
    """Load sequences from a FASTA file.
    
    Parses a FASTA file and returns a list of dictionaries suitable for use
    with BioLM API requests. Each dictionary contains:
    - "sequence": The sequence string
    - "id": Sequence identifier from header (if available)
    - "metadata": Additional metadata from header (if available)
    
    Supports multi-line sequences (wrapped sequences).
    
    Args:
        file_path: Path to FASTA file (str, Path) or file-like object
        
    Returns:
        List of dictionaries, each containing sequence data
        
    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If file is empty or malformed
        
    Example:
        >>> items = load_fasta("sequences.fasta")
        >>> items[0]
        {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1', 'metadata': {}}
    """
    # Handle file path vs file-like object
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {file_path}")
        file_obj = open(file_path, "r", encoding="utf-8")
        should_close = True
    else:
        file_obj = file_path
        should_close = False
    
    try:
        content = file_obj.read()
        
        if not content.strip():
            raise ValueError("FASTA file is empty")
        
        sequences = []
        current_id = None
        current_sequence = []
        current_metadata = {}
        
        lines = content.splitlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Header line
            if line.startswith(">"):
                # Save previous sequence if exists
                if current_id is not None or current_sequence:
                    seq_str = "".join(current_sequence)
                    if seq_str:  # Only add non-empty sequences
                        sequences.append({
                            "sequence": seq_str,
                            "id": current_id or f"sequence_{len(sequences) + 1}",
                            "metadata": current_metadata,
                        })
                
                # Parse new header
                current_id, current_metadata = _parse_fasta_header(line)
                current_sequence = []
            else:
                # Sequence line (may be continuation of multi-line sequence)
                current_sequence.append(line)
        
        # Don't forget the last sequence
        if current_id is not None or current_sequence:
            seq_str = "".join(current_sequence)
            if seq_str:  # Only add non-empty sequences
                sequences.append({
                    "sequence": seq_str,
                    "id": current_id or f"sequence_{len(sequences) + 1}",
                    "metadata": current_metadata,
                })
        
        if not sequences:
            raise ValueError("No sequences found in FASTA file")
        
        return sequences
        
    finally:
        if should_close:
            file_obj.close()


def to_fasta(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path, IO],
    sequence_key: str = "sequence",
) -> None:
    """Write sequences to a FASTA file.
    
    Converts a list of dictionaries (API response format) to FASTA format.
    Each dictionary should contain a sequence field (default: "sequence").
    
    Args:
        data: List of dictionaries containing sequence data
        file_path: Output file path (str, Path) or file-like object
        sequence_key: Key to use for sequence data (default: "sequence")
        
    Raises:
        ValueError: If sequence_key is missing from any item
        KeyError: If required keys are missing
        
    Example:
        >>> data = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"}]
        >>> to_fasta(data, "output.fasta")
    """
    if not data:
        raise ValueError("Cannot write empty data to FASTA file")
    
    # Handle file path vs file-like object
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        file_obj = open(file_path, "w", encoding="utf-8")
        should_close = True
    else:
        file_obj = file_path
        should_close = False
    
    try:
        for i, item in enumerate(data):
            # Get sequence
            if sequence_key not in item:
                raise ValueError(
                    f"Item {i} missing required key '{sequence_key}'. "
                    f"Available keys: {list(item.keys())}"
                )
            
            sequence = item[sequence_key]
            if not isinstance(sequence, str):
                raise ValueError(
                    f"Item {i}: sequence must be a string, got {type(sequence)}"
                )
            
            # Get ID from item, or generate one
            seq_id = item.get("id") or item.get("sequence_id") or f"sequence_{i + 1}"
            
            # Build header
            header_parts = [seq_id]
            
            # Add metadata if present
            metadata = item.get("metadata", {})
            if metadata:
                # Add description if present
                if "description" in metadata:
                    header_parts.append(metadata["description"])
                # Add other metadata as pipe-separated values
                other_metadata = [
                    str(v) for k, v in metadata.items()
                    if k != "description" and v is not None
                ]
                if other_metadata:
                    header_parts.extend(other_metadata)
            
            header = ">" + "|".join(header_parts)
            
            # Write header and sequence
            file_obj.write(header + "\n")
            
            # Write sequence (optionally wrap at 80 characters for readability)
            # For now, write as single line (can be enhanced later)
            file_obj.write(sequence + "\n")
            
    finally:
        if should_close:
            file_obj.close()

