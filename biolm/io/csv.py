"""CSV format input/output utilities."""
import csv
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union


def load_csv(
    file_path: Union[str, Path, IO],
    sequence_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load data from a CSV file.

    Parses a CSV file and returns a list of dictionaries suitable for use
    with BioLM API requests. Each row becomes a dictionary with column
    headers as keys. All values are kept as strings (no type inference).

    Args:
        file_path: Path to CSV file (str, Path) or file-like object.
        sequence_key: Optional key name to validate exists in CSV; if provided, raises ValueError if column is missing.

    Returns:
        List of dictionaries, one per row.

    Raises:
        FileNotFoundError: If file path doesn't exist.
        ValueError: If file is empty or sequence_key column is missing.

    Example:
        >>> items = load_csv("data.csv", sequence_key="sequence")
        >>> items[0]
        {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1', 'score': '0.95'}
    """
    # Handle file path vs file-like object
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        file_obj = open(file_path, "r", encoding="utf-8", newline="")
        should_close = True
    else:
        file_obj = file_path
        should_close = False
    
    try:
        # Detect if file is empty
        first_char = file_obj.read(1)
        if not first_char:
            raise ValueError("CSV file is empty")
        file_obj.seek(0)
        
        reader = csv.DictReader(file_obj)
        
        # Check for sequence_key if specified
        if sequence_key:
            if sequence_key not in reader.fieldnames:
                raise ValueError(
                    f"CSV file missing required column '{sequence_key}'. "
                    f"Available columns: {reader.fieldnames or 'none'}"
                )
        
        rows = list(reader)
        
        if not rows:
            raise ValueError("CSV file contains no data rows")
        
        # Ensure all values are strings (CSV reader may return empty strings)
        # and handle None values
        result = []
        for row in rows:
            cleaned_row = {}
            for key, value in row.items():
                # Keep as string, empty string if None
                cleaned_row[key] = value if value is not None else ""
            result.append(cleaned_row)
        
        return result
        
    finally:
        if should_close:
            file_obj.close()


def to_csv(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path, IO],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Write data to a CSV file.

    Converts a list of dictionaries (API response format) to CSV format.

    Args:
        data: List of dictionaries to write.
        file_path: Output file path (str, Path) or file-like object.
        fieldnames: Optional list of column names; if not provided, inferred from the first item's keys (missing keys filled with empty strings).

    Raises:
        ValueError: If data is empty.

    Example:
        >>> data = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "score": 0.95}]
        >>> to_csv(data, "output.csv")
    """
    if not data:
        raise ValueError("Cannot write empty data to CSV file")
    
    # Handle file path vs file-like object
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        file_obj = open(file_path, "w", encoding="utf-8", newline="")
        should_close = True
    else:
        file_obj = file_path
        should_close = False
    
    try:
        # Determine fieldnames
        if fieldnames is None:
            # Infer from first item
            if not data:
                raise ValueError("Cannot infer fieldnames from empty data")
            fieldnames = list(data[0].keys())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_fieldnames = []
        for field in fieldnames:
            if field not in seen:
                seen.add(field)
                unique_fieldnames.append(field)
        fieldnames = unique_fieldnames
        
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        
        for item in data:
            # Convert all values to strings, handle missing keys
            row = {}
            for field in fieldnames:
                if field in item:
                    value = item[field]
                    # Convert to string, handle None
                    row[field] = str(value) if value is not None else ""
                else:
                    row[field] = ""
            
            writer.writerow(row)
            
    finally:
        if should_close:
            file_obj.close()

