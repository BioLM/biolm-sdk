"""JSON format input/output utilities."""
import json
import sys
from pathlib import Path
from typing import IO, Any, Dict, List, Union


def load_json(file_path: Union[str, Path, IO]) -> List[Dict[str, Any]]:
    """Load data from a JSON file or JSONL (newline-delimited JSON) file.
    
    Parses a JSON file and returns a list of dictionaries suitable for use
    with BioLM API requests. Supports:
    - Single JSON object: Returns list with one item
    - JSON array: Returns list of items
    - JSONL format (newline-delimited): Returns list of items, one per line
    
    Args:
        file_path: Path to JSON/JSONL file (str, Path), file-like object, or "-" for stdin
        
    Returns:
        List of dictionaries, each containing data suitable for API requests
        
    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If file is empty or malformed
        json.JSONDecodeError: If JSON is invalid
        
    Example:
        >>> items = load_json("data.json")
        >>> items[0]
        {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1'}
        
        >>> items = load_json("data.jsonl")  # JSONL format
        >>> len(items)
        3
    """
    # Handle stdin
    if file_path == "-" or (isinstance(file_path, str) and file_path == "-"):
        file_obj = sys.stdin
        should_close = False
    # Handle file path vs file-like object
    elif isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        file_obj = open(file_path, "r", encoding="utf-8")
        should_close = True
    else:
        file_obj = file_path
        should_close = False
    
    try:
        content = file_obj.read()
        
        if not content.strip():
            raise ValueError("JSON file is empty")
        
        # Try to detect JSONL format (newline-delimited JSON)
        # JSONL has one JSON object per line
        lines = content.strip().split('\n')
        if len(lines) > 1:
            # Check if each line is a valid JSON object
            try:
                items = []
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        item = json.loads(line)
                        if isinstance(item, dict):
                            items.append(item)
                        else:
                            # If any line is not a dict, treat as regular JSON
                            raise ValueError("Mixed format detected")
                return items
            except (json.JSONDecodeError, ValueError):
                # Not JSONL, fall through to regular JSON parsing
                pass
        
        # Parse as regular JSON
        data = json.loads(content)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Single object - wrap in list
            return [data]
        elif isinstance(data, list):
            # Array of objects
            if not data:
                raise ValueError("JSON array is empty")
            # Validate all items are dicts
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Item {i} in JSON array is not a dictionary, got {type(item).__name__}"
                    )
            return data
        else:
            raise ValueError(
                f"JSON root must be an object or array, got {type(data).__name__}"
            )
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    finally:
        if should_close:
            file_obj.close()


def to_json(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path, IO],
    indent: int = 2,
    jsonl: bool = False,
) -> None:
    """Write data to a JSON file or JSONL (newline-delimited JSON) file.

    Converts a list of dictionaries (API response format) to JSON format.

    Args:
        data: List of dictionaries to write.
        file_path: Output file path (str, Path), file-like object, or "-" for stdout.
        indent: Indentation level for JSON (default: 2). Use None for compact JSON.
        jsonl: If True, write as JSONL (one JSON object per line); if False, write as JSON array. Default: False.

    Raises:
        ValueError: If data is empty.

    Example:
        >>> data = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "score": 0.95}]
        >>> to_json(data, "output.json")
        >>> to_json(data, "output.jsonl", jsonl=True)
    """
    if not data:
        raise ValueError("Cannot write empty data to JSON file")
    
    # Handle stdout
    if file_path == "-" or (isinstance(file_path, str) and file_path == "-"):
        file_obj = sys.stdout
        should_close = False
    # Handle file path vs file-like object
    elif isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        # Auto-detect JSONL from extension
        if file_path.suffix == ".jsonl":
            jsonl = True
        file_obj = open(file_path, "w", encoding="utf-8")
        should_close = True
    else:
        file_obj = file_path
        should_close = False
    
    try:
        if jsonl:
            # Write as JSONL (one JSON object per line)
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                file_obj.write(json_line + "\n")
        else:
            # Write as JSON array
            json.dump(data, file_obj, indent=indent, ensure_ascii=False)
            file_obj.write("\n")  # Add trailing newline for consistency
            
    finally:
        if should_close:
            file_obj.close()

