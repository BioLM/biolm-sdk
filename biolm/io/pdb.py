"""PDB format input/output utilities."""
from pathlib import Path
from typing import IO, Any, Dict, List, Union


def load_pdb(file_path: Union[str, Path, IO]) -> List[Dict[str, Any]]:
    """Load PDB structure(s) from a PDB file.
    
    Reads a PDB file and returns a list of dictionaries suitable for use
    with BioLM API requests. For single-model PDBs, returns one item.
    For multi-model PDBs (with MODEL/ENDMDL records), returns one item per model.
    
    Args:
        file_path: Path to PDB file (str, Path) or file-like object
        
    Returns:
        List of dictionaries, each containing "pdb" key with PDB content.
        Single-model files return [{"pdb": "..."}].
        Multi-model files return [{"pdb": "..."}, {"pdb": "..."}, ...].
        
    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If file is empty
        
    Example:
        >>> items = load_pdb("structure.pdb")
        >>> items[0]
        {'pdb': 'ATOM      1  N   MET A   1      20.154  16.967  19.502...'}
    """
    # Handle file path vs file-like object
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDB file not found: {file_path}")
        file_obj = open(file_path, "r", encoding="utf-8")
        should_close = True
    else:
        file_obj = file_path
        should_close = False
    
    try:
        content = file_obj.read()
        
        if not content.strip():
            raise ValueError("PDB file is empty")
        
        # Check for multi-model PDB (contains MODEL/ENDMDL records)
        if "MODEL" in content and "ENDMDL" in content:
            # Split into models
            models = []
            current_model = []
            in_model = False
            
            lines = content.splitlines()
            for line in lines:
                if line.startswith("MODEL"):
                    if current_model and in_model:
                        # Save previous model
                        models.append("\n".join(current_model))
                    current_model = [line]
                    in_model = True
                elif line.startswith("ENDMDL"):
                    current_model.append(line)
                    models.append("\n".join(current_model))
                    current_model = []
                    in_model = False
                elif in_model:
                    current_model.append(line)
                else:
                    # Content before first MODEL or after last ENDMDL
                    # Include in first/last model or as separate content
                    if not models:
                        current_model.append(line)
                    else:
                        # Append to last model
                        if current_model:
                            current_model.append(line)
            
            # Handle any remaining content
            if current_model:
                models.append("\n".join(current_model))
            
            # Return list of dicts, one per model
            return [{"pdb": model.strip()} for model in models if model.strip()]
        else:
            # Single model PDB
            return [{"pdb": content.strip()}]
            
    finally:
        if should_close:
            file_obj.close()


def to_pdb(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path, IO],
    pdb_key: str = "pdb",
) -> None:
    """Write PDB structure(s) to a PDB file.
    
    Converts a list of dictionaries (API response format) to PDB format.
    Each dictionary should contain a PDB content field (default: "pdb").
    If multiple items are provided, they are concatenated.
    
    Args:
        data: List of dictionaries containing PDB data
        file_path: Output file path (str, Path) or file-like object
        pdb_key: Key to use for PDB content (default: "pdb")
        
    Raises:
        ValueError: If data is empty or pdb_key is missing from any item
        
    Example:
        >>> data = [{"pdb": "ATOM      1  N   MET A   1..."}]
        >>> to_pdb(data, "output.pdb")
    """
    if not data:
        raise ValueError("Cannot write empty data to PDB file")
    
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
            if pdb_key not in item:
                raise ValueError(
                    f"Item {i} missing required key '{pdb_key}'. "
                    f"Available keys: {list(item.keys())}"
                )
            
            pdb_content = item[pdb_key]
            if not isinstance(pdb_content, str):
                raise ValueError(
                    f"Item {i}: PDB content must be a string, got {type(pdb_content)}"
                )
            
            # Write PDB content
            file_obj.write(pdb_content)
            
            # Add newline between multiple structures if not already present
            if i < len(data) - 1 and not pdb_content.endswith("\n"):
                file_obj.write("\n")
            
    finally:
        if should_close:
            file_obj.close()

