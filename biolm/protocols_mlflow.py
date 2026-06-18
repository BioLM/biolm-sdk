"""MLflow logging for protocol execution results.

This module provides functionality to log protocol results to MLflow based on
the protocol's outputs configuration. It handles result selection, template
expression evaluation, and MLflow run creation.
"""
from __future__ import annotations

import gzip
import json
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

from biolm.core.expression_evaluator import (
    evaluate_expression,
    evaluate_template_value,
    evaluate_where_clause,
    extract_template_expr,
)


class MLflowNotAvailableError(Exception):
    """Raised when MLflow is not installed."""

    def __init__(self):
        super().__init__(
            "MLflow is not installed. Install it with: pip install biolmai[mlflow]"
        )


def _check_mlflow_available():
    """Check if MLflow is available, raise error if not."""
    if not MLFLOW_AVAILABLE:
        raise MLflowNotAvailableError()


def load_results(results: Union[List[Dict], str]) -> List[Dict]:
    """Load results from a list or JSONL file.

    Supports plain JSONL and compressed formats:
    * .jsonl - plain text
    * .jsonl.gz - gzip compressed
    * .zip - zip archive containing a .jsonl file (e.g. results_{id}.jsonl.zip)

    Args:
        results: Either a list of dicts or a path to a JSONL file (optionally compressed)

    Returns:
        List of result dictionaries
    """
    if isinstance(results, list):
        return results
    elif isinstance(results, str):
        path = Path(results)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {results}")

        suffix = path.suffix.lower()
        if suffix == ".gz" or path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                lines = f.readlines()
        elif suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                jsonl_names = [n for n in zf.namelist() if n.endswith(".jsonl")]
                if not jsonl_names:
                    raise ValueError(
                        f"No .jsonl file found in zip archive: {path}. "
                        "Expected at least one JSONL file in the archive."
                    )
                with zf.open(jsonl_names[0], "r") as f:
                    lines = f.read().decode("utf-8").splitlines()
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        results_list = []
        for line in lines:
            line = line.strip()
            if line:
                results_list.append(json.loads(line))
        return results_list
    else:
        raise TypeError(f"results must be a list or file path, got {type(results)}")


def load_outputs_config(
    outputs_config: Union[List[Dict], str, Dict]
) -> List[Dict]:
    """Load outputs configuration from various formats.
    
    Args:
        outputs_config: Can be:
            - List of output rule dicts
            - Path to YAML file containing outputs config
            - Path to protocol YAML file (will extract outputs section)
            - Protocol dict (will extract outputs section)
            
    Returns:
        List of output rule dictionaries
    """
    import yaml
    
    if isinstance(outputs_config, list):
        return outputs_config
    elif isinstance(outputs_config, dict):
        # Extract outputs section if it's a full protocol dict
        if "outputs" in outputs_config:
            return outputs_config["outputs"]
        # Otherwise assume it's already the outputs config
        return [outputs_config]
    elif isinstance(outputs_config, str):
        # Load from YAML file
        path = Path(outputs_config)
        if not path.exists():
            raise FileNotFoundError(f"Outputs config file not found: {outputs_config}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Extract outputs section if it's a protocol file
        if isinstance(data, dict) and "outputs" in data:
            return data["outputs"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(
                f"Invalid outputs config format in {outputs_config}. "
                "Expected list of output rules or protocol with outputs section."
            )
    else:
        raise TypeError(
            f"outputs_config must be a list, dict, or file path, got {type(outputs_config)}"
        )


def select_results(results: List[Dict], output_rule: Dict) -> List[Dict]:
    """Select results based on an output rule.
    
    Args:
        results: List of result dictionaries
        output_rule: Output rule configuration
        
    Returns:
        List of selected result dictionaries
    """
    selected = results.copy()
    
    # Apply where filter
    where_expr = output_rule.get("where")
    if where_expr:
        selected = [
            row for row in selected if evaluate_where_clause(where_expr, row)
        ]
    
    # Apply order_by
    order_by = output_rule.get("order_by", [])
    if order_by:
        # Sort by multiple fields (most significant first)
        for order_spec in reversed(order_by):
            field = order_spec.get("field")
            order = order_spec.get("order", "asc")
            reverse = order == "desc"
            
            # Sort by field value
            selected.sort(key=lambda x: x.get(field), reverse=reverse)
    
    # Apply limit
    limit = output_rule.get("limit", 200)
    if limit is not None:
        # Evaluate limit if it's an expression
        is_template, expr = extract_template_expr(str(limit))
        if is_template:
            # Use first row as context for limit expression
            if selected:
                limit = int(evaluate_expression(expr, selected[0]))
            else:
                limit = 200
        else:
            limit = int(limit)
        
        selected = selected[:limit]
    
    return selected


def combine_output_rules(
    results: List[Dict], output_rules: List[Dict]
) -> Dict[str, Any]:
    """Combine results from multiple output rules.
    
    Args:
        results: Full list of result dictionaries
        output_rules: List of output rule configurations
        
    Returns:
        Dictionary mapping result index to combined logging data
    """
    # Track which results are selected by which rules
    result_to_rules: Dict[int, List[Dict]] = {}
    
    for rule in output_rules:
        selected = select_results(results, rule)
        # Track which original indices were selected by comparing dicts
        # Create a mapping of selected rows to their original indices
        for selected_row in selected:
            # Find the index in original results by comparing content
            for idx, original_row in enumerate(results):
                if selected_row is original_row or selected_row == original_row:
                    if idx not in result_to_rules:
                        result_to_rules[idx] = []
                    result_to_rules[idx].append(rule)
                    break
    
    # Build combined logging data for each selected result
    combined_data: Dict[int, Dict[str, Any]] = {}
    
    for idx, rules in result_to_rules.items():
        result = results[idx]
        combined_log = {
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }
        
        # Combine all rules' log specifications
        for rule in rules:
            log_spec = rule.get("log", {})
            
            # Merge params (last wins on collision)
            if "params" in log_spec:
                for key, value in log_spec["params"].items():
                    combined_log["params"][key] = value
            
            # Merge metrics (last wins on collision)
            if "metrics" in log_spec:
                for key, value in log_spec["metrics"].items():
                    combined_log["metrics"][key] = value
            
            # Merge tags (last wins on collision)
            if "tags" in log_spec:
                for key, value in log_spec["tags"].items():
                    combined_log["tags"][key] = value
            
            # Collect artifacts
            if "artifacts" in log_spec:
                combined_log["artifacts"].extend(log_spec["artifacts"])
        
        combined_data[idx] = {
            "result": result,
            "log": combined_log,
        }
    
    return combined_data


def compute_aggregates(
    selected_results: List[Dict],
    all_results: List[Dict],
    aggregates: List[Dict],
    aggregate_over: str = "selected",
) -> Dict[str, Any]:
    """Compute aggregate statistics.
    
    Args:
        selected_results: Results selected by output rules
        all_results: All results from protocol execution
        aggregates: List of aggregate specifications
        aggregate_over: "selected" or "all"
        
    Returns:
        Dictionary of aggregate metric names to values
    """
    # Choose which results to aggregate over
    if aggregate_over == "all":
        data = all_results
    else:
        data = selected_results
    
    aggregate_metrics = {}
    
    for agg_spec in aggregates:
        field = agg_spec.get("field")
        ops = agg_spec.get("ops", [])
        
        # Handle __rows__ special field
        if field == "__rows__":
            count = len(data)
            for op in ops:
                if op == "count":
                    aggregate_metrics["__rows__.count"] = count
            continue
        
        # Extract field values
        values = []
        for row in data:
            if field in row:
                value = row[field]
                # Only include numeric values
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            continue
        
        # Compute statistics
        for op in ops:
            metric_name = f"{field}.{op}"
            
            if op == "count":
                aggregate_metrics[metric_name] = len(values)
            elif op == "mean":
                aggregate_metrics[metric_name] = statistics.mean(values)
            elif op == "sum":
                aggregate_metrics[metric_name] = sum(values)
            elif op == "min":
                aggregate_metrics[metric_name] = min(values)
            elif op == "max":
                aggregate_metrics[metric_name] = max(values)
            elif op == "std":
                if len(values) > 1:
                    aggregate_metrics[metric_name] = statistics.stdev(values)
                else:
                    aggregate_metrics[metric_name] = 0.0
            elif op == "p50":
                aggregate_metrics[metric_name] = statistics.median(values)
            elif op == "p90":
                aggregate_metrics[metric_name] = _percentile(values, 90)
            elif op == "p95":
                aggregate_metrics[metric_name] = _percentile(values, 95)
            elif op == "p99":
                aggregate_metrics[metric_name] = _percentile(values, 99)
    
    return aggregate_metrics


def _percentile(values: List[float], p: int) -> float:
    """Compute percentile."""
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    floor = int(k)
    ceil = floor + 1
    
    if ceil >= len(sorted_values):
        return sorted_values[-1]
    
    weight = k - floor
    return sorted_values[floor] * (1 - weight) + sorted_values[ceil] * weight


def generate_seqparse(sequence: str, sequence_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    """Generate seqparse format JSON for a sequence.

    According to https://github.com/Lattice-Automation/seqparse, the format is::

        {
          "name": string,
          "type": "dna" | "rna" | "aa" | "unknown",
          "seq": string,
          "annotations": Annotation[]
        }

    Args:
        sequence: The sequence string
        sequence_id: Optional sequence identifier (used as name)
        metadata: Optional metadata dictionary (may contain annotations)

    Returns:
        JSON string in seqparse format
    """
    # Detect sequence type
    seq_upper = sequence.upper().replace(":", "")  # Remove colons from paired sequences
    if all(c in "ATCGN" for c in seq_upper):
        seq_type = "dna"
    elif all(c in "AUCGN" for c in seq_upper):
        seq_type = "rna"
    elif all(c in "ACDEFGHIKLMNPQRSTVWY*" for c in seq_upper):
        seq_type = "aa"
    else:
        seq_type = "unknown"
    
    # Extract annotations from metadata if present
    annotations = []
    if metadata:
        # If metadata has an 'annotations' key, use it directly
        if "annotations" in metadata and isinstance(metadata["annotations"], list):
            annotations = metadata["annotations"]
        # Otherwise, try to convert metadata fields to annotations
        # (This is a best-effort conversion - proper annotations need name, start, end)
        elif isinstance(metadata, dict):
            # For now, we'll just use empty annotations
            # Users should provide proper annotations in metadata if needed
            annotations = []
    
    entry = {
        "name": sequence_id or "sequence",
        "type": seq_type,
        "seq": sequence,
        "annotations": annotations
    }
    
    return json.dumps(entry, indent=2)


def prepare_artifact(
    artifact_spec: Dict, result: Dict
) -> tuple[str, Union[str, bytes]]:
    """Prepare an artifact for logging.
    
    Args:
        artifact_spec: Artifact specification from output rule
        result: Result dictionary
        
    Returns:
        Tuple of (artifact_name, artifact_content)
    """
    artifact_type = artifact_spec.get("type")
    artifact_name = artifact_spec.get("name", "artifact")
    
    if artifact_type == "seqparse":
        # Handle entries or content
        if "entries" in artifact_spec:
            # Evaluate entries expressions - each entry becomes a seqparse object
            seqparse_entries = []
            for entry_spec in artifact_spec["entries"]:
                sequence = None
                sequence_id = None
                metadata = None
                
                if "sequence" in entry_spec:
                    sequence = evaluate_template_value(entry_spec["sequence"], result)
                if "id" in entry_spec:
                    sequence_id = evaluate_template_value(entry_spec["id"], result)
                if "metadata" in entry_spec:
                    metadata = evaluate_template_value(entry_spec["metadata"], result)
                
                if sequence:
                    seqparse_entry = generate_seqparse(sequence, sequence_id, metadata)
                    # Parse the JSON string to get the dict, then add to list
                    seqparse_entries.append(json.loads(seqparse_entry))
            
            # If multiple entries, return as array; if single, return as object
            if len(seqparse_entries) == 1:
                content = json.dumps(seqparse_entries[0], indent=2)
            else:
                # Multiple entries - return as array of seqparse objects
                content = json.dumps(seqparse_entries, indent=2)
        else:
            # Single sequence from content
            content_expr = artifact_spec.get("content", "")
            if content_expr:
                sequence = evaluate_template_value(content_expr, result)
                sequence_id = result.get("id") or result.get("sequence_id")
                metadata = result.get("metadata") or {}
                content = generate_seqparse(sequence, sequence_id, metadata)
            else:
                raise ValueError("seqparse artifact must have 'entries' or 'content'")
        
        return artifact_name, content
    
    elif artifact_type in ["pdb", "text", "json"]:
        content_expr = artifact_spec.get("content", "")
        if content_expr:
            content = evaluate_template_value(content_expr, result)
            if artifact_type == "json" and isinstance(content, dict):
                content = json.dumps(content, indent=2)
            elif artifact_type == "json":
                content = str(content)
            return artifact_name, content
        else:
            raise ValueError(f"{artifact_type} artifact must have 'content'")
    
    else:
        # For other types, return as-is for now
        content_expr = artifact_spec.get("content", "")
        if content_expr:
            content = evaluate_template_value(content_expr, result)
            return artifact_name, str(content)
        raise ValueError(f"Unsupported artifact type: {artifact_type}")


def prepare_logging_data(
    results: List[Dict],
    outputs_config: List[Dict],
    protocol_metadata: Optional[Dict] = None,
    aggregate_over: str = "selected",
) -> Dict[str, Any]:
    """Prepare all logging data (Stage 1: No MLflow interaction).
    
    Args:
        results: List of result dictionaries
        outputs_config: List of output rule configurations
        protocol_metadata: Optional protocol metadata (name, version, inputs, etc.)
        aggregate_over: "selected" or "all" for aggregate computation
        
    Returns:
        Dictionary with prepared logging data
    """
    # Combine output rules to get selected results with logging specs
    combined_data = combine_output_rules(results, outputs_config)
    
    # Prepare child run data
    child_runs = []
    for idx, data in combined_data.items():
        result = data["result"]
        log_spec = data["log"]
        
        # Evaluate all template expressions in log spec
        params = {}
        for key, value in log_spec.get("params", {}).items():
            params[key] = evaluate_template_value(value, result)
        
        metrics = {}
        for key, value in log_spec.get("metrics", {}).items():
            metrics[key] = evaluate_template_value(value, result)
        
        tags = {}
        for key, value in log_spec.get("tags", {}).items():
            tags[key] = evaluate_template_value(value, result)
        
        # Prepare artifacts
        artifacts = []
        for artifact_spec in log_spec.get("artifacts", []):
            try:
                artifact_name, artifact_content = prepare_artifact(artifact_spec, result)
                artifacts.append((artifact_name, artifact_content))
            except Exception as e:
                raise ValueError(
                    f"Failed to prepare artifact '{artifact_spec.get('name', 'unknown')}': {e}"
                ) from e
        
        # Auto-generate sequence.json if sequence field exists
        if "sequence" in result and not any(
            a[0] == "sequence.json" for a in artifacts
        ):
            sequence = result["sequence"]
            sequence_id = result.get("id") or result.get("sequence_id")
            metadata = result.get("metadata") or {}
            seqparse_content = generate_seqparse(sequence, sequence_id, metadata)
            artifacts.append(("sequence.json", seqparse_content))
        
        child_runs.append(
            {
                "result": result,
                "params": params,
                "metrics": metrics,
                "tags": tags,
                "artifacts": artifacts,
            }
        )
    
    # Collect all aggregates from all output rules
    all_aggregates = []
    selected_results = [data["result"] for data in combined_data.values()]
    
    for rule in outputs_config:
        log_spec = rule.get("log", {})
        if "aggregates" in log_spec:
            all_aggregates.extend(log_spec["aggregates"])
    
    # Compute aggregates
    aggregate_metrics = {}
    if all_aggregates:
        aggregate_metrics = compute_aggregates(
            selected_results, results, all_aggregates, aggregate_over
        )
    
    # Prepare parent run metadata
    parent_metadata = {
        "protocol_name": protocol_metadata.get("name") if protocol_metadata else None,
        "protocol_version": protocol_metadata.get("version")
        if protocol_metadata
        else None,
        "inputs": protocol_metadata.get("inputs") if protocol_metadata else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return {
        "parent_metadata": parent_metadata,
        "parent_tags": {"type": "protocol"},
        "child_runs": child_runs,
        "aggregate_metrics": aggregate_metrics,
        "results": results,
    }


def log_to_mlflow(
    prepared_data: Dict[str, Any],
    experiment_name: str,
    mlflow_uri: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Log prepared data to MLflow (Stage 2: MLflow interaction only).
    
    Args:
        prepared_data: Prepared logging data from prepare_logging_data()
        experiment_name: MLflow experiment name
        mlflow_uri: Optional MLflow tracking URI
        dry_run: If True, don't actually log to MLflow
        
    Returns:
        Dictionary with run IDs and status
    """
    _check_mlflow_available()
    
    if dry_run:
        return {
            "dry_run": True,
            "experiment_name": experiment_name,
            "parent_run_id": None,
            "child_run_ids": [],
        }
    
    # Set tracking URI
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    
    # Get or create experiment
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    except Exception:
        # Create experiment if it doesn't exist
        experiment_id = client.create_experiment(experiment_name)
    
    # Generate parent run name from protocol slug + date
    import re
    import hashlib
    
    parent_metadata = prepared_data["parent_metadata"]
    protocol_name = parent_metadata.get("protocol_name") or "Protocol"
    
    # Convert protocol name to slug (lowercase, replace spaces/special chars with hyphens)
    protocol_slug = re.sub(r'[^a-z0-9]+', '-', protocol_name.lower()).strip('-')
    
    # Get date from timestamp
    timestamp = parent_metadata.get("timestamp", "")
    date_part = ""
    if timestamp:
        try:
            date_part = timestamp.split("T")[0]  # Get YYYY-MM-DD
        except:
            pass
    
    # Create parent run name: protocol-slug + date (using hyphen separator)
    if date_part:
        parent_run_name = f"{protocol_slug}-{date_part}"
    else:
        parent_run_name = protocol_slug
    
    # Create parent run
    with mlflow.start_run(experiment_id=experiment_id, run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        
        # Log parent tags
        mlflow.set_tags(prepared_data["parent_tags"])
        
        # Log parent metadata as tags
        parent_metadata = prepared_data["parent_metadata"]
        for key, value in parent_metadata.items():
            if value is not None:
                if key == "inputs" and isinstance(value, dict):
                    # Log inputs as individual tags
                    for input_key, input_value in value.items():
                        mlflow.set_tag(f"input.{input_key}", str(input_value))
                else:
                    mlflow.set_tag(key, str(value))
        
        # Log aggregate metrics to parent run (skip None/NaN/Inf - MLflow rejects them)
        for metric_name, metric_value in prepared_data["aggregate_metrics"].items():
            if metric_value is None:
                continue
            if isinstance(metric_value, float) and (
                metric_value != metric_value or abs(metric_value) == float("inf")
            ):
                continue
            mlflow.log_metric(metric_name, metric_value)
        
        # Log full results as JSONL artifact to parent run
        results_list = prepared_data.get("results", [])
        if results_list:
            import os
            import tempfile
            results_content = "\n".join(
                json.dumps(row, default=str) for row in results_list
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                results_path = os.path.join(temp_dir, "results.jsonl")
                with open(results_path, "w") as f:
                    f.write(results_content)
                mlflow.log_artifact(results_path)
        
        # Create child runs
        child_run_ids = []
        for idx, child_data in enumerate(prepared_data["child_runs"], 1):
            result = child_data["result"]
            
            # Generate sequence ID from sequence content
            child_run_name = None
            if "sequence" in result and result["sequence"]:
                # Hash the sequence and take first 8 characters
                sequence = str(result["sequence"])
                seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:8]
                child_run_name = f"seq-{seq_hash}"
            else:
                # Fallback to index-based naming if no sequence
                child_run_name = f"result-{idx}"
            
            with mlflow.start_run(
                experiment_id=experiment_id, nested=True, run_name=child_run_name
            ) as child_run:
                child_run_id = child_run.info.run_id
                child_run_ids.append(child_run_id)
                
                # Log tags
                child_tags = child_data["tags"].copy()
                child_tags["type"] = "model"
                mlflow.set_tags(child_tags)
                
                # Log parameters (skip None - MLflow rejects them)
                for param_name, param_value in child_data["params"].items():
                    if param_value is not None:
                        mlflow.log_param(param_name, param_value)
                
                # Log metrics (skip None/NaN/Inf - MLflow rejects them)
                for metric_name, metric_value in child_data["metrics"].items():
                    if metric_value is None:
                        continue
                    if isinstance(metric_value, float) and (
                        metric_value != metric_value
                        or abs(metric_value) == float("inf")
                    ):
                        continue
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log artifacts
                for artifact_name, artifact_content in child_data["artifacts"]:
                    # Write to temporary file with the correct name and log
                    import tempfile
                    import os
                    
                    # Create a temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Write file with the correct name in the temp directory
                        file_path = os.path.join(temp_dir, artifact_name)
                        with open(file_path, "w") as f:
                            if isinstance(artifact_content, bytes):
                                f.write(artifact_content.decode("utf-8"))
                            else:
                                f.write(artifact_content)
                        
                        # Log the file directly (artifact_name is the filename, not a directory)
                        mlflow.log_artifact(file_path)
    
    return {
        "dry_run": False,
        "experiment_name": experiment_name,
        "parent_run_id": parent_run_id,
        "child_run_ids": child_run_ids,
    }


def log_protocol_results(
    results: Union[List[Dict], str],
    outputs_config: Union[List[Dict], str, Dict],
    account_name: str,
    workspace_name: str,
    protocol_name: str,
    protocol_metadata: Optional[Dict] = None,
    mlflow_uri: Optional[str] = None,
    dry_run: bool = False,
    aggregate_over: str = "selected",
) -> Dict[str, Any]:
    """Main entry point for logging protocol results to MLflow.
    
    The MLflow experiment name is built as "{account_name}/{workspace_name}/{protocol_name}".
    
    Args:
        results: List of result dicts or path to JSONL file
        outputs_config: Outputs config (list, dict, or file path)
        account_name: Account name for the experiment path
        workspace_name: Workspace name for the experiment path
        protocol_name: Protocol name (slug) for the experiment path
        protocol_metadata: Optional protocol metadata (name, version, inputs, etc.)
        mlflow_uri: Optional MLflow tracking URI (default: https://mlflow.biolm.ai/)
        dry_run: If True, prepare data but don't log to MLflow
        aggregate_over: "selected" or "all" for aggregate computation
        
    Returns:
        Dictionary with logging results (includes experiment_name)
    """
    experiment_name = f"{account_name}/{workspace_name}/{protocol_name}"
    
    # Stage 1: Prepare all data (no MLflow interaction)
    results_list = load_results(results)
    outputs_list = load_outputs_config(outputs_config)
    
    prepared_data = prepare_logging_data(
        results_list, outputs_list, protocol_metadata, aggregate_over
    )
    
    # Stage 2: Log to MLflow (compartmentalized)
    if mlflow_uri is None:
        mlflow_uri = "https://mlflow.biolm.ai/"
    
    logging_result = log_to_mlflow(
        prepared_data, experiment_name, mlflow_uri, dry_run
    )
    
    result_dict = {
        **logging_result,
        "num_results": len(results_list),
        "num_selected": len(prepared_data["child_runs"]),
        "num_aggregates": len(prepared_data["aggregate_metrics"]),
    }
    
    # Include prepared data in dry run mode for detailed output
    if dry_run:
        result_dict["prepared_data"] = prepared_data
    
    return result_dict

