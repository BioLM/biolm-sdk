"""MLflow operations for dataset management.

This module provides functionality to manage datasets using MLflow runs.
Each dataset is represented as an MLflow run with specific tags.
"""
import os
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

from biolm.protocols_mlflow import MLflowNotAvailableError


def _get_username() -> Optional[str]:
    """Get current username from API using OAuth token.
    
    Returns:
        Username string if available, None otherwise
    """
    try:
        from biolm.core.const import ACCESS_TOK_PATH, BIOLMAI_BASE_DOMAIN
        from biolm.core.auth import parse_credentials_file
        
        if not os.path.exists(ACCESS_TOK_PATH):
            return None
        
        creds = parse_credentials_file(ACCESS_TOK_PATH)
        if not creds:
            return None
        
        access_token = creds.get("access")
        if not access_token:
            return None
        
        # Make API call to get user info
        import requests
        url = f"{BIOLMAI_BASE_DOMAIN}/api/users/me/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            user_data = resp.json()
            return user_data.get("username")
    except Exception:
        # If anything fails, return None (will fallback to default)
        pass
    
    return None


def _get_default_experiment_name() -> str:
    """Get default experiment name with username prefix.
    
    Returns:
        Experiment name like "{username}/datasets" or "datasets" if username unavailable
    """
    username = _get_username()
    if username:
        return f"{username}/datasets"
    return "datasets"


def _check_mlflow_available():
    """Check if MLflow is available, raise error if not."""
    if not MLFLOW_AVAILABLE:
        raise MLflowNotAvailableError()


def _get_mlflow_client(mlflow_uri: Optional[str] = None) -> MlflowClient:
    """Initialize MLflow client with optional URI.
    
    Args:
        mlflow_uri: Optional MLflow tracking URI. If None, uses default.
        
    Returns:
        MlflowClient instance
    """
    _check_mlflow_available()
    
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    
    return MlflowClient()


def _get_or_create_experiment(client: MlflowClient, experiment_name: str) -> str:
    """Get or create an MLflow experiment.
    
    Args:
        client: MLflow client instance
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID as string
    """
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        return experiment.experiment_id
    except Exception:
        # Experiment doesn't exist, create it
        return client.create_experiment(experiment_name)


def list_datasets(
    experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    all_runs: bool = False
) -> List[Dict[str, Any]]:
    """List all datasets (MLflow runs with type: dataset tag).
    
    Args:
        experiment_name: MLflow experiment name (default: "{username}/datasets")
        mlflow_uri: Optional MLflow tracking URI
        all_runs: If True, list all runs in experiment, not just those with type: dataset tag
        
    Returns:
        List of dataset dictionaries with run information
    """
    _check_mlflow_available()
    
    if experiment_name is None:
        experiment_name = _get_default_experiment_name()
    
    client = _get_mlflow_client(mlflow_uri)
    
    try:
        experiment_id = _get_or_create_experiment(client, experiment_name)
    except Exception as e:
        # If we can't get/create experiment, raise the error instead of silently failing
        raise RuntimeError(f"Failed to get or create experiment '{experiment_name}': {e}") from e
    
    # Search for runs - with or without filter depending on all_runs flag
    try:
        if all_runs:
            # List all runs in the experiment (for debugging)
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                max_results=1000
            )
        else:
            # Search for runs with type: dataset tag
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="tags.type = 'dataset'",
                max_results=1000
            )
    except Exception as e:
        # If search with filter fails, try without filter to see if there are any runs
        if not all_runs:
            try:
                all_runs_list = client.search_runs(
                    experiment_ids=[experiment_id],
                    max_results=1000
                )
                # If there are runs but none match the filter, that's fine
                if len(all_runs_list) == 0:
                    return []
                # If there are runs but filter failed, raise the error
                raise RuntimeError(f"Failed to search runs with filter: {e}") from e
            except Exception as e2:
                # If even basic search fails, raise the error
                raise RuntimeError(f"Failed to search runs in experiment '{experiment_name}': {e2}") from e2
        else:
            # If even basic search fails, raise the error
            raise RuntimeError(f"Failed to search runs in experiment '{experiment_name}': {e}") from e
    
    datasets = []
    for run in runs:
        run_info = run.info
        run_data = run.data
        
        # Extract dataset_id from tags (fallback to run_id)
        dataset_id = run_data.tags.get("dataset_id") or run_info.run_id
        
        # Count artifacts
        artifact_count = 0
        try:
            artifacts = client.list_artifacts(run_info.run_id)
            artifact_count = len(list(artifacts))
        except Exception:
            pass
        
        datasets.append({
            "dataset_id": dataset_id,
            "run_id": run_info.run_id,
            "name": run_info.run_name or dataset_id,
            "status": run_info.status,
            "start_time": run_info.start_time,
            "end_time": run_info.end_time,
            "tags": dict(run_data.tags),
            "params": dict(run_data.params),
            "metrics": dict(run_data.metrics),
            "artifact_count": artifact_count,
        })
    
    return datasets


def get_dataset(
    dataset_id: str,
    experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get a specific dataset by dataset_id tag or run_id.
    
    Args:
        dataset_id: Dataset identifier (tag value or run_id)
        experiment_name: MLflow experiment name (default: "{username}/datasets")
        mlflow_uri: Optional MLflow tracking URI
        
    Returns:
        Dataset dictionary with run information, or None if not found
    """
    _check_mlflow_available()
    
    if experiment_name is None:
        experiment_name = _get_default_experiment_name()
    
    client = _get_mlflow_client(mlflow_uri)
    
    try:
        experiment_id = _get_or_create_experiment(client, experiment_name)
    except Exception:
        return None
    
    # First, try to find by dataset_id tag
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.dataset_id = '{dataset_id}' AND tags.type = 'dataset'",
            max_results=1
        )
        if runs:
            run = runs[0]
        else:
            # Try to get by run_id directly
            try:
                run = client.get_run(dataset_id)
                # Verify it has the dataset tag
                if run.data.tags.get("type") != "dataset":
                    return None
            except Exception:
                return None
    except Exception:
        # Try to get by run_id directly
        try:
            run = client.get_run(dataset_id)
            # Verify it has the dataset tag
            if run.data.tags.get("type") != "dataset":
                return None
        except Exception:
            return None
    
    run_info = run.info
    run_data = run.data
    
    # List artifacts
    artifacts = []
    try:
        artifact_list = client.list_artifacts(run_info.run_id)
        for artifact in artifact_list:
            artifacts.append({
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": getattr(artifact, "file_size", None),
            })
    except Exception:
        pass
    
    return {
        "dataset_id": run_data.tags.get("dataset_id") or run_info.run_id,
        "run_id": run_info.run_id,
        "name": run_info.run_name or dataset_id,
        "status": run_info.status,
        "start_time": run_info.start_time,
        "end_time": run_info.end_time,
        "tags": dict(run_data.tags),
        "params": dict(run_data.params),
        "metrics": dict(run_data.metrics),
        "artifacts": artifacts,
    }


def upload_dataset(
    dataset_id: str,
    file_path: Union[str, Path],
    experiment_name: Optional[str] = None,
    name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    recursive: bool = False
) -> Dict[str, Any]:
    """Upload files as artifacts to a dataset (MLflow run).
    
    Args:
        dataset_id: Dataset identifier
        file_path: Path to file or directory to upload
        experiment_name: MLflow experiment name (default: "{username}/datasets")
        name: Optional dataset name/description (stored as run name)
        mlflow_uri: Optional MLflow tracking URI
        recursive: If True, upload directory recursively
        
    Returns:
        Dictionary with run_id and status information
    """
    _check_mlflow_available()
    
    if experiment_name is None:
        experiment_name = _get_default_experiment_name()
    
    client = _get_mlflow_client(mlflow_uri)
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File or directory not found: {file_path}")
    
    # Get or create experiment
    experiment_id = _get_or_create_experiment(client, experiment_name)
    
    # Check if dataset already exists
    existing_run = None
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.dataset_id = '{dataset_id}' AND tags.type = 'dataset'",
            max_results=1
        )
        if runs:
            existing_run = runs[0]
    except Exception:
        pass
    
    # Use existing run or create new one
    if existing_run:
        run_id = existing_run.info.run_id
        # Start run context to log artifacts
        with mlflow.start_run(run_id=run_id):
            if file_path.is_file():
                mlflow.log_artifact(str(file_path))
            else:
                mlflow.log_artifacts(str(file_path), artifact_path=None)
    else:
        # Create new run
        run_name = name or dataset_id
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            run_id = run.info.run_id
            
            # Set tags
            mlflow.set_tags({
                "type": "dataset",
                "dataset_id": dataset_id,
            })
            
            # Log artifacts
            if file_path.is_file():
                mlflow.log_artifact(str(file_path))
            else:
                mlflow.log_artifacts(str(file_path), artifact_path=None)
    
    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "status": "success",
    }


def download_dataset(
    dataset_id: str,
    output_path: Union[str, Path],
    experiment_name: Optional[str] = None,
    artifact_path: Optional[str] = None,
    mlflow_uri: Optional[str] = None
) -> Dict[str, Any]:
    """Download artifacts from a dataset.
    
    Args:
        dataset_id: Dataset identifier (tag value or run_id)
        output_path: Local directory to save artifacts
        experiment_name: MLflow experiment name (default: "{username}/datasets")
        artifact_path: Optional specific artifact path to download (default: all artifacts)
        mlflow_uri: Optional MLflow tracking URI
        
    Returns:
        Dictionary with download status and path information
    """
    _check_mlflow_available()
    
    if experiment_name is None:
        experiment_name = _get_default_experiment_name()
    
    client = _get_mlflow_client(mlflow_uri)
    output_path = Path(output_path)
    
    # Get dataset
    dataset = get_dataset(dataset_id, experiment_name, mlflow_uri)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_id}' not found")
    
    run_id = dataset["run_id"]
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download artifacts
    try:
        client.download_artifacts(
            run_id=run_id,
            path=artifact_path if artifact_path else "",
            dst_path=str(output_path)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download artifacts: {e}") from e
    
    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "output_path": str(output_path),
        "status": "success",
    }

