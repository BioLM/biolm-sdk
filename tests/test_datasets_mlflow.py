"""Tests for dataset MLflow functionality."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime

import pytest
from click.testing import CliRunner

from biolm.plugins.mlflow.protocols import MLflowNotAvailableError
from biolm.plugins.mlflow.datasets import (
    list_datasets,
    get_dataset,
    upload_dataset,
    download_dataset,
    _check_mlflow_available,
)
from biolm.plugins.mlflow.dataset_backend import MLflowDatasetBackend
from biolm.datasets import DatasetClient
from biolm.datasets.backends import clear_backends, get_backend
from biolm.cli import cli


class TestMLflowAvailability:
    """Test MLflow availability checks."""
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", False)
    def test_check_mlflow_available_raises_error(self):
        """Test that _check_mlflow_available raises error when MLflow is not available."""
        with pytest.raises(MLflowNotAvailableError):
            _check_mlflow_available()
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    def test_check_mlflow_available_succeeds(self):
        """Test that _check_mlflow_available succeeds when MLflow is available."""
        _check_mlflow_available()  # Should not raise


class TestDatasetOperations:
    """Test dataset operations with mocked MLflow."""
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_list_datasets(self, mock_client_class, mock_mlflow):
        """Test listing datasets."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock runs
        mock_run1 = MagicMock()
        mock_run1.info.run_id = "run-1"
        mock_run1.info.run_name = "dataset-1"
        mock_run1.info.status = "FINISHED"
        mock_run1.info.start_time = 1000000
        mock_run1.info.end_time = 1001000
        mock_run1.data.tags = {"type": "dataset", "dataset_id": "ds-1"}
        mock_run1.data.params = {}
        mock_run1.data.metrics = {}
        
        mock_run2 = MagicMock()
        mock_run2.info.run_id = "run-2"
        mock_run2.info.run_name = "dataset-2"
        mock_run2.info.status = "FINISHED"
        mock_run2.info.start_time = 2000000
        mock_run2.info.end_time = 2001000
        mock_run2.data.tags = {"type": "dataset", "dataset_id": "ds-2"}
        mock_run2.data.params = {}
        mock_run2.data.metrics = {}
        
        mock_client.search_runs.return_value = [mock_run1, mock_run2]
        
        # Mock artifacts
        mock_artifact = MagicMock()
        mock_artifact.path = "file.txt"
        mock_client.list_artifacts.return_value = [mock_artifact]
        
        # Test
        datasets = list_datasets(experiment_name="datasets")
        
        # Assertions
        assert len(datasets) == 2
        assert datasets[0]["dataset_id"] == "ds-1"
        assert datasets[0]["run_id"] == "run-1"
        assert datasets[1]["dataset_id"] == "ds-2"
        assert datasets[1]["run_id"] == "run-2"
        mock_client.search_runs.assert_called_once()
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_get_dataset_by_tag(self, mock_client_class, mock_mlflow):
        """Test getting dataset by dataset_id tag."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock run
        mock_run = MagicMock()
        mock_run.info.run_id = "run-1"
        mock_run.info.run_name = "dataset-1"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1000000
        mock_run.info.end_time = 1001000
        mock_run.data.tags = {"type": "dataset", "dataset_id": "ds-1"}
        mock_run.data.params = {"param1": "value1"}
        mock_run.data.metrics = {"metric1": 0.95}
        
        mock_client.search_runs.return_value = [mock_run]
        
        # Mock artifacts
        mock_artifact = MagicMock()
        mock_artifact.path = "file.txt"
        mock_artifact.is_dir = False
        mock_artifact.file_size = 1024
        mock_client.list_artifacts.return_value = [mock_artifact]
        
        # Test
        dataset = get_dataset("ds-1", experiment_name="datasets")
        
        # Assertions
        assert dataset is not None
        assert dataset["dataset_id"] == "ds-1"
        assert dataset["run_id"] == "run-1"
        assert len(dataset["artifacts"]) == 1
        assert dataset["artifacts"][0]["path"] == "file.txt"
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_get_dataset_by_run_id(self, mock_client_class, mock_mlflow):
        """Test getting dataset by run_id."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock search returns empty (tag search fails)
        mock_client.search_runs.return_value = []
        
        # Mock run by ID
        mock_run = MagicMock()
        mock_run.info.run_id = "run-1"
        mock_run.info.run_name = "dataset-1"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1000000
        mock_run.info.end_time = 1001000
        mock_run.data.tags = {"type": "dataset", "dataset_id": "ds-1"}
        mock_run.data.params = {}
        mock_run.data.metrics = {}
        
        mock_client.get_run.return_value = mock_run
        mock_client.list_artifacts.return_value = []
        
        # Test
        dataset = get_dataset("run-1", experiment_name="datasets")
        
        # Assertions
        assert dataset is not None
        assert dataset["run_id"] == "run-1"
        mock_client.get_run.assert_called_once_with("run-1")
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_get_dataset_not_found(self, mock_client_class, mock_mlflow):
        """Test getting non-existent dataset."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock search returns empty
        mock_client.search_runs.return_value = []
        mock_client.get_run.side_effect = Exception("Run not found")
        
        # Test
        dataset = get_dataset("nonexistent", experiment_name="datasets")
        
        # Assertions
        assert dataset is None
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_upload_dataset_new(self, mock_client_class, mock_mlflow, tmp_path):
        """Test uploading to a new dataset."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.create_experiment.return_value = "exp-123"
        
        # Mock search returns empty (new dataset)
        mock_client.search_runs.return_value = []
        
        # Mock run context
        mock_run = MagicMock()
        mock_run.info.run_id = "run-1"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Test
        result = upload_dataset(
            dataset_id="ds-1",
            file_path=str(test_file),
            experiment_name="datasets",
            name="Test Dataset"
        )
        
        # Assertions
        assert result["dataset_id"] == "ds-1"
        assert result["run_id"] == "run-1"
        assert result["status"] == "success"
        mock_mlflow.set_tags.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_upload_dataset_existing(self, mock_client_class, mock_mlflow, tmp_path):
        """Test uploading to an existing dataset."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock existing run
        mock_existing_run = MagicMock()
        mock_existing_run.info.run_id = "run-1"
        mock_client.search_runs.return_value = [mock_existing_run]
        
        # Mock run context
        mock_mlflow.start_run.return_value.__enter__.return_value = None
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Test
        result = upload_dataset(
            dataset_id="ds-1",
            file_path=str(test_file),
            experiment_name="datasets"
        )
        
        # Assertions
        assert result["dataset_id"] == "ds-1"
        assert result["run_id"] == "run-1"
        mock_mlflow.log_artifact.assert_called_once()
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_upload_dataset_file_not_found(self, mock_client_class, mock_mlflow):
        """Test uploading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            upload_dataset(
                dataset_id="ds-1",
                file_path="/nonexistent/file.txt",
                experiment_name="datasets"
            )
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_download_dataset(self, mock_client_class, mock_mlflow, tmp_path):
        """Test downloading dataset artifacts."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock run
        mock_run = MagicMock()
        mock_run.info.run_id = "run-1"
        mock_run.info.run_name = "dataset-1"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1000000
        mock_run.info.end_time = 1001000
        mock_run.data.tags = {"type": "dataset", "dataset_id": "ds-1"}
        mock_run.data.params = {}
        mock_run.data.metrics = {}
        
        mock_client.search_runs.return_value = [mock_run]
        mock_client.list_artifacts.return_value = []
        mock_client.download_artifacts.return_value = None
        
        # Test
        output_dir = tmp_path / "downloads"
        result = download_dataset(
            dataset_id="ds-1",
            output_path=str(output_dir),
            experiment_name="datasets"
        )
        
        # Assertions
        assert result["dataset_id"] == "ds-1"
        assert result["run_id"] == "run-1"
        assert result["status"] == "success"
        mock_client.download_artifacts.assert_called_once()
    
    @patch("biolm.plugins.mlflow.datasets.MLFLOW_AVAILABLE", True)
    @patch("biolm.plugins.mlflow.datasets.mlflow")
    @patch("biolm.plugins.mlflow.datasets.MlflowClient")
    def test_download_dataset_not_found(self, mock_client_class, mock_mlflow):
        """Test downloading non-existent dataset."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock search returns empty
        mock_client.search_runs.return_value = []
        mock_client.get_run.side_effect = Exception("Run not found")
        
        # Test
        with pytest.raises(ValueError, match="not found"):
            download_dataset(
                dataset_id="nonexistent",
                output_path="./downloads",
                experiment_name="datasets"
            )



class TestMLflowDatasetBackend:
    """Test push/pull adapter with mocked low-level helpers."""

    def teardown_method(self):
        clear_backends()

    @patch("biolm.plugins.mlflow.dataset_backend.MLflowDatasetBackend.push")
    def test_get_backend_mlflow(self, _mock_push):
        backend = get_backend("mlflow")
        assert backend.name == "mlflow"

    @patch("biolm.plugins.mlflow.datasets.upload_dataset")
    @patch("biolm.plugins.mlflow.datasets._check_mlflow_available")
    @patch("biolm.plugins.mlflow.datasets._get_mlflow_client")
    @patch("biolm.plugins.mlflow.datasets._get_or_create_experiment")
    @patch("biolm.plugins.mlflow.datasets._get_default_experiment_name", return_value="datasets")
    def test_backend_push(
        self,
        _exp_name,
        mock_get_exp,
        mock_client_fn,
        mock_check,
        mock_upload,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("biolm.hub.config.read_config", lambda: {})
        mock_upload.return_value = {"dataset_id": "ds-1", "run_id": "run-1", "status": "success"}
        mock_get_exp.return_value = "exp-1"
        mock_client = MagicMock()
        mock_client.search_runs.return_value = []
        mock_client_fn.return_value = mock_client

        client = DatasetClient(primary_root=tmp_path / ".biolm" / "datasets")
        ds = client.create("ds-1", tags=["t"])
        (ds.path / "data" / "f.txt").write_text("x")

        result = MLflowDatasetBackend().push(ds, experiment_name="datasets")
        assert result["dataset_id"] == "ds-1"
        mock_upload.assert_called_once()

    @patch("biolm.plugins.mlflow.datasets.download_dataset")
    @patch("biolm.plugins.mlflow.datasets.get_dataset")
    @patch("biolm.plugins.mlflow.datasets._check_mlflow_available")
    def test_backend_pull(self, mock_check, mock_get, mock_download, tmp_path):
        mock_get.return_value = {
            "dataset_id": "ds-1",
            "run_id": "run-1",
            "tags": {
                "type": "dataset",
                "dataset_id": "ds-1",
                "biolm.dataset_type": "files",
                "biolm.tags": '["a"]',
            },
        }
        mock_download.return_value = {
            "dataset_id": "ds-1",
            "run_id": "run-1",
            "output_path": str(tmp_path / "out"),
            "status": "success",
        }
        dest = tmp_path / "out"
        result = MLflowDatasetBackend().pull("ds-1", dest)
        assert result["dataset_id"] == "ds-1"
        assert (dest / "dataset.yaml").is_file()


class TestCLIPushPull:
    """CLI push/pull against mocked backend."""

    def teardown_method(self):
        clear_backends()

    @patch("biolm.cli.are_credentials_valid", return_value=True)
    @patch.object(MLflowDatasetBackend, "push")
    def test_cli_push(self, mock_push, mock_auth, tmp_path, monkeypatch):
        monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("biolm.hub.config.read_config", lambda: {})
        mock_push.return_value = {"dataset_id": "ds-1", "run_id": "run-1", "backend": "mlflow"}

        root = tmp_path / ".biolm" / "datasets"
        client = DatasetClient(primary_root=root)
        client.create("ds-1")

        # Ensure get_backend returns our class instance that uses the patched method
        clear_backends()
        get_backend("mlflow")  # registers real class; method is patched on class

        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "push", "ds-1", "--backend", "mlflow"])
        assert result.exit_code == 0, result.output
        assert "Pushed" in result.output
        mock_push.assert_called()

    @patch("biolm.cli.are_credentials_valid", return_value=True)
    @patch.object(MLflowDatasetBackend, "pull")
    def test_cli_pull(self, mock_pull, mock_auth, tmp_path, monkeypatch):
        monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("biolm.hub.config.read_config", lambda: {})

        def _pull(dataset_id, dest, **opts):
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            from biolm.datasets.schema import build_meta, write_dataset_yaml
            write_dataset_yaml(dest, build_meta(dataset_id))
            return {"dataset_id": dataset_id, "backend": "mlflow", "path": str(dest)}

        mock_pull.side_effect = _pull
        clear_backends()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dataset", "pull", "remote-ds", "--backend", "mlflow"],
        )
        assert result.exit_code == 0, result.output
        assert "Pulled" in result.output

    @patch("biolm.cli.are_credentials_valid", return_value=False)
    def test_cli_push_requires_auth(self, mock_auth, tmp_path, monkeypatch):
        monkeypatch.setattr("biolm.core.paths.Path.home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("biolm.hub.config.read_config", lambda: {})
        root = tmp_path / ".biolm" / "datasets"
        DatasetClient(primary_root=root).create("ds-1")
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "push", "ds-1", "--backend", "mlflow"])
        assert result.exit_code == 1
        assert "Authentication" in result.output or "authenticate" in result.output.lower()
