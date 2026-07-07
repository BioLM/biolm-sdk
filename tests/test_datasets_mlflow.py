"""Tests for dataset MLflow functionality."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime

import pytest
from click.testing import CliRunner

from biolm.datasets_mlflow import (
    MLflowNotAvailableError,
    list_datasets,
    get_dataset,
    upload_dataset,
    download_dataset,
    _check_mlflow_available,
    _get_mlflow_client,
    _get_or_create_experiment,
)
from biolm.cli import cli


class TestMLflowAvailability:
    """Test MLflow availability checks."""
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", False)
    def test_check_mlflow_available_raises_error(self):
        """Test that _check_mlflow_available raises error when MLflow is not available."""
        with pytest.raises(MLflowNotAvailableError):
            _check_mlflow_available()
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    def test_check_mlflow_available_succeeds(self):
        """Test that _check_mlflow_available succeeds when MLflow is available."""
        _check_mlflow_available()  # Should not raise


class TestDatasetOperations:
    """Test dataset operations with mocked MLflow."""
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
    def test_upload_dataset_file_not_found(self, mock_client_class, mock_mlflow):
        """Test uploading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            upload_dataset(
                dataset_id="ds-1",
                file_path="/nonexistent/file.txt",
                experiment_name="datasets"
            )
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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
    
    @patch("biolmai.datasets_mlflow.MLFLOW_AVAILABLE", True)
    @patch("biolmai.datasets_mlflow.mlflow")
    @patch("biolmai.datasets_mlflow.MlflowClient")
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


class TestCLIDatasetCommands:
    """Test CLI dataset commands."""
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.list_datasets")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_list(self, mock_auth, mock_list_datasets, mock_mlflow_check):
        """Test CLI dataset list command."""
        mock_auth.return_value = True
        mock_list_datasets.return_value = [
            {
                "dataset_id": "ds-1",
                "run_id": "run-1",
                "name": "Dataset 1",
                "status": "FINISHED",
                "artifact_count": 5,
            }
        ]
        
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "list"])
        
        assert result.exit_code == 0
        assert "Dataset 1" in result.output
        assert "ds-1" in result.output
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.list_datasets")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_list_json(self, mock_auth, mock_list_datasets, mock_mlflow_check):
        """Test CLI dataset list command with JSON output."""
        mock_auth.return_value = True
        mock_list_datasets.return_value = [
            {
                "dataset_id": "ds-1",
                "run_id": "run-1",
                "name": "Dataset 1",
                "status": "FINISHED",
                "artifact_count": 5,
            }
        ]
        
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "list", "--format", "json"])
        
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert isinstance(output_data, list)
        assert len(output_data) == 1
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.list_datasets")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_list_empty(self, mock_auth, mock_list_datasets, mock_mlflow_check):
        """Test CLI dataset list command with no datasets."""
        mock_auth.return_value = True
        mock_list_datasets.return_value = []
        
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "list"])
        
        assert result.exit_code == 0
        assert "No datasets found" in result.output
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.get_dataset")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_show(self, mock_auth, mock_get_dataset, mock_mlflow_check):
        """Test CLI dataset show command."""
        mock_auth.return_value = True
        mock_get_dataset.return_value = {
            "dataset_id": "ds-1",
            "run_id": "run-1",
            "name": "Dataset 1",
            "status": "FINISHED",
            "tags": {"type": "dataset"},
            "params": {},
            "metrics": {},
            "artifacts": [],
        }
        
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "show", "ds-1"])
        
        assert result.exit_code == 0
        assert "Dataset 1" in result.output
        assert "ds-1" in result.output
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.get_dataset")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_show_not_found(self, mock_auth, mock_get_dataset, mock_mlflow_check):
        """Test CLI dataset show command with non-existent dataset."""
        mock_auth.return_value = True
        mock_get_dataset.return_value = None
        
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "show", "nonexistent"])
        
        assert result.exit_code == 1
        assert "not found" in result.output.lower()
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.upload_dataset")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_upload(self, mock_auth, mock_upload, mock_mlflow_check, tmp_path):
        """Test CLI dataset upload command."""
        mock_auth.return_value = True
        mock_upload.return_value = {
            "dataset_id": "ds-1",
            "run_id": "run-1",
            "status": "success",
        }
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dataset", "upload", "ds-1", str(test_file)
        ])
        
        assert result.exit_code == 0
        assert "Successfully uploaded" in result.output
        mock_upload.assert_called_once()
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.download_dataset")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_download(self, mock_auth, mock_download, mock_mlflow_check, tmp_path):
        """Test CLI dataset download command."""
        mock_auth.return_value = True
        mock_download.return_value = {
            "dataset_id": "ds-1",
            "run_id": "run-1",
            "output_path": str(tmp_path),
            "status": "success",
        }
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dataset", "download", "ds-1", str(tmp_path)
        ])
        
        assert result.exit_code == 0
        assert "Successfully downloaded" in result.output
        mock_download.assert_called_once()
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.download_dataset")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_download_not_found(self, mock_auth, mock_download, mock_mlflow_check):
        """Test CLI dataset download command with non-existent dataset."""
        mock_auth.return_value = True
        mock_download.side_effect = ValueError("Dataset 'nonexistent' not found")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dataset", "download", "nonexistent", "./downloads"
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestErrorHandling:
    """Test error handling."""
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_list_not_authenticated(self, mock_auth, mock_mlflow_check):
        """Test CLI dataset list without authentication (are_credentials_valid returns False)."""
        mock_auth.return_value = False

        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "list"])

        assert result.exit_code == 1
        assert "Authentication" in result.output or "Not Authenticated" in result.output or "authenticate" in result.output.lower()
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_list_mlflow_not_available(self, mock_auth, mock_check):
        """Test CLI dataset list without MLflow."""
        mock_auth.return_value = True
        mock_check.side_effect = MLflowNotAvailableError()
        
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset", "list"])
        
        assert result.exit_code == 1
        assert "MLflow" in result.output
    
    @patch("biolmai.datasets_mlflow._check_mlflow_available")
    @patch("biolm.cli.upload_dataset")
    @patch("biolm.cli.are_credentials_valid")
    def test_cli_dataset_upload_file_not_found(self, mock_auth, mock_upload, mock_mlflow_check):
        """Test CLI dataset upload with non-existent file."""
        mock_auth.return_value = True
        mock_upload.side_effect = FileNotFoundError("File not found")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "dataset", "upload", "ds-1", "/nonexistent/file.txt"
        ])
        
        # Click may use exit code 2 for usage/file errors
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

