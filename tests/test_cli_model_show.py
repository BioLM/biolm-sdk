"""Tests for biolm model show command."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from click.testing import CliRunner

from biolm.cli import cli


@pytest.fixture
def mock_model():
    """Mock model data."""
    return {
        'model_name': 'ESM2-8M',
        'model_slug': 'esm2-8m',
        'encoder': True,
        'predictor': False,
        'generator': False,
        'description': 'ESM2 8M parameter model',
    }


@pytest.fixture
def mock_models(mock_model):
    """Mock models list."""
    return [mock_model]


class TestModelShow:
    """Test biolm model show command."""
    
    @patch('biolm.cli.list_models')
    def test_show_model_basic(self, mock_list_models, mock_models):
        """Test basic model show."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'show', 'esm2-8m'])
        
        assert result.exit_code == 0
        assert 'ESM2-8M' in result.output
        assert 'esm2-8m' in result.output
    
    @patch('biolm.cli.list_models')
    def test_show_model_not_found(self, mock_list_models):
        """Test showing non-existent model."""
        mock_list_models.return_value = []
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'show', 'nonexistent'])
        
        assert result.exit_code == 1
        # CLI may show "not found" or "could not fetch models"
        assert 'not found' in result.output.lower() or 'could not fetch' in result.output.lower()
    
    @patch('biolm.cli.list_models')
    def test_show_model_json_output(self, mock_list_models, mock_models):
        """Test JSON output format."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'show', 'esm2-8m', '--format', 'json'])
        
        assert result.exit_code == 0
        # Should be valid JSON
        output_data = json.loads(result.output)
        assert 'model_name' in output_data or 'model_slug' in output_data
    
    @patch('biolm.cli.list_models')
    def test_show_model_save_to_file(self, mock_list_models, mock_models):
        """Test saving output to file."""
        mock_list_models.return_value = mock_models
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = f.name
        
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ['model', 'show', 'esm2-8m', '--format', 'json', '--output', output_file])
            
            assert result.exit_code == 0
            assert Path(output_file).exists()
            
            # Verify file contents
            with open(output_file, 'r') as f:
                data = json.load(f)
                assert 'model_name' in data or 'model_slug' in data
        finally:
            Path(output_file).unlink()
    
    @patch('biolm.cli.list_models')
    @patch('biolmai.core.http.BioLMApiClient')
    def test_show_model_with_schemas(self, mock_client_class, mock_list_models, mock_models):
        """Test showing model with schemas."""
        mock_list_models.return_value = mock_models
        
        # Mock async client
        mock_client = AsyncMock()
        mock_client.schema = AsyncMock(return_value={'type': 'object', 'properties': {}})
        mock_client.shutdown = AsyncMock()
        mock_client_class.return_value = mock_client
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'show', 'esm2-8m', '--include-schemas'])
        
        assert result.exit_code == 0
        assert 'ESM2-8M' in result.output
    
    @patch('biolm.cli.list_models')
    def test_show_model_by_name(self, mock_list_models, mock_models):
        """Test finding model by name instead of slug."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'show', 'ESM2-8M'])
        
        assert result.exit_code == 0
        assert 'ESM2-8M' in result.output
    
    @patch('biolm.cli.list_models')
    def test_show_model_network_error(self, mock_list_models):
        """Test handling network errors."""
        mock_list_models.side_effect = Exception("Network error")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'show', 'esm2-8m'])
        
        assert result.exit_code == 1
        assert 'Error' in result.output

