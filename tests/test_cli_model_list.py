"""Tests for biolm model list command."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from biolm.cli import cli


@pytest.fixture
def mock_models():
    """Mock model data."""
    return [
        {
            'model_name': 'ESM2-8M',
            'model_slug': 'esm2-8m',
            'encoder': True,
            'predictor': False,
            'generator': False,
        },
        {
            'model_name': 'ESMFold',
            'model_slug': 'esmfold',
            'encoder': False,
            'predictor': True,
            'generator': False,
        },
        {
            'name': 'ABLang2',
            'slug': 'ablang2',
            'encoder': True,
            'predictor': True,
            'generator': False,
        },
    ]


class TestModelList:
    """Test biolm model list command."""
    
    @patch('biolm.cli.list_models')
    def test_list_models_basic(self, mock_list_models, mock_models):
        """Test basic model listing."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list'])
        
        assert result.exit_code == 0
        assert 'ESM2-8M' in result.output
        assert 'ESMFold' in result.output
        assert 'ABLang2' in result.output
    
    @patch('biolm.cli.list_models')
    def test_list_models_filter(self, mock_list_models, mock_models):
        """Test filtering models."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list', '--filter', 'encoder=true'])
        
        assert result.exit_code == 0
        assert 'ESM2-8M' in result.output
        assert 'ABLang2' in result.output
        assert 'ESMFold' not in result.output
    
    @patch('biolm.cli.list_models')
    def test_list_models_sort(self, mock_list_models, mock_models):
        """Test sorting models."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list', '--sort', 'model_name'])
        
        assert result.exit_code == 0
        # Check that output contains model names
        assert 'ESM2-8M' in result.output
    
    @patch('biolm.cli.list_models')
    def test_list_models_json_output(self, mock_list_models, mock_models):
        """Test JSON output format."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list', '--format', 'json'])
        
        assert result.exit_code == 0
        # Should be valid JSON
        output_data = json.loads(result.output)
        assert isinstance(output_data, list)
        assert len(output_data) == 3
    
    @patch('biolm.cli.list_models')
    def test_list_models_csv_output(self, mock_list_models, mock_models):
        """Test CSV output format."""
        mock_list_models.return_value = mock_models
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_file = f.name
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ['model', 'list', '--format', 'csv', '--output', output_file])
            assert result.exit_code == 0
            content = Path(output_file).read_text()
            assert 'model_name' in content or 'model_slug' in content
        finally:
            try:
                Path(output_file).unlink()
            except OSError:
                pass
    
    @patch('biolm.cli.list_models')
    def test_list_models_save_to_file(self, mock_list_models, mock_models):
        """Test saving output to file."""
        mock_list_models.return_value = mock_models
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = f.name
        
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ['model', 'list', '--format', 'json', '--output', output_file])
            
            assert result.exit_code == 0
            assert Path(output_file).exists()
            
            # Verify file contents
            with open(output_file, 'r') as f:
                data = json.load(f)
                assert isinstance(data, list)
                assert len(data) == 3
        finally:
            Path(output_file).unlink()
    
    @patch('biolm.cli.list_models')
    def test_list_models_view_compact(self, mock_list_models, mock_models):
        """Test compact view."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list', '--view', 'compact'])
        
        assert result.exit_code == 0
        assert 'ESM2-8M' in result.output
    
    @patch('biolm.cli.list_models')
    def test_list_models_no_models(self, mock_list_models):
        """Test handling when no models are found."""
        mock_list_models.return_value = []
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list'])
        
        assert result.exit_code == 1
        assert 'No models found' in result.output or 'Error' in result.output
    
    @patch('biolm.cli.list_models')
    def test_list_models_filter_no_matches(self, mock_list_models, mock_models):
        """Test filtering with no matches."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list', '--filter', 'encoder=false', '--filter', 'predictor=false'])
        
        # Should exit with 0 but show no matches message
        assert result.exit_code == 0
    
    @patch('biolm.cli.list_models')
    def test_list_models_invalid_filter(self, mock_list_models, mock_models):
        """Test invalid filter expression."""
        mock_list_models.return_value = mock_models
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list', '--filter', 'invalid'])
        
        assert result.exit_code == 1
        assert 'Invalid filter' in result.output or 'Error' in result.output
    
    @patch('biolm.cli.list_models')
    def test_list_models_network_error(self, mock_list_models):
        """Test handling network errors."""
        mock_list_models.side_effect = Exception("Network error")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'list'])
        
        assert result.exit_code == 1
        assert 'Error' in result.output

