"""Tests for biolm model run command."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from click.testing import CliRunner

from biolm.cli import cli


@pytest.fixture
def mock_model():
    """Mock Model instance."""
    model = MagicMock()
    model.encode = MagicMock(return_value=[{'embedding': [0.1, 0.2, 0.3]}])
    model.predict = MagicMock(return_value=[{'prediction': 'result'}])
    model.generate = MagicMock(return_value=[{'generated': 'sequence'}])
    model.lookup = MagicMock(return_value=[{'lookup': 'result'}])
    return model


@pytest.fixture
def sample_fasta_file(tmp_path):
    """Create a sample FASTA file."""
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(">seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nMSILVTRPSPAGEEL\n")
    return str(fasta_file)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("sequence,id\nACDEFGHIKLMNPQRSTVWY,seq1\nMSILVTRPSPAGEEL,seq2\n")
    return str(csv_file)


@pytest.fixture
def sample_json_file(tmp_path):
    """Create a sample JSON file."""
    json_file = tmp_path / "test.json"
    data = [
        {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"},
        {"sequence": "MSILVTRPSPAGEEL", "id": "seq2"}
    ]
    json_file.write_text(json.dumps(data))
    return str(json_file)


class TestModelRun:
    """Test biolm model run command."""
    
    @patch('biolm.cli.Model')
    def test_run_encode_with_fasta(self, mock_model_class, sample_fasta_file, mock_model):
        """Test running encode action with FASTA input."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_run_predict_with_csv(self, mock_model_class, sample_csv_file, mock_model):
        """Test running predict action with CSV input."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esmfold', 'predict',
            '--input', sample_csv_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.predict.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_run_with_api_items_envelope(self, mock_model_class, tmp_path, mock_model):
        """Test JSON files using API request envelope format."""
        mock_model_class.return_value = mock_model

        json_file = tmp_path / "envelope.json"
        json_file.write_text('{"items": [{"sequence": "ACDEFGHIKLMNPQRSTVWY"}]}')

        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', str(json_file),
            '--output', '-'
        ])

        assert result.exit_code == 0
        call_args = mock_model.encode.call_args
        assert call_args[1]['items'] == [{"sequence": "ACDEFGHIKLMNPQRSTVWY"}]

    @patch('biolm.cli.Model')
    def test_run_with_json(self, mock_model_class, sample_json_file, mock_model):
        """Test running with JSON input."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_json_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_run_with_params(self, mock_model_class, sample_fasta_file, mock_model):
        """Test running with parameters."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--params', '{"normalize": true}',
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        # Check that params were passed
        call_args = mock_model.encode.call_args
        assert call_args[1]['params'] == {'normalize': True}
    
    @patch('biolm.cli.Model')
    def test_run_with_params_file(self, mock_model_class, sample_fasta_file, tmp_path, mock_model):
        """Test running with parameters from file."""
        mock_model_class.return_value = mock_model
        
        params_file = tmp_path / "params.json"
        params_file.write_text('{"normalize": true, "max_length": 100}')
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--params', str(params_file),
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        call_args = mock_model.encode.call_args
        assert call_args[1]['params']['normalize'] is True
    
    @patch('biolm.cli.Model')
    def test_run_save_to_file(self, mock_model_class, sample_fasta_file, tmp_path, mock_model):
        """Test saving output to file."""
        mock_model_class.return_value = mock_model
        
        output_file = tmp_path / "output.json"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
    
    @patch('biolm.cli.Model')
    def test_run_different_actions(self, mock_model_class, sample_fasta_file, mock_model):
        """Test different actions (encode, predict, generate, lookup)."""
        mock_model_class.return_value = mock_model
        
        actions = ['encode', 'predict', 'generate']
        for action in actions:
            runner = CliRunner()
            result = runner.invoke(cli, [
                'model', 'run', 'esm2-8m', action,
                '--input', sample_fasta_file,
                '--output', '-'
            ])
            
            assert result.exit_code == 0
            getattr(mock_model, action).assert_called()
            mock_model.reset_mock()
    
    @patch('biolm.cli.Model')
    def test_run_lookup_action(self, mock_model_class, sample_json_file, mock_model):
        """Test lookup action."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'lookup',
            '--input', sample_json_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.lookup.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_run_with_output_format(self, mock_model_class, sample_fasta_file, tmp_path, mock_model):
        """Test specifying output format."""
        mock_model_class.return_value = mock_model
        # encode must return fasta-compatible dicts (with 'sequence' key) for to_fasta
        mock_model.encode.return_value = [
            {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1'},
            {'sequence': 'MSILVTRPSPAGEEL', 'id': 'seq2'},
        ]
        
        output_file = tmp_path / "output.fasta"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', str(output_file),
            '--format', 'fasta'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
    
    @patch('biolm.cli.Model')
    def test_run_missing_input(self, mock_model_class, mock_model):
        """Test error when input is missing."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode'
        ])
        
        assert result.exit_code == 1
        assert 'Input is required' in result.output or 'input' in result.output.lower()
    
    @patch('biolm.cli.Model')
    def test_run_invalid_file(self, mock_model_class, mock_model):
        """Test error with invalid file path."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', '/nonexistent/file.fasta'
        ])
        
        assert result.exit_code == 1
        assert 'not found' in result.output.lower() or 'Error' in result.output
    
    @patch('biolm.cli.Model')
    def test_run_invalid_params_json(self, mock_model_class, sample_fasta_file, mock_model):
        """Test error with invalid JSON in params."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--params', 'invalid json'
        ])
        
        assert result.exit_code == 1
        assert 'Invalid JSON' in result.output or 'Error' in result.output
    
    @patch('biolm.cli.Model')
    @patch('biolm.core.http.BioLMApiClient')
    def test_run_with_batch_size(self, mock_client_class, mock_model_class, sample_fasta_file, mock_model):
        """Test running with batch size detection."""
        mock_model_class.return_value = mock_model
        
        # Mock schema fetching
        mock_client = AsyncMock()
        mock_client.schema = AsyncMock(return_value={
            'properties': {
                'items': {
                    'maxItems': 50
                }
            }
        })
        mock_client.shutdown = AsyncMock()
        mock_client_class.return_value = mock_client
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
    
    @patch('biolm.cli.Model')
    def test_run_with_progress(self, mock_model_class, sample_fasta_file, mock_model):
        """Test running with progress bar."""
        # Make model return multiple results to trigger progress
        mock_model.encode.return_value = [
            {'embedding': [0.1, 0.2]} for _ in range(10)
        ]
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--progress',
            '--output', '-'
        ])
        
        assert result.exit_code == 0
    
    @patch('biolm.cli.Model')
    def test_run_with_type_override(self, mock_model_class, sample_csv_file, mock_model):
        """Test running with type override."""
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_csv_file,
            '--type', 'sequence',
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_run_api_error(self, mock_model_class, sample_fasta_file):
        """Test handling API errors."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("API error")
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file
        ])
        
        assert result.exit_code == 1
        assert 'Error' in result.output

