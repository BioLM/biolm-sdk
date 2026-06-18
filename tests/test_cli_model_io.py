"""Tests for CLI model IO integration."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from biolm.cli import cli


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


@pytest.fixture
def sample_jsonl_file(tmp_path):
    """Create a sample JSONL file."""
    jsonl_file = tmp_path / "test.jsonl"
    lines = [
        json.dumps({"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"}),
        json.dumps({"sequence": "MSILVTRPSPAGEEL", "id": "seq2"})
    ]
    jsonl_file.write_text("\n".join(lines))
    return str(jsonl_file)


@pytest.fixture
def sample_pdb_file(tmp_path):
    """Create a sample PDB file."""
    pdb_file = tmp_path / "test.pdb"
    pdb_content = """ATOM      1  N   MET A   1      20.154  16.967  10.410  1.00 20.00           N
ATOM      2  CA  MET A   1      21.477  16.967   9.789  1.00 20.00           C
"""
    pdb_file.write_text(pdb_content)
    return str(pdb_file)


class TestModelIO:
    """Test IO module integration with CLI."""
    
    @patch('biolm.cli.Model')
    def test_format_auto_detection_fasta(self, mock_model_class, sample_fasta_file):
        """Test auto-detection of FASTA format."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [{'embedding': [0.1, 0.2]}]
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        # Should successfully load FASTA
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_format_auto_detection_csv(self, mock_model_class, sample_csv_file):
        """Test auto-detection of CSV format."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [{'embedding': [0.1, 0.2]}]
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_csv_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_format_auto_detection_json(self, mock_model_class, sample_json_file):
        """Test auto-detection of JSON format."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [{'embedding': [0.1, 0.2]}]
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
    def test_format_auto_detection_jsonl(self, mock_model_class, sample_jsonl_file):
        """Test auto-detection of JSONL format."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [{'embedding': [0.1, 0.2]}]
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_jsonl_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_format_auto_detection_pdb(self, mock_model_class, sample_pdb_file):
        """Test auto-detection of PDB format."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [{'embedding': [0.1, 0.2]}]
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_pdb_file,
            '--output', '-'
        ])
        
        assert result.exit_code == 0
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_output_format_json(self, mock_model_class, sample_fasta_file, tmp_path):
        """Test output format JSON."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            {'embedding': [0.1, 0.2], 'id': 'seq1'},
            {'embedding': [0.3, 0.4], 'id': 'seq2'}
        ]
        mock_model_class.return_value = mock_model
        
        output_file = tmp_path / "output.json"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', str(output_file),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        # Verify it's valid JSON
        with open(output_file) as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2
    
    @patch('biolm.cli.Model')
    def test_output_format_fasta(self, mock_model_class, sample_fasta_file, tmp_path):
        """Test output format FASTA."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1'},
            {'sequence': 'MSILVTRPSPAGEEL', 'id': 'seq2'}
        ]
        mock_model_class.return_value = mock_model
        
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
        # Verify it's valid FASTA
        content = output_file.read_text()
        assert content.startswith('>')
        assert 'ACDEFGHIKLMNPQRSTVWY' in content
    
    @patch('biolm.cli.Model')
    def test_output_format_csv(self, mock_model_class, sample_fasta_file, tmp_path):
        """Test output format CSV."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            {'embedding': [0.1, 0.2], 'id': 'seq1'},
            {'embedding': [0.3, 0.4], 'id': 'seq2'}
        ]
        mock_model_class.return_value = mock_model
        
        output_file = tmp_path / "output.csv"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', str(output_file),
            '--format', 'csv'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        # Verify it's valid CSV
        content = output_file.read_text()
        assert ',' in content
        assert 'id' in content or 'embedding' in content
    
    @patch('biolm.cli.Model')
    def test_output_format_auto_detect_from_extension(self, mock_model_class, sample_fasta_file, tmp_path):
        """Test output format auto-detection from file extension."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1'}
        ]
        mock_model_class.return_value = mock_model
        
        output_file = tmp_path / "output.json"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', str(output_file)
            # No --format specified, should auto-detect from .json extension
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        # Should be JSON format
        with open(output_file) as f:
            data = json.load(f)
            assert isinstance(data, list)
    
    @patch('biolm.cli.Model')
    def test_stdin_input_requires_format(self, mock_model_class):
        """Test that stdin input requires format specification."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', '-'
        ], input='{"sequence": "ACDEFGHIKLMNPQRSTVWY"}')
        
        assert result.exit_code == 1
        assert 'Format must be specified' in result.output or 'format' in result.output.lower()
    
    @patch('biolm.cli.Model')
    def test_stdin_input_with_format(self, mock_model_class, tmp_path):
        """Test stdin input with format specified."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [{'embedding': [0.1, 0.2]}]
        mock_model_class.return_value = mock_model
        
        output_file = tmp_path / "output.json"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', '-',
            '--format', 'json',
            '--output', str(output_file)
        ], input='[{"sequence": "ACDEFGHIKLMNPQRSTVWY"}]')
        
        assert result.exit_code == 0
        assert output_file.exists()
        mock_model.encode.assert_called_once()
    
    @patch('biolm.cli.Model')
    def test_file_format_conversion_fasta_to_json(self, mock_model_class, sample_fasta_file, tmp_path):
        """Test converting from FASTA input to JSON output."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            {'embedding': [0.1, 0.2], 'id': 'seq1'},
            {'embedding': [0.3, 0.4], 'id': 'seq2'}
        ]
        mock_model_class.return_value = mock_model
        
        output_file = tmp_path / "output.json"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_fasta_file,
            '--output', str(output_file),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        # Verify JSON output
        with open(output_file) as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2
    
    @patch('biolm.cli.Model')
    def test_file_format_conversion_csv_to_fasta(self, mock_model_class, sample_csv_file, tmp_path):
        """Test converting from CSV input to FASTA output."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            {'sequence': 'ACDEFGHIKLMNPQRSTVWY', 'id': 'seq1'},
            {'sequence': 'MSILVTRPSPAGEEL', 'id': 'seq2'}
        ]
        mock_model_class.return_value = mock_model
        
        output_file = tmp_path / "output.fasta"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', sample_csv_file,
            '--output', str(output_file),
            '--format', 'fasta'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        # Verify FASTA output
        content = output_file.read_text()
        assert content.startswith('>')
        assert 'ACDEFGHIKLMNPQRSTVWY' in content
    
    @patch('biolm.cli.Model')
    def test_unknown_format_error(self, mock_model_class, tmp_path):
        """Test error handling for unknown format."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        input_file = tmp_path / "test.unknown"
        input_file.write_text("some data")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'model', 'run', 'esm2-8m', 'encode',
            '--input', str(input_file)
        ])
        
        assert result.exit_code == 1
        assert 'format' in result.output.lower() or 'Error' in result.output

