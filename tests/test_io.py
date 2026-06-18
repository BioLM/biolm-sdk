"""Tests for biolmai.io module."""
import io
import tempfile
from pathlib import Path

import pytest

from biolm.io import load_csv, load_fasta, load_pdb, to_csv, to_fasta, to_pdb


class TestLoadFasta:
    """Tests for load_fasta function."""

    def test_load_single_sequence(self, tmp_path):
        """Test loading a single sequence FASTA file."""
        fasta_file = tmp_path / "single.fasta"
        fasta_file.write_text(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
        
        result = load_fasta(fasta_file)
        
        assert len(result) == 1
        assert result[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
        assert result[0]["id"] == "seq1"

    def test_load_multi_sequence(self):
        """Test loading a multi-sequence FASTA file."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        fasta_file = fixtures_dir / "sample.fasta"
        
        result = load_fasta(fasta_file)
        
        assert len(result) == 4
        assert result[0]["id"] == "seq1"
        assert result[1]["id"] == "seq2"
        assert result[2]["id"] == "seq3"
        assert result[3]["id"] == "seq4"

    def test_load_multi_line_sequence(self, tmp_path):
        """Test loading FASTA with wrapped sequences."""
        fasta_file = tmp_path / "wrapped.fasta"
        fasta_file.write_text(">seq1\nACDEFG\nHIKLMN\nPQRSTV\nWY\n")
        
        result = load_fasta(fasta_file)
        
        assert len(result) == 1
        assert result[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"

    def test_load_fasta_with_metadata(self, tmp_path):
        """Test loading FASTA with pipe-separated metadata."""
        fasta_file = tmp_path / "metadata.fasta"
        fasta_file.write_text(">seq1|protein|test\nACDEFGHIKLMNPQRSTVWY\n")
        
        result = load_fasta(fasta_file)
        
        assert len(result) == 1
        assert result[0]["id"] == "seq1"
        assert "metadata_1" in result[0]["metadata"]
        assert result[0]["metadata"]["metadata_1"] == "protein"

    def test_load_fasta_with_description(self, tmp_path):
        """Test loading FASTA with space-separated description."""
        fasta_file = tmp_path / "desc.fasta"
        fasta_file.write_text(">seq1 description here\nACDEFGHIKLMNPQRSTVWY\n")
        
        result = load_fasta(fasta_file)
        
        assert len(result) == 1
        assert result[0]["id"] == "seq1"
        assert result[0]["metadata"]["description"] == "description here"

    def test_load_fasta_no_header(self, tmp_path):
        """Test loading FASTA without header (generates ID)."""
        fasta_file = tmp_path / "no_header.fasta"
        fasta_file.write_text("ACDEFGHIKLMNPQRSTVWY\n")
        
        result = load_fasta(fasta_file)
        
        assert len(result) == 1
        assert result[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
        assert result[0]["id"] == "sequence_1"

    def test_load_fasta_file_like_object(self):
        """Test loading from file-like object."""
        file_obj = io.StringIO(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
        
        result = load_fasta(file_obj)
        
        assert len(result) == 1
        assert result[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"

    def test_load_fasta_empty_file(self, tmp_path):
        """Test loading empty FASTA file raises error."""
        fasta_file = tmp_path / "empty.fasta"
        fasta_file.write_text("")
        
        with pytest.raises(ValueError, match="empty"):
            load_fasta(fasta_file)

    def test_load_fasta_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_fasta("nonexistent.fasta")


class TestToFasta:
    """Tests for to_fasta function."""

    def test_to_fasta_single_sequence(self, tmp_path):
        """Test writing a single sequence to FASTA."""
        data = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"}]
        output_file = tmp_path / "output.fasta"
        
        to_fasta(data, output_file)
        
        content = output_file.read_text()
        assert ">seq1" in content
        assert "ACDEFGHIKLMNPQRSTVWY" in content

    def test_to_fasta_multi_sequence(self, tmp_path):
        """Test writing multiple sequences to FASTA."""
        data = [
            {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"},
            {"sequence": "MKTAYIAKQRQISFVKSHFSRQ", "id": "seq2"},
        ]
        output_file = tmp_path / "output.fasta"
        
        to_fasta(data, output_file)
        
        content = output_file.read_text()
        assert ">seq1" in content
        assert ">seq2" in content
        assert "ACDEFGHIKLMNPQRSTVWY" in content
        assert "MKTAYIAKQRQISFVKSHFSRQ" in content

    def test_to_fasta_with_metadata(self, tmp_path):
        """Test writing FASTA with metadata."""
        data = [{
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "id": "seq1",
            "metadata": {"description": "Test sequence", "type": "protein"},
        }]
        output_file = tmp_path / "output.fasta"
        
        to_fasta(data, output_file)
        
        content = output_file.read_text()
        assert ">seq1" in content
        # Metadata should be in header
        assert "Test sequence" in content or "protein" in content

    def test_to_fasta_custom_sequence_key(self, tmp_path):
        """Test writing FASTA with custom sequence key."""
        data = [{"seq": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"}]
        output_file = tmp_path / "output.fasta"
        
        to_fasta(data, output_file, sequence_key="seq")
        
        content = output_file.read_text()
        assert "ACDEFGHIKLMNPQRSTVWY" in content

    def test_to_fasta_file_like_object(self):
        """Test writing to file-like object."""
        data = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"}]
        file_obj = io.StringIO()
        
        to_fasta(data, file_obj)
        
        content = file_obj.getvalue()
        assert ">seq1" in content
        assert "ACDEFGHIKLMNPQRSTVWY" in content

    def test_to_fasta_missing_sequence_key(self, tmp_path):
        """Test writing FASTA with missing sequence key raises error."""
        data = [{"id": "seq1"}]  # Missing sequence
        output_file = tmp_path / "output.fasta"
        
        with pytest.raises(ValueError, match="missing required key"):
            to_fasta(data, output_file)

    def test_to_fasta_empty_data(self, tmp_path):
        """Test writing empty data raises error."""
        output_file = tmp_path / "output.fasta"
        
        with pytest.raises(ValueError, match="empty"):
            to_fasta([], output_file)

    def test_to_fasta_round_trip(self, tmp_path):
        """Test round-trip: load → write → load."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        input_file = fixtures_dir / "sample.fasta"
        
        # Load original
        original = load_fasta(input_file)
        
        # Write to temp file
        output_file = tmp_path / "roundtrip.fasta"
        to_fasta(original, output_file)
        
        # Load back
        result = load_fasta(output_file)
        
        assert len(result) == len(original)
        for orig, res in zip(original, result):
            assert orig["sequence"] == res["sequence"]
            assert orig["id"] == res["id"]


class TestLoadCsv:
    """Tests for load_csv function."""

    def test_load_csv_with_headers(self):
        """Test loading CSV with headers."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        csv_file = fixtures_dir / "sample.csv"
        
        result = load_csv(csv_file)
        
        assert len(result) == 3
        assert "sequence" in result[0]
        assert "id" in result[0]
        assert "score" in result[0]
        assert result[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"

    def test_load_csv_validate_sequence_key(self):
        """Test loading CSV with sequence_key validation."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        csv_file = fixtures_dir / "sample.csv"
        
        result = load_csv(csv_file, sequence_key="sequence")
        
        assert len(result) == 3
        assert all("sequence" in item for item in result)

    def test_load_csv_missing_sequence_key(self, tmp_path):
        """Test loading CSV with missing sequence_key raises error."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,score\nseq1,0.95\n")
        
        with pytest.raises(ValueError, match="missing required column"):
            load_csv(csv_file, sequence_key="sequence")

    def test_load_csv_file_like_object(self):
        """Test loading from file-like object."""
        file_obj = io.StringIO("sequence,id\nACDEFGHIKLMNPQRSTVWY,seq1\n")
        
        result = load_csv(file_obj)
        
        assert len(result) == 1
        assert result[0]["sequence"] == "ACDEFGHIKLMNPQRSTVWY"

    def test_load_csv_empty_file(self, tmp_path):
        """Test loading empty CSV file raises error."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        with pytest.raises(ValueError, match="empty"):
            load_csv(csv_file)

    def test_load_csv_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")


class TestToCsv:
    """Tests for to_csv function."""

    def test_to_csv_simple(self, tmp_path):
        """Test writing simple data to CSV."""
        data = [
            {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1", "score": "0.95"},
            {"sequence": "MKTAYIAKQRQISFVKSHFSRQ", "id": "seq2", "score": "0.87"},
        ]
        output_file = tmp_path / "output.csv"
        
        to_csv(data, output_file)
        
        # Verify file was created and has content
        assert output_file.exists()
        content = output_file.read_text()
        assert "sequence" in content
        assert "ACDEFGHIKLMNPQRSTVWY" in content

    def test_to_csv_custom_fieldnames(self, tmp_path):
        """Test writing CSV with custom fieldnames."""
        data = [
            {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1", "extra": "value"},
        ]
        output_file = tmp_path / "output.csv"
        
        to_csv(data, output_file, fieldnames=["sequence", "id"])
        
        content = output_file.read_text()
        # Should only have sequence and id columns
        lines = content.strip().split("\n")
        assert "sequence,id" in lines[0]
        assert "extra" not in content

    def test_to_csv_missing_keys(self, tmp_path):
        """Test writing CSV with missing keys (filled with empty strings)."""
        data = [
            {"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"},
            {"sequence": "MKTAYIAKQRQISFVKSHFSRQ"},  # Missing id
        ]
        output_file = tmp_path / "output.csv"
        
        to_csv(data, output_file)
        
        # Should not raise error, missing keys filled with empty strings
        content = output_file.read_text()
        assert "sequence" in content

    def test_to_csv_file_like_object(self):
        """Test writing to file-like object."""
        data = [{"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "seq1"}]
        file_obj = io.StringIO()
        
        to_csv(data, file_obj)
        
        content = file_obj.getvalue()
        assert "sequence" in content
        assert "ACDEFGHIKLMNPQRSTVWY" in content

    def test_to_csv_empty_data(self, tmp_path):
        """Test writing empty data raises error."""
        output_file = tmp_path / "output.csv"
        
        with pytest.raises(ValueError, match="empty"):
            to_csv([], output_file)

    def test_to_csv_round_trip(self, tmp_path):
        """Test round-trip: load → write → load."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        input_file = fixtures_dir / "sample.csv"
        
        # Load original
        original = load_csv(input_file)
        
        # Write to temp file
        output_file = tmp_path / "roundtrip.csv"
        to_csv(original, output_file)
        
        # Load back
        result = load_csv(output_file)
        
        assert len(result) == len(original)
        # Compare keys (values may differ slightly due to CSV parsing)
        assert set(result[0].keys()) == set(original[0].keys())


class TestLoadPdb:
    """Tests for load_pdb function."""

    def test_load_single_model_pdb(self):
        """Test loading single-model PDB file."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        pdb_file = fixtures_dir / "sample.pdb"
        
        result = load_pdb(pdb_file)
        
        assert len(result) == 1
        assert "pdb" in result[0]
        assert "ATOM" in result[0]["pdb"]
        assert "HEADER" in result[0]["pdb"]

    def test_load_multi_model_pdb(self):
        """Test loading multi-model PDB file."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        pdb_file = fixtures_dir / "multi_model.pdb"
        
        result = load_pdb(pdb_file)
        
        assert len(result) == 2
        assert all("pdb" in item for item in result)
        assert "MODEL        1" in result[0]["pdb"]
        assert "MODEL        2" in result[1]["pdb"]

    def test_load_pdb_file_like_object(self):
        """Test loading from file-like object."""
        pdb_content = "HEADER    TEST\nATOM      1  N   MET A   1\nEND\n"
        file_obj = io.StringIO(pdb_content)
        
        result = load_pdb(file_obj)
        
        assert len(result) == 1
        assert "pdb" in result[0]

    def test_load_pdb_empty_file(self, tmp_path):
        """Test loading empty PDB file raises error."""
        pdb_file = tmp_path / "empty.pdb"
        pdb_file.write_text("")
        
        with pytest.raises(ValueError, match="empty"):
            load_pdb(pdb_file)

    def test_load_pdb_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_pdb("nonexistent.pdb")


class TestToPdb:
    """Tests for to_pdb function."""

    def test_to_pdb_single_structure(self, tmp_path):
        """Test writing a single structure to PDB."""
        data = [{"pdb": "HEADER    TEST\nATOM      1  N   MET A   1\nEND\n"}]
        output_file = tmp_path / "output.pdb"
        
        to_pdb(data, output_file)
        
        content = output_file.read_text()
        assert "HEADER" in content
        assert "ATOM" in content

    def test_to_pdb_multiple_structures(self, tmp_path):
        """Test writing multiple structures to PDB (concatenated)."""
        data = [
            {"pdb": "HEADER    TEST1\nATOM      1  N   MET A   1\nEND\n"},
            {"pdb": "HEADER    TEST2\nATOM      1  N   MET A   1\nEND\n"},
        ]
        output_file = tmp_path / "output.pdb"
        
        to_pdb(data, output_file)
        
        content = output_file.read_text()
        assert "TEST1" in content
        assert "TEST2" in content

    def test_to_pdb_custom_pdb_key(self, tmp_path):
        """Test writing PDB with custom pdb key."""
        data = [{"structure": "HEADER    TEST\nATOM      1  N   MET A   1\nEND\n"}]
        output_file = tmp_path / "output.pdb"
        
        to_pdb(data, output_file, pdb_key="structure")
        
        content = output_file.read_text()
        assert "HEADER" in content

    def test_to_pdb_file_like_object(self):
        """Test writing to file-like object."""
        data = [{"pdb": "HEADER    TEST\nATOM      1  N   MET A   1\nEND\n"}]
        file_obj = io.StringIO()
        
        to_pdb(data, file_obj)
        
        content = file_obj.getvalue()
        assert "HEADER" in content
        assert "ATOM" in content

    def test_to_pdb_missing_pdb_key(self, tmp_path):
        """Test writing PDB with missing pdb key raises error."""
        data = [{"id": "seq1"}]  # Missing pdb
        output_file = tmp_path / "output.pdb"
        
        with pytest.raises(ValueError, match="missing required key"):
            to_pdb(data, output_file)

    def test_to_pdb_empty_data(self, tmp_path):
        """Test writing empty data raises error."""
        output_file = tmp_path / "output.pdb"
        
        with pytest.raises(ValueError, match="empty"):
            to_pdb([], output_file)

    def test_to_pdb_round_trip(self, tmp_path):
        """Test round-trip: load → write → load."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        input_file = fixtures_dir / "sample.pdb"
        
        # Load original
        original = load_pdb(input_file)
        
        # Write to temp file
        output_file = tmp_path / "roundtrip.pdb"
        to_pdb(original, output_file)
        
        # Load back
        result = load_pdb(output_file)
        
        assert len(result) == len(original)
        # Compare content (may have whitespace differences)
        assert "ATOM" in result[0]["pdb"]


class TestIntegration:
    """Integration tests with Model class."""

    def test_fasta_to_api_format(self):
        """Test that FASTA output is compatible with API format."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        fasta_file = fixtures_dir / "sample.fasta"
        
        items = load_fasta(fasta_file)
        
        # Verify structure is correct for API
        assert isinstance(items, list)
        assert all(isinstance(item, dict) for item in items)
        assert all("sequence" in item for item in items)
        
        # Verify can be used with Model (structure check only)
        # Actual API call would require authentication
        assert len(items) > 0

    def test_csv_to_api_format(self):
        """Test that CSV output is compatible with API format."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        csv_file = fixtures_dir / "sample.csv"
        
        items = load_csv(csv_file)
        
        # Verify structure is correct for API
        assert isinstance(items, list)
        assert all(isinstance(item, dict) for item in items)
        
        # Verify can be used with Model (structure check only)
        assert len(items) > 0

