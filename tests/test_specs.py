"""
Unit tests for StructureSpec and MatrixExtractionSpec (PR #101 regression).

Covers:
- D1: StructureSpec.detect_format(), index handling, to_spec() round-trip
- D1: MatrixExtractionSpec matrix mode, list mode, _dot_access, to_spec()
"""
import dataclasses
import json
import pytest

pytest.importorskip("duckdb")

from biolm.pipeline.data import (
    MatrixExtractionSpec,
    StructureSpec,
    _dot_access,
)


# ---------------------------------------------------------------------------
# StructureSpec
# ---------------------------------------------------------------------------


class TestStructureSpec:
    """Unit tests for StructureSpec."""

    # --- detect_format ---

    def test_detect_format_pdb_key(self):
        """Key 'pdb' → format 'pdb'."""
        spec = StructureSpec(key="pdb")
        assert spec.detect_format() == "pdb"

    def test_detect_format_pdbs_key(self):
        """Key 'pdbs' (list-style AF2) → format 'pdb' (key contains 'pdb')."""
        spec = StructureSpec(key="pdbs")
        assert spec.detect_format() == "pdb"

    def test_detect_format_cif_key(self):
        """Key 'structure_cif' → format 'cif' (no 'pdb' substring)."""
        spec = StructureSpec(key="structure_cif")
        assert spec.detect_format() == "cif"

    def test_detect_format_unknown_key_falls_back_to_cif(self):
        """Key with no 'pdb' substring → 'cif' (default fallback)."""
        spec = StructureSpec(key="coordinates")
        assert spec.detect_format() == "cif"

    def test_detect_format_explicit_override_pdb_key_cif_format(self):
        """Explicit format='cif' wins even when key contains 'pdb'."""
        spec = StructureSpec(key="pdb", format="cif")
        assert spec.detect_format() == "cif"

    def test_detect_format_explicit_override_cif_key_pdb_format(self):
        """Explicit format='pdb' wins even when key is 'structure_cif'."""
        spec = StructureSpec(key="structure_cif", format="pdb")
        assert spec.detect_format() == "pdb"

    # --- index / list-valued key behaviour ---

    def test_store_structure_index_in_bounds(self, tmp_path):
        """_store_structure with list-valued key, index=0 extracts first element."""
        import unittest.mock as mock
        from biolm.pipeline.data import PredictionStage

        spec = StructureSpec(key="pdbs", index=0)
        stage = mock.MagicMock()
        stage._structure_output = spec
        stage.model_name = "esmfold"

        datastore = mock.MagicMock()
        result = {"pdbs": ["ATOM   1 ...", "ATOM   2 ..."]}

        # Call the real _store_structure method with the mock stage as self
        PredictionStage._store_structure(stage, datastore, seq_id=1, result=result)

        datastore.add_structure.assert_called_once()
        call_kwargs = datastore.add_structure.call_args
        # First positional arg is seq_id=1, second model_name, third is structure_str
        args, kwargs = call_kwargs
        structure_str = kwargs.get("structure_str") or args[2]
        assert structure_str == "ATOM   1 ..."

    def test_store_structure_index_out_of_bounds_silently_nops(self, tmp_path):
        """_store_structure with out-of-bounds index is a silent no-op (returns early)."""
        import unittest.mock as mock
        from biolm.pipeline.data import PredictionStage

        spec = StructureSpec(key="pdbs", index=99)
        stage = mock.MagicMock()
        stage._structure_output = spec
        stage.model_name = "esmfold"

        datastore = mock.MagicMock()
        result = {"pdbs": ["ATOM   1 ..."]}  # only index 0 exists

        PredictionStage._store_structure(stage, datastore, seq_id=1, result=result)

        # No structure should have been stored
        datastore.add_structure.assert_not_called()

    def test_store_structure_missing_key_is_nop(self):
        """_store_structure when key is absent in result is a no-op."""
        import unittest.mock as mock
        from biolm.pipeline.data import PredictionStage

        spec = StructureSpec(key="pdb", index=0)
        stage = mock.MagicMock()
        stage._structure_output = spec
        stage.model_name = "esmfold"

        datastore = mock.MagicMock()
        result = {}  # no 'pdb' key

        PredictionStage._store_structure(stage, datastore, seq_id=1, result=result)
        datastore.add_structure.assert_not_called()

    # --- to_spec round-trip ---

    def test_to_spec_roundtrip_minimal(self):
        """StructureSpec(key='pdb') → to_spec() → dict with key 'pdb', JSON-serializable."""
        spec = StructureSpec(key="pdb")
        d = dataclasses.asdict(spec)
        # JSON-serializable
        json_str = json.dumps(d)
        assert json.loads(json_str)["key"] == "pdb"

    def test_to_spec_roundtrip_full(self):
        """StructureSpec with all fields round-trips through dataclasses.asdict."""
        spec = StructureSpec(key="structure_cif", format="cif", plddt_key="plddt_mean", index=2)
        d = dataclasses.asdict(spec)
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["key"] == "structure_cif"
        assert restored["format"] == "cif"
        assert restored["plddt_key"] == "plddt_mean"
        assert restored["index"] == 2

    def test_to_spec_dict_has_all_fields(self):
        """dataclasses.asdict includes all declared fields."""
        spec = StructureSpec(key="pdb")
        d = dataclasses.asdict(spec)
        assert set(d.keys()) == {"key", "format", "plddt_key", "index"}


# ---------------------------------------------------------------------------
# MatrixExtractionSpec
# ---------------------------------------------------------------------------


class TestMatrixExtractionSpec:
    """Unit tests for MatrixExtractionSpec and related helpers."""

    # --- _dot_access ---

    def test_dot_access_simple_path(self):
        """_dot_access({'a': {'b': 1}}, 'a.b') returns 1."""
        assert _dot_access({"a": {"b": 1}}, "a.b") == 1

    def test_dot_access_single_key(self):
        """_dot_access({'x': 42}, 'x') returns 42."""
        assert _dot_access({"x": 42}, "x") == 42

    def test_dot_access_missing_top_level_raises_key_error(self):
        """_dot_access with missing top-level key raises KeyError."""
        with pytest.raises(KeyError):
            _dot_access({}, "missing")

    def test_dot_access_missing_nested_key_raises_key_error(self):
        """_dot_access with missing nested key raises KeyError."""
        with pytest.raises(KeyError):
            _dot_access({"a": {}}, "a.b")

    def test_dot_access_non_dict_intermediate_raises_key_error(self):
        """_dot_access on a non-dict intermediate node raises KeyError."""
        with pytest.raises(KeyError):
            _dot_access({"a": 42}, "a.b")

    # --- Matrix mode (SPURS) ---

    def _make_matrix_stage(self, spec: MatrixExtractionSpec):
        """Return a mock PredictionStage-like object wired to use _extract_matrix."""
        import unittest.mock as mock
        from biolm.pipeline.data import PredictionStage

        stage = mock.MagicMock()
        stage._matrix_extraction = spec
        # Bind the real _extract_matrix to this mock
        stage._extract_matrix = lambda result: PredictionStage._extract_matrix(stage, result)
        return stage

    def test_extract_matrix_position_numbering_no_digit_in_label(self):
        """
        Regression for commit 70d6d9c: row label 'M' (no digit) → pos label 'M1' (1-based),
        so prediction_type = 'ddg_M1A' not 'ddg_MA'.
        """
        spec = MatrixExtractionSpec(
            prefix="ddg",
            values_key="ddG_matrix.values",
            row_labels_key="ddG_matrix.residue_axis",
            col_labels_key="ddG_matrix.amino_acid_axis",
        )
        stage = self._make_matrix_stage(spec)
        result = {
            "ddG_matrix": {
                "values": [[-0.5]],          # 1 row × 1 col
                "residue_axis": ["M"],        # NO digit in label
                "amino_acid_axis": ["A"],
            }
        }
        pairs = stage._extract_matrix(result)
        assert len(pairs) == 1
        pred_type, value = pairs[0]
        # Row label "M" + 1-based index 1 → "M1", col label "A" → "M1A"
        assert pred_type == "ddg_M1A", (
            f"Expected 'ddg_M1A' but got '{pred_type}'. "
            "Regression: label without digit must get 1-based position appended."
        )
        assert value == pytest.approx(-0.5)

    def test_extract_matrix_position_numbering_with_digit_in_label(self):
        """
        Row label 'M1' (already has a digit) → pos label stays 'M1' (no double-numbering).
        Prediction type must be 'ddg_M1A', not 'ddg_M11A'.
        """
        spec = MatrixExtractionSpec(
            prefix="ddg",
            values_key="ddG_matrix.values",
            row_labels_key="ddG_matrix.residue_axis",
            col_labels_key="ddG_matrix.amino_acid_axis",
        )
        stage = self._make_matrix_stage(spec)
        result = {
            "ddG_matrix": {
                "values": [[-1.2]],
                "residue_axis": ["M1"],   # ALREADY has digit
                "amino_acid_axis": ["A"],
            }
        }
        pairs = stage._extract_matrix(result)
        assert len(pairs) == 1
        pred_type, value = pairs[0]
        assert pred_type == "ddg_M1A", (
            f"Expected 'ddg_M1A' but got '{pred_type}'. "
            "Label already containing a digit must NOT be re-numbered."
        )
        assert value == pytest.approx(-1.2)

    def test_extract_matrix_multi_row_col(self):
        """3×2 matrix produces 6 prediction pairs with correct labels."""
        spec = MatrixExtractionSpec(
            prefix="ddg",
            values_key="ddG_matrix.values",
            row_labels_key="ddG_matrix.residue_axis",
            col_labels_key="ddG_matrix.amino_acid_axis",
        )
        stage = self._make_matrix_stage(spec)
        result = {
            "ddG_matrix": {
                "values": [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ],
                "residue_axis": ["A", "B", "C"],   # no digits → get 1/2/3 appended
                "amino_acid_axis": ["X", "Y"],
            }
        }
        pairs = stage._extract_matrix(result)
        assert len(pairs) == 6
        types = [p[0] for p in pairs]
        assert "ddg_A1X" in types
        assert "ddg_B2X" in types
        assert "ddg_C3Y" in types

    def test_extract_matrix_bad_dotpath_returns_empty(self):
        """Missing matrix key logs warning and returns []."""
        spec = MatrixExtractionSpec(
            prefix="ddg",
            values_key="ddG_matrix.values",
            row_labels_key="ddG_matrix.residue_axis",
            col_labels_key="ddG_matrix.amino_acid_axis",
        )
        stage = self._make_matrix_stage(spec)
        result = {}  # no ddG_matrix key at all
        pairs = stage._extract_matrix(result)
        assert pairs == []

    # --- List mode (ThermoMPNN) ---

    def test_extract_matrix_list_mode(self):
        """List mode: [{mutation: 'M1A', ddg: -0.5}] → [('ddg_M1A', -0.5)]."""
        spec = MatrixExtractionSpec(
            prefix="ddg",
            mutation_key="mutation",
            value_key="ddg",
        )
        stage = self._make_matrix_stage(spec)
        result = [
            {"mutation": "M1A", "ddg": -0.5},
            {"mutation": "L2V", "ddg": 0.3},
        ]
        pairs = stage._extract_matrix(result)
        assert len(pairs) == 2
        assert ("ddg_M1A", -0.5) in pairs
        assert ("ddg_L2V", 0.3) in pairs

    def test_extract_matrix_list_mode_skips_incomplete_items(self):
        """List mode skips items where mutation_key or value_key is missing."""
        spec = MatrixExtractionSpec(
            prefix="ddg",
            mutation_key="mutation",
            value_key="ddg",
        )
        stage = self._make_matrix_stage(spec)
        result = [
            {"mutation": "M1A"},        # missing value_key
            {"ddg": -0.5},              # missing mutation_key
            {"mutation": "L2V", "ddg": 0.7},  # complete
        ]
        pairs = stage._extract_matrix(result)
        assert len(pairs) == 1
        assert pairs[0] == ("ddg_L2V", 0.7)

    # --- to_spec round-trip ---

    def test_to_spec_roundtrip_matrix_mode(self):
        """MatrixExtractionSpec (matrix mode) round-trips via dataclasses.asdict."""
        spec = MatrixExtractionSpec(
            prefix="score",
            values_key="matrix.vals",
            row_labels_key="matrix.rows",
            col_labels_key="matrix.cols",
        )
        d = dataclasses.asdict(spec)
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["prefix"] == "score"
        assert restored["values_key"] == "matrix.vals"
        assert restored["mutation_key"] is None
        assert restored["value_key"] is None

    def test_to_spec_roundtrip_list_mode(self):
        """MatrixExtractionSpec (list mode) round-trips via dataclasses.asdict."""
        spec = MatrixExtractionSpec(
            prefix="ddg",
            mutation_key="mutation",
            value_key="ddg",
        )
        d = dataclasses.asdict(spec)
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["mutation_key"] == "mutation"
        assert restored["value_key"] == "ddg"

    def test_to_spec_dict_has_all_fields(self):
        """dataclasses.asdict includes all declared fields for MatrixExtractionSpec."""
        spec = MatrixExtractionSpec()
        d = dataclasses.asdict(spec)
        expected_fields = {
            "prefix", "values_key", "row_labels_key", "col_labels_key",
            "mutation_key", "value_key",
        }
        assert set(d.keys()) == expected_fields
