import json
import logging

import pytest

from biolm import biolm
from biolm.client import BioLM
from biolm.core.http import BioLMApi
from biolm.models import Model

LOGGER = logging.getLogger(__name__)

@pytest.fixture(scope='module')
def model():
    return BioLMApi("esmfold", raise_httpx=False, unwrap_single=False, retry_error_batches=False)

def test_valid_sequence(model):
    result = model.predict(items=[{"sequence": "MDNELE"}], stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert isinstance(res, dict)
    assert "mean_plddt" in res
    assert "pdb" in res
    assert not "status_code" in res
    assert not "error" in res


def test_valid_sequence_same_via_biolm_biolm_fn_model():
    """Same single predict outcome via BioLMApi, BioLM(), biolm(), and Model() (default unwrap_single=True)."""
    items = [{"sequence": "MDNELE"}]
    kw = dict(items=items, stop_on_error=False)
    api = BioLMApi("esmfold", raise_httpx=False, retry_error_batches=False)
    r_api = api.predict(**kw)
    r_biolm = BioLM(entity="esmfold", action="predict", items=items, raise_httpx=False, retry_error_batches=False)
    r_fn = biolm(entity="esmfold", action="predict", items=items, raise_httpx=False, retry_error_batches=False)
    r_model = Model("esmfold", raise_httpx=False, retry_error_batches=False).predict(items=items, stop_on_error=False, progress=False)
    for name, r in [("BioLMApi", r_api), ("BioLM", r_biolm), ("biolm()", r_fn), ("Model", r_model)]:
        assert isinstance(r, dict), f"{name}: expected dict (unwrap_single=True)"
        assert "mean_plddt" in r and "pdb" in r, f"{name}: expected keys"

def test_valid_sequences(model):
    result = model.predict(items=[{"sequence": "MDNELE"},
                                         {"sequence": "MUUUUDANLEPY"}], stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 2
    for res in result:
        assert isinstance(res, dict)
        assert "mean_plddt" in res
        assert "pdb" in res
        assert not "status_code" in res
        assert not "error" in res
        assert isinstance(result, dict) or isinstance(result, list)

def test_invalid_sequence_single(model):
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    result = model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert "error" in res
    assert "status_code" in res
    # Should exist a single [0] error for the 0th 'sequence' item
    assert "Consecutive occurrences of ':' " in res['error']['items__0__sequence'][0]
    # Should really be HTTP 422, but need to change API-side
    assert res["status_code"] in (400, 422)

def test_mixed_sequence_batch_lol_items_first_invalid(model):
    items = [[{"sequence": "MENDELSEMYEFF::FEEFMLYRRTELSYYYUPPPPPU"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "error" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

def test_mixed_sequence_batch_lol_items_first_invalid_multiple_in_batch(model):
    items = [[{"sequence": "MENDELSEMYEFF::FEEFMLYRRTELSYYYUPPPPPU"},
              {"sequence": "invalid123"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 3
    assert "error" in result[0]
    assert "error" in result[1]
    assert "mean_plddt" in result[2]
    assert "pdb" in result[2]


def test_mixed_valid_invalid_sequence_batch_continue_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items, stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "error" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

def test_mixed_valid_invalid_sequence_batch_stop_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 1
    # Should stop on first error, so only one result
    assert len(result) == 1
    assert "error" in result[0]

def test_stop_on_error_with_previous_success(model):
    items = [
        [{"sequence": "MDNELE"}],
        [{"sequence": "MDNELE"}],
        [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
        [{"sequence": "MDNELE"}],
    ]
    result = model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert "pdb" in result[0]
    assert "pdb" in result[1]
    assert "error" in result[2]

def test_raise_httpx():
    model = BioLMApi("esmfold", raise_httpx=True)
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    with pytest.raises(Exception):
        model.predict(items=items)

def test_no_double_error_key(model):
    # Single bad item should produce one error key only
    items = [{"sequence": "BAD::BAD"}]
    result = model.predict(items=items)
    # Normalize to single result dict
    res = result[0] if isinstance(result, list) else result
    # Must have exactly one "error" key at top level
    keys = list(res.keys())
    assert keys.count("error") == 1
    # If the error value is a dict, it must not itself contain "error"
    if isinstance(res["error"], dict):
        assert "error" not in res["error"]

def test_single_predict_to_disk(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    model.predict(items=[{"sequence": "MDNELE"}], output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "mean_plddt" in data

def test_single_predict_to_disk_with_error(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    model.predict(items=[{"sequence": "MD::NELE"}], output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "error" in data

def test_batch_predict_to_disk(tmp_path, model):
    items = [{"sequence": "MDNELE"}, {"sequence": "MENDEL"}, {"sequence": "ISOTYPE"}]
    file_path = tmp_path / "batch.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        assert "mean_plddt" in rec

def test_batch_predict_to_disk_stop_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=True)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 2
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        if i == 0:
            assert "mean_plddt" in rec
        else:
            assert "error" in rec

def test_batch_predict_to_disk_continue_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=False)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        if i == 1:
            assert "error" in rec
        else:
            assert "mean_plddt" in rec

def test_batch_predict_to_disk_continue_on_error_multiple_per_batch(tmp_path, model):
    items = [[{"sequence": "MDNELE"},
              {"sequence": "MDNELE"}],
             [{"sequence": "DN::A"},
              {"sequence": "MDNELE"}],
             [{"sequence": "ISOTYPE"},
              {"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch_multiple.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=False)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 6
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        if i in (2, 3):
            assert "error" in rec
        else:
            assert "mean_plddt" in rec

def test_invalid_input_items_type(model, monkeypatch):
    # Defensive: monkeypatch the _batch_call_autoschema_or_manual method to return None
    monkeypatch.setattr(model, "_batch_call_autoschema_or_manual", lambda *a, **kw: None)
    # Should raise TypeError because items is not a list/tuple
    with pytest.raises(TypeError, match="Parameter 'items' must be of type list, tuple"):
        model.predict(items={"sequence": "MDNELE"})


def test_disk_output_skip_existing_file(tmp_path, model):
    """Test that when file exists and overwrite=False, it skips API call and returns existing file contents."""
    file_path = tmp_path / "existing.jsonl"
    items = [{"sequence": "MDNELE"}]
    
    # First, write some data to the file
    initial_data = {"mean_plddt": 95.5, "pdb": "ATOM 1 N MET", "test": "initial"}
    with open(file_path, 'w') as f:
        f.write(json.dumps(initial_data) + '\n')
    
    # Verify file exists
    assert file_path.exists()
    
    # Call with overwrite=False - should skip API call and return existing data
    result = model.predict(items=items, output='disk', file_path=str(file_path), overwrite=False)
    
    # Should return the existing file contents
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["test"] == "initial"
    assert result[0]["mean_plddt"] == 95.5
    
    # Verify file was not overwritten (still has initial data)
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    file_data = json.loads(lines[0])
    assert file_data["test"] == "initial"


def test_disk_output_overwrite_existing_file(tmp_path, model):
    """Test that when overwrite=True, it makes API call and overwrites existing file."""
    file_path = tmp_path / "existing.jsonl"
    items = [{"sequence": "MDNELE"}]
    
    # First, write some initial data to the file
    initial_data = {"mean_plddt": 0.0, "pdb": "OLD DATA", "test": "should_be_overwritten"}
    with open(file_path, 'w') as f:
        f.write(json.dumps(initial_data) + '\n')
    
    # Verify file exists and has initial data
    assert file_path.exists()
    initial_lines = file_path.read_text().splitlines()
    assert len(initial_lines) == 1
    assert json.loads(initial_lines[0])["test"] == "should_be_overwritten"
    
    # Call with overwrite=True - should make API call and overwrite file
    result = model.predict(items=items, output='disk', file_path=str(file_path), overwrite=True)
    
    # When output='disk' and overwrite=True, result should be None (file is written, nothing returned)
    assert result is None
    
    # Verify file was overwritten with new API response
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    file_data = json.loads(lines[0])
    # Should have new data from API (not the initial test data)
    assert "test" not in file_data  # The test key should not be in API response
    assert "mean_plddt" in file_data
    assert "pdb" in file_data
    # Verify it's actually different from initial data
    assert file_data["mean_plddt"] != 0.0 or file_data["pdb"] != "OLD DATA"


def test_disk_output_skip_empty_file(tmp_path, model):
    """Test that when file exists but is empty, it returns empty list and skips API call."""
    file_path = tmp_path / "empty.jsonl"
    items = [{"sequence": "MDNELE"}]
    
    # Create empty file
    file_path.touch()
    assert file_path.exists()
    assert file_path.stat().st_size == 0
    
    # Call with overwrite=False - should skip API call and return empty list
    result = model.predict(items=items, output='disk', file_path=str(file_path), overwrite=False)
    
    # Should return empty list (no valid JSON lines)
    assert isinstance(result, list)
    assert len(result) == 0
    
    # Verify file was not modified (still empty)
    assert file_path.stat().st_size == 0


def test_disk_output_skip_file_with_invalid_json(tmp_path, model):
    """Test that when file exists with invalid JSON, it skips invalid lines and returns valid ones."""
    file_path = tmp_path / "mixed.jsonl"
    items = [{"sequence": "MDNELE"}]
    
    # Write file with mix of valid and invalid JSON
    with open(file_path, 'w') as f:
        f.write('{"valid": "data1"}\n')
        f.write('invalid json line\n')
        f.write('{"valid": "data2"}\n')
        f.write('{"incomplete": \n')  # Invalid JSON
        f.write('{"valid": "data3"}\n')
    
    # Call with overwrite=False - should skip API call and return valid JSON lines
    result = model.predict(items=items, output='disk', file_path=str(file_path), overwrite=False)
    
    # Should return only valid JSON lines
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["valid"] == "data1"
    assert result[1]["valid"] == "data2"
    assert result[2]["valid"] == "data3"
    
    # Verify file was not modified
    lines = file_path.read_text().splitlines()
    assert len(lines) == 5  # Original 5 lines still there


def test_disk_output_skip_biolm_wrapper(tmp_path):
    """Test that BioLM high-level wrapper also respects overwrite parameter."""
    from biolm.client import BioLM
    file_path = tmp_path / "biolm_wrapper.jsonl"
    
    # First, write some data to the file
    initial_data = {"mean_plddt": 95.5, "pdb": "ATOM 1 N MET", "test": "initial"}
    with open(file_path, 'w') as f:
        f.write(json.dumps(initial_data) + '\n')
    
    # Call with overwrite=False - should skip API call and return existing data
    result = BioLM(
        entity="esmfold",
        action="predict",
        type="sequence",
        items="MDNELE",
        output='disk',
        file_path=str(file_path),
        overwrite=False
    )
    
    # Should return the existing file contents
    assert isinstance(result, dict)  # BioLM unwraps single results by default
    assert result["test"] == "initial"
    assert result["mean_plddt"] == 95.5


def test_disk_output_skip_list_of_lists(tmp_path, model):
    """Test that overwrite=False works correctly with list of lists (pre-batched items)."""
    file_path = tmp_path / "lol.jsonl"
    
    # First, write some data to the file (simulating results from list of lists)
    with open(file_path, 'w') as f:
        f.write(json.dumps({"batch1": "result1"}) + '\n')
        f.write(json.dumps({"batch1": "result2"}) + '\n')
        f.write(json.dumps({"batch2": "result3"}) + '\n')
    
    # Call with overwrite=False and list of lists
    items = [[{"sequence": "MDNELE"}], [{"sequence": "MENDEL"}], [{"sequence": "ISOTYPE"}]]
    result = model.predict(items=items, output='disk', file_path=str(file_path), overwrite=False)
    
    # Should return the existing file contents
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["batch1"] == "result1"
    assert result[1]["batch1"] == "result2"
    assert result[2]["batch2"] == "result3"
