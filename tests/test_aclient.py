import json
import logging

import aiofiles
import pytest
from biolm.core.http import BioLMApi, BioLMApiClient

LOGGER = logging.getLogger(__name__)

@pytest.fixture(scope='function')
def model():
    return BioLMApiClient("esmfold", raise_httpx=False, unwrap_single=False, retry_error_batches=False)

@pytest.mark.asyncio
async def test_valid_sequence(model):
    result = await model.predict(items=[{"sequence": "MDNELE"}], stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert isinstance(res, dict)
    assert "mean_plddt" in res
    assert "pdb" in res
    assert not "status_code" in res
    assert not "error" in res

@pytest.mark.asyncio
async def test_valid_sequences(model):
    result = await model.predict(items=[{"sequence": "MDNELE"},
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

@pytest.mark.asyncio
async def test_invalid_sequence_single(model):
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    result = await model.predict(items=items, stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert "error" in res
    assert "status_code" in res
    assert "Consecutive occurrences of ':' " in res['error']['items__0__sequence'][0]
    assert res["status_code"] in (400, 422)

@pytest.mark.asyncio
async def test_mixed_sequence_batch(model):
    items = [{"sequence": "MENDELSEMYEFF:FEEFMLYRRTELSYYYUPPPPPU"},
             {"sequence": "MDNELE"}]
    result = await model.predict(items=items, stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "mean_plddt" in result[0]
    assert "pdb" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

@pytest.mark.asyncio
async def test_mixed_valid_invalid_sequence_batch_continue_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = await model.predict(items=items, stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "error" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

@pytest.mark.asyncio
async def test_mixed_valid_invalid_sequence_batch_stop_on_error(model):
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"},
             {"sequence": "MDNELE"}]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(result) == 2
    assert "error" in result[0]
    assert "error" in result[1]

@pytest.mark.asyncio
async def test_mixed_valid_invalid_sequence_nonbatch_stop_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result) == 1
    assert "error" in result[0]

@pytest.mark.asyncio
async def test_stop_on_error_with_previous_success(model):
    items = [
        [{"sequence": "MDNELE"}],
        [{"sequence": "MDNELE"}],
        [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
        [{"sequence": "MDNELE"}]
    ]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert "pdb" in result[0]
    assert "pdb" in result[1]
    assert "error" in result[2]
    assert 'items__0__sequence' in result[2].get('error')

@pytest.mark.asyncio
async def test_stop_on_error_with_previous_success_reordered(model):
    items = [
        {"sequence": "MDNELE"},
        {"sequence": "MDNELE"},  # ESMFold allow two per req, so first two is one batch
        {"sequence": "MDNELE"},  # The entire second batch is bad
        {"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"},  # Second batch is bad
    ]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 4
    assert "pdb" in result[0]
    assert "pdb" in result[1]
    assert "error" in result[2]
    assert 'items__1__sequence' in result[2].get('error')
    assert 'items__1__sequence' in result[3].get('error')

@pytest.mark.asyncio
async def test_raise_httpx():
    model = BioLMApi("esmfold", raise_httpx=True)
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    with pytest.raises(Exception):
        await model.predict(items=items)

@pytest.mark.asyncio
async def test_no_double_error_key(model):
    items = [{"sequence": "BAD::BAD"}]
    result = await model.predict(items=items, stop_on_error=False)
    res = result[0] if isinstance(result, list) else result
    keys = list(res.keys())
    assert keys.count("error") == 1
    if isinstance(res["error"], dict):
        assert "error" not in res["error"]

@pytest.mark.asyncio
async def test_single_predict_to_disk(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    await model.predict(items=[{"sequence": "MDNELE"}], output='disk', file_path=str(file_path), stop_on_error=False)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "mean_plddt" in data

@pytest.mark.asyncio
async def test_single_predict_to_disk_with_error(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    await model.predict(items=[{"sequence": "MD::NELE"}], output='disk', file_path=str(file_path), stop_on_error=False)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "error" in data

@pytest.mark.asyncio
async def test_batch_predict_to_disk(tmp_path, model):
    items = [{"sequence": "MDNELE"}, {"sequence": "MENDEL"}, {"sequence": "ISOTYPE"}]
    file_path = tmp_path / "batch.jsonl"
    await model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=False)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        assert "mean_plddt" in rec

@pytest.mark.asyncio
async def test_batch_predict_to_disk_stop_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    await model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=True)
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

@pytest.mark.asyncio
async def test_batch_predict_to_disk_continue_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    await model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=False)
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

@pytest.mark.asyncio
async def test_invalid_input_items_type(model, monkeypatch):
    # Defensive: monkeypatch the _batch_call_autoschema_or_manual method to return None
    monkeypatch.setattr(model, "_batch_call_autoschema_or_manual", lambda *a, **kw: None)
    with pytest.raises(TypeError, match="Parameter 'items' must be of type list, tuple"):
        await model.predict(items={"sequence": "MDNELE"})


@pytest.mark.asyncio
async def test_disk_output_skip_existing_file_async(tmp_path):
    """Test async client: when file exists and overwrite=False, it skips API call and returns existing file contents."""
    model = BioLMApiClient("esmfold", raise_httpx=False)
    file_path = tmp_path / "existing.jsonl"
    items = [{"sequence": "MDNELE"}]
    
    # First, write some data to the file
    initial_data = {"mean_plddt": 95.5, "pdb": "ATOM 1 N MET", "test": "initial"}
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(initial_data) + '\n')
    
    # Verify file exists
    assert file_path.exists()
    
    # Call with overwrite=False - should skip API call and return existing data
    result = await model.predict(items=items, output='disk', file_path=str(file_path), overwrite=False)
    
    # Default unwrap_single=True: single line in file → single dict
    assert isinstance(result, dict)
    assert result["test"] == "initial"
    assert result["mean_plddt"] == 95.5
    assert result["pdb"] == "ATOM 1 N MET"
    
    # Verify file was not overwritten (still has initial data)
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    lines = content.splitlines()
    assert len(lines) == 1
    file_data = json.loads(lines[0])
    assert file_data["test"] == "initial"


@pytest.mark.asyncio
async def test_disk_output_skip_existing_file_async_unwrap_single_false(tmp_path):
    """When file exists, overwrite=False, and unwrap_single=False, return list of one dict."""
    model = BioLMApiClient("esmfold", raise_httpx=False, unwrap_single=False)
    file_path = tmp_path / "existing.jsonl"
    items = [{"sequence": "MDNELE"}]
    initial_data = {"mean_plddt": 95.5, "pdb": "ATOM 1 N MET", "test": "list_of_one"}
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(initial_data) + '\n')
    assert file_path.exists()

    result = await model.predict(items=items, output='disk', file_path=str(file_path), overwrite=False)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["test"] == "list_of_one"
    assert result[0]["mean_plddt"] == 95.5


@pytest.mark.asyncio
async def test_disk_output_overwrite_existing_file_async(tmp_path):
    """Test async client: when overwrite=True, it makes API call and overwrites existing file."""
    model = BioLMApiClient("esmfold", raise_httpx=False)
    file_path = tmp_path / "existing.jsonl"
    items = [{"sequence": "MDNELE"}]
    
    # First, write some initial data to the file
    initial_data = {"mean_plddt": 0.0, "pdb": "OLD DATA", "test": "should_be_overwritten"}
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(initial_data) + '\n')
    
    # Verify file exists and has initial data
    assert file_path.exists()
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    initial_lines = content.splitlines()
    assert len(initial_lines) == 1
    assert json.loads(initial_lines[0])["test"] == "should_be_overwritten"
    
    # Call with overwrite=True - should make API call and overwrite file
    result = await model.predict(items=items, output='disk', file_path=str(file_path), overwrite=True)
    
    # When output='disk' and overwrite=True, result should be None (file is written, nothing returned)
    assert result is None
    
    # Verify file was overwritten with new API response
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    lines = content.splitlines()
    assert len(lines) == 1
    file_data = json.loads(lines[0])
    # Should have new data from API (not the initial test data)
    assert "test" not in file_data  # The test key should not be in API response
    assert "mean_plddt" in file_data
    assert "pdb" in file_data
    # Verify it's actually different from initial data
    assert file_data["mean_plddt"] != 0.0 or file_data["pdb"] != "OLD DATA"


