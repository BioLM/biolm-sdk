import json

import pytest

from biolm.core.http import BioLMApiClient


@pytest.mark.asyncio
async def test_retry_error_batches_live_partial_batch():
    # 8 items, only item 3 is invalid
    items = [{"sequence": "MSILVTRPSPAGEEL"} for _ in range(8)]
    items[3]["sequence"] = "BAD::BAD"  # Invalid sequence triggers error
    client = BioLMApiClient("esm2-8m", retry_error_batches=True, raise_httpx=False)
    results = await client._batch_call_autoschema_or_manual("encode", items)
    assert len(results) == 8
    for i, r in enumerate(results):
        if i == 3:
            assert "error" in r
            assert "status_code" in r
        else:
            assert "embeddings" in r

@pytest.mark.asyncio
async def test_retry_error_batches_live_stop_on_error():
    # 8 items, item 2 is invalid, should stop after first error batch
    items = [{"sequence": "MSILVTRPSPAGEEL"} for _ in range(8)]
    items[2]["sequence"] = "BAD::BAD"
    client = BioLMApiClient("esm2-8m", retry_error_batches=True, raise_httpx=False)
    results = await client._batch_call_autoschema_or_manual("encode", items, stop_on_error=True)
    # Should return only up to the batch containing the error
    assert isinstance(results, list)
    # The batch size for esm2-8m is 8, so all results should be returned
    # But only one should be an error
    assert len(results) == 8
    assert any("error" in r for r in results)
    assert any("embeddings" in r for r in results)

@pytest.mark.asyncio
async def test_retry_error_batches_live_disk(tmp_path):
    # 8 items, item 5 is invalid
    items = [{"sequence": "MSILVTRPSPAGEEL"} for _ in range(8)]
    items[5]["sequence"] = "BAD::BAD"
    client = BioLMApiClient("esm2-8m", retry_error_batches=True, raise_httpx=False)
    file_path = tmp_path / "out.jsonl"
    await client._batch_call_autoschema_or_manual("encode", items, output="disk", file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 8
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if i == 5:
            assert "error" in rec
        else:
            assert "embeddings" in rec

@pytest.mark.asyncio
async def test_retry_error_batches_live_batch_of_batches():
    # List of lists: 2 batches, one with an error
    items = [
        [{"sequence": "MSILVTRPSPAGEEL"}, {"sequence": "MSILVTRPSPAGEEL"}],
        [{"sequence": "BAD::BAD"}, {"sequence": "MSILVTRPSPAGEEL"}],
    ]
    client = BioLMApiClient("esm2-8m", retry_error_batches=True, raise_httpx=False)
    results = await client._batch_call_autoschema_or_manual("encode", items)
    assert isinstance(results, list)
    assert len(results) == 4
    assert "error" in results[2]
    assert "embeddings" in results[3]
    assert "embeddings" in results[0]
    assert "embeddings" in results[1]

@pytest.mark.asyncio
async def test_retry_error_batches_live_all_good():
    # All valid items
    items = [{"sequence": "MSILVTRPSPAGEEL"} for _ in range(8)]
    client = BioLMApiClient("esm2-8m", retry_error_batches=True, raise_httpx=False)
    results = await client._batch_call_autoschema_or_manual("encode", items)
    assert len(results) == 8
    for r in results:
        assert "embeddings" in r
