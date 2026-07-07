import pytest
import sys

from biolm.core.http import BioLMApiClient

if sys.version_info < (3, 8):
    from asynctest import CoroutineMock as AsyncMock
else:
    from unittest.mock import AsyncMock


@pytest.fixture
def model():
    return BioLMApiClient("esm2-8m", raise_httpx=False, unwrap_single=False, retry_error_batches=False)

@pytest.mark.asyncio
async def test_batch_call_with_schema_error_unpacking(monkeypatch, model):
    items = [{"sequence": f"SEQ{i}"} for i in range(100)]
    batch_size = 5

    # Patch get_max_batch_items to force batching at 5
    monkeypatch.setattr(model, "_get_max_batch_size", AsyncMock(return_value=batch_size))

    # Patch _api_call to simulate error in batch 3 (items 10-14)
    async def fake_api_call(endpoint, payload, raw=False):
        # payload['items'] is the batch
        batch = payload['items']
        start_idx = items.index(batch[0])
        if start_idx == 10:
            # Simulate a single error dict for the whole batch
            return {"error": "HTTP 500", "status_code": 500}
        else:
            # Simulate normal embedding dicts
            return {"results": [{"embeddings": [start_idx + j]} for j in range(len(batch))]}

    monkeypatch.setattr(model, "_api_call", fake_api_call)

    # Now call encode, which will use the patched _api_call via _batch_call_with_schema
    results = await model.encode(items=items, stop_on_error=False)
    assert isinstance(results, list)
    assert len(results) == 100
    # Check that items 10-14 are all error dicts, others are embedding dicts
    for i, r in enumerate(results):
        if 10 <= i < 15:
            assert "error" in r and r["status_code"] == 500
        else:
            assert "embeddings" in r
