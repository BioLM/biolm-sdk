import sys
if sys.version_info < (3, 8):
    from asynctest import CoroutineMock as AsyncMock
    from unittest.mock import patch
else:
    from unittest.mock import AsyncMock, patch

import pytest

import biolm.core.http as client_mod


@pytest.mark.asyncio
async def test_get_max_batch_size_returns_value_from_schema():
    # Arrange
    model_name = "test-model"
    action = "encode"
    expected_max_items = 42
    schema = {
        "properties": {
            "items": {
                "maxItems": expected_max_items
            }
        }
    }
    # Patch the schema method to return our schema
    api_client = client_mod.BioLMApiClient(model_name)
    with patch.object(api_client, "schema", new=AsyncMock(return_value=schema)):
        max_batch = await api_client._get_max_batch_size(model_name, action)
        assert max_batch == expected_max_items

@pytest.mark.asyncio
async def test_get_max_batch_size_returns_none_if_schema_none():
    model_name = "test-model"
    action = "encode"
    api_client = client_mod.BioLMApiClient(model_name)
    with patch.object(api_client, "schema", new=AsyncMock(return_value=None)):
        max_batch = await api_client._get_max_batch_size(model_name, action)
        assert max_batch is None

@pytest.mark.asyncio
async def test_get_max_batch_size_returns_none_if_no_maxItems():
    model_name = "test-model"
    action = "encode"
    schema = {
        "properties": {
            "items": {
                # no maxItems
            }
        }
    }
    api_client = client_mod.BioLMApiClient(model_name)
    with patch.object(api_client, "schema", new=AsyncMock(return_value=schema)):
        max_batch = await api_client._get_max_batch_size(model_name, action)
        assert max_batch is None

@pytest.mark.asyncio
async def test_get_max_batch_size_returns_none_if_schema_malformed():
    model_name = "test-model"
    action = "encode"
    # properties missing
    schema = {}
    api_client = client_mod.BioLMApiClient(model_name)
    with patch.object(api_client, "schema", new=AsyncMock(return_value=schema)):
        max_batch = await api_client._get_max_batch_size(model_name, action)
        assert max_batch is None

@pytest.mark.asyncio
async def test_schema_cache_is_used():
    # Use a unique key so we don't collide with other tests that cache ("test-model", "encode")
    model_name = "test-model-schema-cache"
    action = "encode"
    schema = {
        "properties": {
            "items": {
                "maxItems": 10
            }
        }
    }
    api_client = client_mod.BioLMApiClient(model_name)

    with patch.object(api_client._http_client, "get", new=AsyncMock()) as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        # Patch .json to be a regular function, not an async function
        mock_resp.json = lambda: schema
        mock_get.return_value = mock_resp

        # First call: should hit the HTTP client and cache the result
        result1 = await api_client.schema(model_name, action)
        assert result1 == schema
        assert mock_get.call_count == 1

        # Second call: should use the lru_cache, not call HTTP client again
        mock_get.reset_mock()
        result2 = await api_client.schema(model_name, action)
        assert result2 == schema
        mock_get.assert_not_called()


@pytest.mark.asyncio
async def test_schema_returns_none_on_http_error():
    model_name = "test-model"
    action = "encode"
    api_client = client_mod.BioLMApiClient(model_name)
    # Simulate http client raising an exception
    with patch.object(api_client._http_client, "get", new=AsyncMock(side_effect=Exception("fail"))):
        result = await api_client.schema(model_name, action)
        assert result is None

@pytest.mark.asyncio
async def test_extract_max_items_positive():
    schema = {
        "properties": {
            "items": {
                "maxItems": 123
            }
        }
    }
    assert client_mod.BioLMApiClient.extract_max_items(schema) == 123

def test_extract_max_items_none_if_not_int():
    schema = {
        "properties": {
            "items": {
                "maxItems": "not-an-int"
            }
        }
    }
    assert client_mod.BioLMApiClient.extract_max_items(schema) is None

def test_extract_max_items_none_if_missing():
    schema = {
        "properties": {
            "items": {}
        }
    }
    assert client_mod.BioLMApiClient.extract_max_items(schema) is None

def test_extract_max_items_none_if_no_properties():
    schema = {}
    assert client_mod.BioLMApiClient.extract_max_items(schema) is None

@pytest.mark.asyncio
async def test_batch_call_with_schema_uses_max_batch(monkeypatch):
    # This test ensures that _batch_call_with_schema uses the max batch size from schema
    model_name = "test-model"
    action = "encode"
    max_batch = 3
    items = [{"sequence": f"SEQ{i}"} for i in range(7)]
    api_client = client_mod.BioLMApiClient(model_name)
    # Patch _get_max_batch_size to return max_batch
    monkeypatch.setattr(api_client, "_get_max_batch_size", AsyncMock(return_value=max_batch))
    # Patch _batch_call/batch_call to just return the items it receives
    monkeypatch.setattr(api_client, "call", AsyncMock(side_effect=lambda *a, **kw: a[1]))
    # Call _batch_call_with_schema
    results = await api_client._batch_call_autoschema_or_manual(action, items)
    # Should be a flat list of all items, in order
    assert isinstance(results, list)
    assert len(results) == len(items)
    # Should have been called ceil(len(items)/max_batch) times
    assert api_client.call.call_count == 3
    # Each call should have received at most max_batch items
    for call in api_client.call.call_args_list:
        batch = call.args[1]
        assert len(batch) <= max_batch

@pytest.mark.asyncio
async def test_batch_call_with_schema_default_to_1(monkeypatch):
    # If _get_max_batch_size returns None, should default to 1
    model_name = "test-model"
    action = "encode"
    items = [{"sequence": f"SEQ{i}"} for i in range(2)]
    api_client = client_mod.BioLMApiClient(model_name)
    monkeypatch.setattr(api_client, "_get_max_batch_size", AsyncMock(return_value=None))
    monkeypatch.setattr(api_client, "call", AsyncMock(side_effect=lambda *a, **kw: a[1]))
    results = await api_client._batch_call_autoschema_or_manual(action, items)
    assert isinstance(results, list)
    assert len(results) == len(items)
    assert api_client.call.call_count == 2
    for call in api_client.call.call_args_list:
        batch = call.args[1]
        assert len(batch) == 1

