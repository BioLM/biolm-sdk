"""Tests for concurrent batch processing in BioLMApiClient."""
import asyncio
import sys
import time

import pytest

from biolm.core.http import BioLMApiClient

if sys.version_info < (3, 8):
    from asynctest import CoroutineMock as AsyncMock
else:
    from unittest.mock import AsyncMock


@pytest.fixture
def client():
    return BioLMApiClient(
        "test-model",
        raise_httpx=False,
        unwrap_single=False,
        retry_error_batches=False,
        concurrent_batches=True,
    )


@pytest.mark.asyncio
async def test_concurrent_execution_and_result_order(monkeypatch, client):
    """Multiple batches run with overlapping concurrency; results match input order."""
    items = [{"sequence": f"SEQ{i}"} for i in range(20)]
    batch_size = 4
    call_times = []
    call_lock = asyncio.Lock()

    async def fake_api_call(endpoint, payload, raw=False):
        batch = payload["items"]
        async with call_lock:
            call_times.append(time.monotonic())
        await asyncio.sleep(0.02)
        start_idx = items.index(batch[0])
        return {"results": [{"embeddings": [start_idx + j]} for j in range(len(batch))]}

    monkeypatch.setattr(client, "_get_max_batch_size", AsyncMock(return_value=batch_size))
    monkeypatch.setattr(client, "_api_call", fake_api_call)

    results = await client.encode(items=items, stop_on_error=False)
    assert isinstance(results, list)
    assert len(results) == 20
    for i, r in enumerate(results):
        assert "embeddings" in r
        assert r["embeddings"] == [i]

    assert len(call_times) == 5
    time_span = max(call_times) - min(call_times)
    assert time_span < 0.15


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_batches(monkeypatch, client):
    """With many batches, at most semaphore (16) concurrent requests run."""
    running = 0
    max_running = 0
    lock = asyncio.Lock()

    async def fake_post(endpoint, payload):
        nonlocal running, max_running
        async with lock:
            running += 1
            max_running = max(max_running, running)
        await asyncio.sleep(0.03)
        async with lock:
            running -= 1
        batch = payload.get("items", payload.get("query", []))
        n = len(batch)
        return type("Resp", (), {"status_code": 200, "headers": {"Content-Type": "application/json"}, "json": lambda: {"results": [{"x": i} for i in range(n)]}})()

    monkeypatch.setattr(client._http_client, "post", fake_post)
    monkeypatch.setattr(client, "_get_max_batch_size", AsyncMock(return_value=2))
    items = [{"sequence": f"SEQ{i}"} for i in range(40)]
    await client.encode(items=items, stop_on_error=False)
    assert max_running <= 16


@pytest.mark.asyncio
async def test_stop_on_error_uses_sequential_path(monkeypatch, client):
    """With stop_on_error=True, batches are processed sequentially (no overlap)."""
    call_starts = []
    call_ends = []

    async def fake_call(func, batch, params=None, raw=False):
        call_starts.append(time.monotonic())
        await asyncio.sleep(0.02)
        call_ends.append(time.monotonic())
        return [{"embeddings": [i]} for i in range(len(batch))]

    monkeypatch.setattr(client, "call", fake_call)
    monkeypatch.setattr(client, "_get_max_batch_size", AsyncMock(return_value=2))
    items = [{"sequence": f"SEQ{i}"} for i in range(6)]
    await client.encode(items=items, stop_on_error=True)
    assert len(call_starts) == 3
    for i in range(1, len(call_starts)):
        assert call_starts[i] >= call_ends[i - 1] - 0.001


@pytest.mark.asyncio
async def test_concurrent_batches_false_uses_sequential_path(monkeypatch):
    """With concurrent_batches=False, no concurrent execution."""
    client_seq = BioLMApiClient(
        "test-model",
        raise_httpx=False,
        unwrap_single=False,
        concurrent_batches=False,
    )
    call_starts = []
    call_ends = []

    async def fake_call(func, batch, params=None, raw=False):
        call_starts.append(time.monotonic())
        await asyncio.sleep(0.02)
        call_ends.append(time.monotonic())
        return [{"embeddings": [i]} for i in range(len(batch))]

    monkeypatch.setattr(client_seq, "call", fake_call)
    monkeypatch.setattr(client_seq, "_get_max_batch_size", AsyncMock(return_value=2))
    items = [{"sequence": f"SEQ{i}"} for i in range(6)]
    await client_seq.encode(items=items, stop_on_error=False)
    assert len(call_starts) == 3
    for i in range(1, len(call_starts)):
        assert call_starts[i] >= call_ends[i - 1] - 0.001


@pytest.mark.asyncio
async def test_result_order_when_batches_complete_out_of_order(monkeypatch, client):
    """Batches complete in non-order (e.g. 2,0,1,3); final results match item order."""
    order = []

    async def fake_api_call(endpoint, payload, raw=False):
        batch = payload["items"]
        idx = batch[0]["sequence"]
        assert idx.startswith("SEQ")
        idx = int(idx[3:])
        order.append(idx)
        await asyncio.sleep(0.01 * (3 - (idx % 4)))
        return {"results": [{"item": idx * 10 + j} for j in range(len(batch))]}

    monkeypatch.setattr(client, "_get_max_batch_size", AsyncMock(return_value=2))
    monkeypatch.setattr(client, "_api_call", fake_api_call)
    items = [{"sequence": f"SEQ{i}"} for i in range(8)]
    results = await client.encode(items=items, stop_on_error=False)
    assert len(results) == 8
    for i in range(8):
        batch_start = (i // 2) * 2
        pos_in_batch = i % 2
        assert results[i]["item"] == batch_start * 10 + pos_in_batch


@pytest.mark.asyncio
async def test_disk_output_order_with_concurrent_batches(monkeypatch, client, tmp_path):
    """Disk JSONL lines match item order when batches complete out of order."""
    import json

    async def fake_api_call(endpoint, payload, raw=False):
        batch = payload["items"]
        idx = int(batch[0]["sequence"][3:])
        await asyncio.sleep(0.01 * (3 - (idx % 4)))
        return {"results": [{"idx": idx * 10 + j} for j in range(len(batch))]}

    monkeypatch.setattr(client, "_get_max_batch_size", AsyncMock(return_value=2))
    monkeypatch.setattr(client, "_api_call", fake_api_call)
    items = [{"sequence": f"SEQ{i}"} for i in range(8)]
    path = tmp_path / "out.jsonl"
    await client._batch_call_autoschema_or_manual(
        "encode", items, output="disk", file_path=str(path)
    )
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 8
    for i, line in enumerate(lines):
        rec = json.loads(line)
        batch_start = (i // 2) * 2
        pos_in_batch = i % 2
        assert rec["idx"] == batch_start * 10 + pos_in_batch


@pytest.mark.asyncio
async def test_list_of_lists_concurrent_and_order(monkeypatch, client):
    """List-of-lists with stop_on_error=False uses concurrency; result order preserved."""
    call_times = []
    lock = asyncio.Lock()

    async def fake_call(func, batch, params=None, raw=False):
        async with lock:
            call_times.append(time.monotonic())
        await asyncio.sleep(0.02)
        return [{"ok": i} for i in range(len(batch))]

    monkeypatch.setattr(client, "call", fake_call)
    items = [
        [{"sequence": "A"}],
        [{"sequence": "B"}],
        [{"sequence": "C"}],
    ]
    results = await client._batch_call_autoschema_or_manual("encode", items, stop_on_error=False)
    assert len(results) == 3
    assert results[0]["ok"] == 0
    assert results[1]["ok"] == 0
    assert results[2]["ok"] == 0
    assert len(call_times) == 3
    time_span = max(call_times) - min(call_times)
    assert time_span < 0.1


@pytest.mark.asyncio
async def test_progress_callback_invoked_per_batch(monkeypatch, client):
    """progress_callback is called after each batch with (completed, total)."""
    calls = []

    def progress_callback(completed: int, total: int):
        calls.append((completed, total))

    async def fake_api_call(endpoint, payload, raw=False):
        batch = payload["items"]
        return {"results": [{"x": i} for i in range(len(batch))]}

    monkeypatch.setattr(client, "_api_call", fake_api_call)
    monkeypatch.setattr(client, "_get_max_batch_size", AsyncMock(return_value=2))
    items = [{"sequence": f"SEQ{i}"} for i in range(5)]
    results = await client.predict(items=items, progress_callback=progress_callback)
    assert len(results) == 5
    assert len(calls) >= 1
    assert calls[-1] == (5, 5)
