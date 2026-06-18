import asyncio
import random

import pytest

from biolm import biolm
from biolm.client import BioLM
from biolm.core.http import BioLMApiClient, BioLMApi
from biolm.core.utils import batch_iterable
from biolm.models import Model


def random_sequence(length=5):
    return ''.join(random.choices('ACDEFGHIKLMNPQRSTVWYB', k=length))

@pytest.fixture(scope='function')
def model():
    # Pin to previous default (no retry) so tests that assert batch-level errors keep passing.
    return BioLMApiClient("esm2-8m", raise_httpx=False, unwrap_single=False, retry_error_batches=False)

@pytest.mark.asyncio
async def test_large_batch_encode_consistency(model):
    items = [{"sequence": random_sequence()} for _ in range(3)]
    # 1. New async method (schema-batched, concurrent)
    results_async = await model.encode(items=items, stop_on_error=False)
    # 2. Old method: _batch_call/batch_call (single-item batching, sequential)
    results_internal = await model._batch_call_autoschema_or_manual("encode", items)
    # 3. Universal client (sync, run in thread)
    loop = asyncio.get_event_loop()
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, raise_httpx=False, retry_error_batches=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    # All should be lists of the same length
    assert isinstance(results_async, list)
    assert isinstance(results_internal, list)
    assert isinstance(results_biolm, list)
    assert len(results_async) == len(items)
    assert len(results_internal) == len(items)
    assert len(results_biolm) == len(items)
    # All results should be dicts with "embeddings"
    for r1, r2, r3 in zip(results_async, results_internal, results_biolm):
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        assert isinstance(r3, dict)
        assert "embeddings" in r1
        assert "embeddings" in r2
        assert "embeddings" in r3
        # Optionally, check values are equal (if deterministic)
        # assert r1 == r2 == r3

@pytest.mark.asyncio
async def test_large_batch_encode_with_errors(model):
    items = [{"sequence": random_sequence()} for _ in range(20)] + \
            [{"sequence": "BAD::BAD"} for _ in range(5)] + \
            [{"sequence": random_sequence()} for _ in range(15)]
    results_async = await model.encode(items=items, stop_on_error=False)
    results_internal = await model._batch_call_autoschema_or_manual("encode", items, stop_on_error=False)
    loop = asyncio.get_event_loop()
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=False, raise_httpx=False, retry_error_batches=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    assert len(results_async) == len(items)
    assert len(results_internal) == len(items)
    assert len(results_biolm) == len(items)
    error_indices = set(range(16, 24)) | set(range(24, 32))  # 16–31 inclusive
    for i, (r1, r2, r3) in enumerate(zip(results_async, results_internal, results_biolm)):
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        assert isinstance(r3, dict)
        if i in error_indices:
            assert "error" in r1 and "error" in r2 and "error" in r3
        else:
            assert "embeddings" in r1
            assert "embeddings" in r2
            assert "embeddings" in r3

@pytest.mark.asyncio
async def test_large_batch_encode_stop_on_error(model):
    items = [{"sequence": random_sequence()} for _ in range(10)] + \
            [{"sequence": "BAD::BAD"}] + \
            [{"sequence": random_sequence()} for _ in range(10)]
    assert len(items) == 21
    max_batch = await model._get_max_batch_size(model.model_name, "encode") or 1
    assert max_batch == 8
    # Batch size for this model is 8. If stopping on errors, we should have 8 good results
    results_async = await model.encode(items=items, stop_on_error=True)
    # Same here
    results_internal = await model._batch_call_autoschema_or_manual("encode", items, stop_on_error=True)
    loop = asyncio.get_event_loop()
    # This *does not* stop on errors
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=False, raise_httpx=False, retry_error_batches=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    # Should get the first batch and the second one should be errors
    assert len(results_async) == 16
    assert len(results_internal) == 16
    assert len(results_biolm) == len(items)

    assert "error" in results_async[-1]
    assert "error" in results_internal[-1]
    assert "embeddings" in results_biolm[-1]

    assert "embeddings" in results_async[1]
    assert "embeddings" in results_internal[1]
    assert "embeddings" in results_biolm[1]

@pytest.mark.asyncio
async def test_large_batch_encode_stop_on_error_autocompute(model):
    items = [{"sequence": random_sequence()} for _ in range(10)] + \
            [{"sequence": "BAD::BAD"}] + \
            [{"sequence": random_sequence()} for _ in range(10)]
    assert len(items) == 21
    max_batch = await model._get_max_batch_size(model.model_name, "encode") or 1
    batches = list(batch_iterable(items, max_batch))
    print("Batch sizes:", [len(b) for b in batches])
    # Find which batch contains the error
    error_batch_idx = next(i for i, batch in enumerate(batches) if any("BAD::BAD" in d["sequence"] for d in batch))
    expected = sum(len(b) for b in batches[:error_batch_idx+1])
    results_async = await model.encode(items=items, stop_on_error=True)
    assert len(results_async) == expected


# @pytest.mark.asyncio
# async def test_large_batch_encode_stop_on_error_v2(model):
#     # Prepare items: 4 valid, 1 invalid, 4 valid
#     items = [{"sequence": random_sequence()} for _ in range(4)] + \
#             [{"sequence": "BAD::BAD"}] + \
#             [{"sequence": random_sequence()} for _ in range(4)]

#     # Get the batch size from the schema
#     max_batch = await model._get_max_batch_size(model.model_name, "encode") or 1

#     # Run the three methods
#     results_async = await model.encode(items=items, stop_on_error=True)
#     results_internal = await model._batch_call_autoschema_or_manual("encode", items, stop_on_error=True)
#     loop = asyncio.get_event_loop()
#     def run_biolm():
#         return BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=True, raise_httpx=False)
#     results_biolm = await loop.run_in_executor(None, run_biolm)

#     # Find the batch where the error occurs
#     batches = [items[i:i+max_batch] for i in range(0, len(items), max_batch)]
#     expected_results = []
#     for batch in batches:
#         # Simulate what the client does: if any item is invalid, the whole batch is error
#         if any("BAD::BAD" in d["sequence"] for d in batch):
#             expected_results.extend([{"error": "..."}] * len(batch))
#             break
#         else:
#             expected_results.extend([{"embeddings": "..."}] * len(batch))

#     # Now check the actual results
#     for results in (results_async, results_internal, results_biolm):
#         assert isinstance(results, list)
#         assert len(results) == len(expected_results)
#         # Check that all results before the first error are successes
#         for r, e in zip(results, expected_results):
#             if "embeddings" in e:
#                 assert "embeddings" in r
#             else:
#                 assert "error" in r
#         # After the first error, there should be no more results
#         # (already enforced by the length check above)

@pytest.mark.asyncio
async def test_biolm_stop_on_error_shorter_results(model):
    # 10 valid, 1 invalid, 10 valid
    items = [{"sequence": random_sequence()} for _ in range(10)] + \
            [{"sequence": "BAD::BAD"}] + \
            [{"sequence": random_sequence()} for _ in range(10)]
    items = [[item] for item in items]
    # Async client and old method: continue on error
    results_async = await model.encode(items=items, stop_on_error=False)
    results_internal = await model._batch_call_autoschema_or_manual("encode", items, stop_on_error=False)
    # BioLM: stop on error
    loop = asyncio.get_event_loop()
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=True, raise_httpx=False, retry_error_batches=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    # BioLM should have fewer results than the others
    assert len(results_biolm) < len(results_async)
    assert len(results_biolm) < len(results_internal)
    # The others should be full length
    assert len(results_async) == len(items)
    assert len(results_internal) == len(items)
    # The last result in BioLM should be an error
    assert "error" in results_biolm[-1]
    # The first 10 should be valid
    for r in results_biolm[:10]:
        assert "embeddings" in r

@pytest.mark.asyncio
async def test_predict_stop_on_error_vs_continue(model):
    # 1 valid, 1 invalid, 1 valid
    items = [
        [{"sequence": "MDN<mask>LE"}],
        [{"sequence": "MENDELSEMYEFFF<mask>EFMLYRRTELSYYYUPPPPPU::"}],
        [{"sequence": "MD<mask>ELE"}]
    ]
    # stop_on_error=False: should return all results
    results_continue = await model.predict(items=items, stop_on_error=False)
    assert isinstance(results_continue, list)
    assert len(results_continue) == 3
    # stop_on_error=True: should stop at first error (so only 2 results)
    results_stop = await model.predict(items=items, stop_on_error=True)
    assert isinstance(results_stop, list)
    assert len(results_stop) < len(results_continue)
    assert len(results_stop) == 2
    # The last result should be an error
    assert "error" in results_stop[-1]
    # The first should be a valid prediction
    assert "logits" in results_stop[0]


@pytest.mark.asyncio
async def test_large_batch_encode_with_errors_retry_on():
    """With retry_error_batches=True (default), failed batches are retried per item.
    Good items in a bad batch get embeddings; bad items get errors. Order is preserved.
    Exercise all public interfaces: async client, sync client (BioLMApi), BioLM(), biolm(), Model().
    """
    # 20 good, 5 BAD at indices 20-24, 15 good
    items = [{"sequence": random_sequence()} for _ in range(20)] + \
            [{"sequence": "BAD::BAD"} for _ in range(5)] + \
            [{"sequence": random_sequence()} for _ in range(15)]
    bad_indices = {20, 21, 22, 23, 24}
    loop = asyncio.get_event_loop()

    # 1. Async client (explicit default)
    client_async = BioLMApiClient("esm2-8m", raise_httpx=False, unwrap_single=False, retry_error_batches=True)
    results_async = await client_async.encode(items=items, stop_on_error=False)

    # 2. Sync client (BioLMApi)
    client_sync = BioLMApi("esm2-8m", raise_httpx=False, unwrap_single=False, retry_error_batches=True)
    results_sync = await loop.run_in_executor(
        None,
        lambda: client_sync.encode(items=items, stop_on_error=False),
    )

    # 3. BioLM() class
    results_biolm = await loop.run_in_executor(
        None,
        lambda: BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=False, raise_httpx=False),
    )

    # 4. biolm() top-level function
    results_biolm_fn = await loop.run_in_executor(
        None,
        lambda: biolm(entity="esm2-8m", action="encode", items=items, stop_on_error=False, raise_httpx=False),
    )

    # 5. Model() – user-facing wrapper over BioLMApi
    def run_model():
        model = Model("esm2-8m", raise_httpx=False, retry_error_batches=True)
        return model.encode(items=items, stop_on_error=False, progress=False)
    results_model = await loop.run_in_executor(None, run_model)

    all_results = [
        ("async", results_async),
        ("sync", results_sync),
        ("BioLM()", results_biolm),
        ("biolm()", results_biolm_fn),
        ("Model()", results_model),
    ]
    for name, results in all_results:
        assert len(results) == len(items), f"{name}: length"
        for i, r in enumerate(results):
            assert isinstance(r, dict), f"{name} index {i}"
            if i in bad_indices:
                assert "error" in r, f"{name} index {i} should be error, got keys {list(r.keys())}"
            else:
                assert "embeddings" in r, f"{name} index {i} should be embeddings, got keys {list(r.keys())}"

    # All interfaces must return the same pattern (order and content type per index)
    for i in range(len(items)):
        patterns = [("error" in res[i]) for _name, res in all_results]
        assert all(p == patterns[0] for p in patterns), f"index {i}: mismatch across interfaces {[n for n, _ in all_results]}"
