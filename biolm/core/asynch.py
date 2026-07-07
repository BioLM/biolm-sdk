import asyncio
import functools
import gzip
import json
from asyncio import create_task, gather, run
from itertools import zip_longest
from typing import Dict, List, Optional

import aiohttp.resolver
from aiohttp import ClientSession

from biolm.core.auth import get_user_auth_header
from biolm.core.const import BASE_API_URL, BASE_API_URL_V1, MULTIPROCESS_THREADS

aiohttp.resolver.DefaultResolver = aiohttp.resolver.AsyncResolver


async def get_one(session: ClientSession, url: str) -> None:
    print("Requesting", url)
    async with session.get(url) as resp:
        text = await resp.text()
        # await sleep(2)  # for demo purposes
        text_resp = text.strip().split("\n", 1)[0]
        print("Got response from", url, text_resp)
        return text_resp


async def get_one_biolm(
    session: ClientSession,
    url: str,
    pload: dict,
    headers: dict,
    response_key: str = None,
    compress_requests: bool = True,
    compress_threshold: int = 256,
) -> None:
    print("Requesting", url)
    pload_batch = pload.pop("batch")
    pload_batch_size = pload.pop("batch_size")
    t = aiohttp.ClientTimeout(
        total=1600,  # 27 mins
        # total timeout (time consists connection establishment for
        # a new connection or waiting for a free connection from a
        # pool if pool connection limits are exceeded) default value
        # is 5 minutes, set to `None` or `0` for unlimited timeout
        sock_connect=None,
        # Maximal number of seconds for connecting to a peer for a
        # new connection, not given from a pool. See also connect.
        sock_read=None
        # Maximal number of seconds for reading a portion of data from a peer
    )
    
    # Check if we should compress
    request_headers = dict(headers)
    # Add Accept-Encoding: gzip to support compressed responses
    if "Accept-Encoding" not in request_headers:
        request_headers["Accept-Encoding"] = "gzip, deflate"
    request_data = None
    if compress_requests:
        # Serialize JSON to check size
        json_bytes = json.dumps(pload, ensure_ascii=False).encode("utf-8")
        
        if len(json_bytes) > compress_threshold:
            # Compress the payload in a thread pool to avoid blocking the event loop
            # Use compression level 6 (good balance between speed and compression ratio)
            # aiohttp doesn't have built-in async compression, so we use run_in_executor
            loop = asyncio.get_event_loop()
            compressed_body = await loop.run_in_executor(
                None, functools.partial(gzip.compress, json_bytes, compresslevel=6)
            )
            request_headers["Content-Encoding"] = "gzip"
            request_headers["Content-Type"] = "application/json"
            request_headers["Content-Length"] = str(len(compressed_body))
            request_data = compressed_body
        else:
            # Payload too small, send as JSON
            request_data = pload
    else:
        # Compression disabled, send as JSON
        request_data = pload
    
    # Send request with appropriate data format
    resp_json = None
    status_code = None
    if isinstance(request_data, bytes):
        # Compressed data - send as bytes
        async with session.post(url, headers=request_headers, data=request_data, timeout=t) as resp:
            status_code = resp.status
            # Check if response is compressed
            content_encoding = resp.headers.get('Content-Encoding', '').lower()
            if content_encoding == 'gzip':
                # Response is compressed - aiohttp should decompress automatically,
                # but if not, handle it manually
                try:
                    # Try reading as JSON first (aiohttp should have decompressed)
                    resp_json = await resp.json()
                except Exception:
                    # If that fails, read raw bytes and decompress manually
                    try:
                        raw_bytes = await resp.read()
                        # Decompress in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        decompressed = await loop.run_in_executor(None, gzip.decompress, raw_bytes)
                        resp_json = json.loads(decompressed.decode('utf-8'))
                    except Exception:
                        # Fallback: try text
                        resp_text = await resp.text()
                        try:
                            resp_json = json.loads(resp_text)
                        except Exception:
                            resp_json = resp_text
            else:
                # Not compressed, read normally
                resp_json = await resp.json()
    else:
        # Uncompressed data - send as JSON
        async with session.post(url, headers=request_headers, json=request_data, timeout=t) as resp:
            status_code = resp.status
            # Check if response is compressed
            content_encoding = resp.headers.get('Content-Encoding', '').lower()
            if content_encoding == 'gzip':
                # Response is compressed - aiohttp should decompress automatically,
                # but if not, handle it manually
                try:
                    # Try reading as JSON first (aiohttp should have decompressed)
                    resp_json = await resp.json()
                except Exception:
                    # If that fails, read raw bytes and decompress manually
                    try:
                        raw_bytes = await resp.read()
                        # Decompress in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        decompressed = await loop.run_in_executor(None, gzip.decompress, raw_bytes)
                        resp_json = json.loads(decompressed.decode('utf-8'))
                    except Exception:
                        # Fallback: try text
                        resp_text = await resp.text()
                        try:
                            resp_json = json.loads(resp_text)
                        except Exception:
                            resp_json = resp_text
            else:
                # Not compressed, read normally
                resp_json = await resp.json()
    
    # Process response (same for both compressed and uncompressed)
    resp_json["batch"] = pload_batch
    expected_root_key = response_key
    to_ret = []
    
    # Determine if this is an error response:
    # - Non-200 status code, OR
    # - Top-level "error" key (v3 can return validation errors with 200 status)
    is_error = (not status_code or status_code != 200) or (isinstance(resp_json, dict) and "error" in resp_json)
    
    if is_error:
        # Error response - normalize and apply to all items in batch
        error_dict = resp_json.copy()
        # Remove batch info from error dict
        error_dict.pop("batch", None)
        # Ensure we have an error key for consistency
        if "error" not in error_dict and ("detail" in error_dict or "description" in error_dict):
            # Promote detail/description to error key if error key doesn't exist
            error_dict["error"] = error_dict.get("detail") or error_dict.get("description")
        list_of_individual_seq_results = [error_dict] * pload_batch_size
    elif status_code and status_code == 200:
        # Success response - extract results
        if expected_root_key in resp_json:
            list_of_individual_seq_results = resp_json[expected_root_key]
        else:
            # Fallback if expected key not found
            raise ValueError(f"Expected key '{expected_root_key}' not found in response: {resp_json}")
    else:
        raise ValueError("Unexpected response in parser")
    
    for idx, item in enumerate(list_of_individual_seq_results):
        d = {"status_code": status_code, "batch_id": pload_batch, "batch_item": idx}
        if is_error:
            # Error case - put all error keys at root level
            if isinstance(item, dict):
                d.update(item)
            else:
                d["error"] = item
        else:
            # Success case - wrap in expected_root_key structure
            d[expected_root_key] = []
            d[expected_root_key].append(item)
        to_ret.append(d)
    return to_ret

        # text = await resp.text()
        # await sleep(2)  # for demo purposes
        # text_resp = text.strip().split("\n", 1)[0]
        # print("Got response from", url, text_resp)


async def async_range(count):
    for i in range(count):
        yield (i)
        await asyncio.sleep(0.0)


async def get_all(urls: List[str], num_concurrent: int) -> list:
    url_iterator = iter(urls)
    keep_going = True
    results = []
    async with ClientSession() as session:
        while keep_going:
            tasks = []
            for _ in range(num_concurrent):
                try:
                    url = next(url_iterator)
                except StopIteration:
                    keep_going = False
                    break
                new_task = create_task(get_one(session, url))
                tasks.append(new_task)
            res = await gather(*tasks)
            results.extend(res)
    return results


async def get_all_biolm(
    url: str,
    ploads: List[Dict],
    headers: dict,
    num_concurrent: int,
    response_key: str = None,
    compress_requests: bool = True,
    compress_threshold: int = 256,
) -> list:
    ploads_iterator = iter(ploads)
    keep_going = True
    results = []
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=50, ttl_dns_cache=60)
    ov_tout = aiohttp.ClientTimeout(
        total=None,
        # total timeout (time consists connection establishment for
        # a new connection or waiting for a free connection from a
        # pool if pool connection limits are exceeded) default value
        # is 5 minutes, set to `None` or `0` for unlimited timeout
        sock_connect=None,
        # Maximal number of seconds for connecting to a peer for a
        # new connection, not given from a pool. See also connect.
        sock_read=None
        # Maximal number of seconds for reading a portion of data from a peer
    )
    async with ClientSession(connector=connector, timeout=ov_tout) as session:
        while keep_going:
            tasks = []
            for _ in range(num_concurrent):
                try:
                    pload = next(ploads_iterator)
                except StopIteration:
                    keep_going = False
                    break
                new_task = create_task(
                    get_one_biolm(session, url, pload, headers, response_key, compress_requests, compress_threshold)
                )
                tasks.append(new_task)
            res = await gather(*tasks)
            results.extend(res)
    return results


async def async_main(urls, concurrency) -> list:
    return await get_all(urls, concurrency)


async def async_api_calls(model_name, action, headers, payloads, response_key=None, api_version=3, compress_requests=True, compress_threshold=256):
    """Hit an arbitrary BioLM model inference API."""
    # Normally would POST multiple sequences at once for greater efficiency,
    # but for simplicity sake will do one at at time right now
    if api_version == 1:
        url = f"{BASE_API_URL_V1}/models/{model_name}/{action}/"
    else:
        url = f"{BASE_API_URL}/{model_name}/{action}/"

    if not isinstance(payloads, (list, dict)):
        err = "API request payload must be a list or dict, got {}"
        raise AssertionError(err.format(type(payloads)))

    concurrency = int(MULTIPROCESS_THREADS)
    return await get_all_biolm(url, payloads, headers, concurrency, response_key, compress_requests, compress_threshold)

    # payload = json.dumps(payload)
    # session = requests_retry_session()
    # tout = urllib3.util.Timeout(total=180, read=180)
    # response = retry_minutes(session, url, headers, payload, tout, mins=10)
    # # If token expired / invalid, attempt to refresh.
    # if response.status_code == 401 and os.path.exists(ACCESS_TOK_PATH):
    #     # Add jitter to slow down in case we're multiprocessing so all threads
    #     # don't try to re-authenticate at once
    #     time.sleep(random.random() * 4)
    #     with open(ACCESS_TOK_PATH, 'r') as f:
    #         access_refresh_dict = json.load(f)
    #     refresh = access_refresh_dict.get('refresh')
    #     if not refresh_access_token(refresh):
    #         err = "Unauthenticated! Please run `biolmai status` to debug or " \
    #               "`biolmai login`."
    #         raise AssertionError(err)
    #     headers = get_user_auth_header()  # Need to re-get these now
    #     response = retry_minutes(session, url, headers, payload, tout, mins=10)
    # return response


def async_api_call_wrapper(grouped_df, slug, action, payload_maker, response_key, api_version=3, key="sequence", params=None, compress_requests=True, compress_threshold=256):
    """Wrap API calls to assist with sequence validation as a pre-cursor to
    each API call.
    """
    model_name = slug
    # payload = payload_maker(grouped_df)
    if api_version == 1:
        init_ploads = grouped_df.groupby("batch").apply(
            payload_maker, include_batch_size=True
        )
    else:
        init_ploads = grouped_df.groupby("batch").apply(
            payload_maker, key=key, params=params, include_batch_size=True
        )
    ploads = init_ploads.to_list()
    init_ploads = init_ploads.to_frame(name="pload")
    init_ploads["batch"] = init_ploads.index
    init_ploads = init_ploads.reset_index(drop=True)
    assert len(ploads) == init_ploads.shape[0]
    for inst, b in zip_longest(ploads, init_ploads["batch"].to_list()):
        if inst is None or b is None:
            raise ValueError(
                "ploads and init_ploads['batch'] are not of the same length"
            )
        inst["batch"] = b

    headers = get_user_auth_header()  # Need to pull each time
    # urls = [
    #     "https://github.com",
    #     "https://stackoverflow.com",
    #     "https://python.org",
    # ]
    # concurrency = 3
    api_resp = run(async_api_calls(model_name, action, headers, ploads, response_key, api_version, compress_requests, compress_threshold))
    api_resp = [item for sublist in api_resp for item in sublist]
    api_resp = sorted(api_resp, key=lambda x: x["batch_id"])
    # print(api_resp)
    # api_resp = biolmai.api_call(model_name, action, headers, payload,
    #                             response_key)
    # resp_json = api_resp.json()
    # batch_id = int(grouped_df.batch.iloc[0])
    # batch_size = grouped_df.shape[0]
    # response = predict_resp_many_in_one_to_many_singles(
    #     resp_json, api_resp.status_code, batch_id, None, batch_size)
    return api_resp
