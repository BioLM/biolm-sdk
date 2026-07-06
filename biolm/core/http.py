import asyncio
import atexit
import functools
import gzip
import json
import os
import threading
import time
from collections import namedtuple, OrderedDict
from collections.abc import Iterable
from contextlib import asynccontextmanager
from itertools import chain
from json import dumps as json_dumps
from typing import Callable
from typing import Optional, Union, List, Any, Dict, Tuple

import aiofiles
import httpx
import httpx._content
from httpx import AsyncHTTPTransport
from httpx import ByteStream
from synchronicity import Synchronizer

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version


def custom_httpx_encode_json(json: Any) -> Tuple[Dict[str, str], ByteStream]:
    # disable ascii for json_dumps
    body = json_dumps(json, ensure_ascii=False).encode("utf-8")
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, ByteStream(body)

# fix encoding utf-8 bug
httpx._content.encode_json = custom_httpx_encode_json

import sys

def debug(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

import logging

# Turn this on to dev lots of logs
if os.environ.get("DEBUG", '').upper().strip() in ('TRUE', '1'):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,  # Python 3.8+
    )

from biolm.core.const import BIOLM_BASE_API_URL, ACCESS_TOK_PATH
from biolm.core.utils import is_list_of_lists, batch_iterable

TIMEOUT_MINS = 20  # Match API server's keep-alive/timeout
DEFAULT_TIMEOUT = httpx.Timeout(TIMEOUT_MINS * 60, connect=10.0)

# Connection pool limits
DEFAULT_LIMITS = httpx.Limits(
    max_connections=50,               # Total concurrent connections
    max_keepalive_connections=20,    # Idle connections to keep alive (default)
    keepalive_expiry=65.0            # Keep idle connections for 65s
)

# Retry configuration for network errors
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 4.0  # Start with 1 second

# Mapping of httpx exception types to user-friendly error messages
HTTPX_EXCEPTION_MESSAGES = {
    httpx.ReadError: "Connection read error: The server closed the connection unexpectedly or data could not be read",
    httpx.ConnectError: "Connection error: Failed to establish a connection to the server",
    httpx.NetworkError: "Network error: A network-related error occurred",
    httpx.WriteError: "Connection write error: Failed to send data to the server",
    httpx.CloseError: "Connection close error: Failed to close the connection properly",
    httpx.TimeoutException: "Request timeout: The request took too long to complete",
    httpx.ConnectTimeout: "Connection timeout: Failed to establish a connection within the timeout period",
    httpx.ReadTimeout: "Read timeout: The server did not send data within the timeout period",
    httpx.WriteTimeout: "Write timeout: Failed to send data within the timeout period",
    httpx.PoolTimeout: "Connection pool timeout: No connections available in the pool",
    httpx.HTTPStatusError: "HTTP error: The server returned an error status code",
    httpx.RequestError: "Request error: An error occurred while making the request",
    httpx.TransportError: "Transport error: An error occurred at the transport layer",
}

LookupResult = namedtuple("LookupResult", ["data", "raw"])


def _detect_execution_context() -> str:
    """
    Detect the execution environment.
    
    Returns:
        'jupyter_with_loop': Jupyter/IPython with active event loop
        'jupyter_no_loop': Jupyter/IPython without active loop (rare)
        'async_context': Async application (FastAPI, etc.) with active loop
        'sync_script': Standard Python script (no active loop)
        'unknown': Detection failed (fallback)
    """
    try:
        # Check if we're in Jupyter/IPython
        if 'IPython' in sys.modules:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                # Check if there's a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    return 'jupyter_with_loop'
                except RuntimeError:
                    return 'jupyter_no_loop'
        
        # Check if we're in an async context (but not Jupyter)
        try:
            loop = asyncio.get_running_loop()
            return 'async_context'
        except RuntimeError:
            pass
        
        return 'sync_script'
    except Exception:
        return 'unknown'


def _ensure_nest_asyncio_for_jupyter():
    """
    Automatically apply nest_asyncio if we're in Jupyter with an active event loop.
    
    This enables sync wrappers (BioLM, BioLMApi) to work in Jupyter without
    requiring users to manually call nest_asyncio.apply().
    
    This function is idempotent - calling it multiple times is safe.
    """
    context = _detect_execution_context()
    
    if context == 'jupyter_with_loop':
        try:
            import nest_asyncio
            # Check if already applied (nest_asyncio sets this flag)
            if not getattr(nest_asyncio, '_applied', False):
                nest_asyncio.apply()
        except ImportError:
            # nest_asyncio should be a dependency, but handle gracefully
            import warnings
            warnings.warn(
                "nest_asyncio not available. Sync wrappers may not work in Jupyter. "
                "Install nest_asyncio or use BioLMApiClient directly with await.",
                UserWarning
            )
        except Exception as e:
            # If nest_asyncio.apply() fails, log but don't crash
            import warnings
            warnings.warn(
                f"Failed to apply nest_asyncio: {e}. "
                "Sync wrappers may not work in Jupyter.",
                UserWarning
            )


class _SharedClientFactory:
    """
    Factory for creating and caching shared httpx.AsyncClient instances.
    Clients are cached by event loop ID and configuration (base_url, headers, timeout, limits, http2, transport)
    to enable connection pooling across multiple HttpClient instances within the same event loop.
    """
    
    def __init__(self):
        # Nested dict: loop_id -> config_key -> client
        self._cache: Dict[int, Dict[Tuple, httpx.AsyncClient]] = {}
        self._async_lock = asyncio.Lock()  # For async operations within a loop
        self._thread_lock = threading.Lock()  # For thread-safe cache access
    
    def _make_cache_key(
        self,
        base_url: str,
        headers: Dict[str, str],
        timeout: httpx.Timeout,
        limits: httpx.Limits,
        http2: bool,
        transport: Optional[AsyncHTTPTransport]
    ) -> Tuple:
        """Create a hashable key from client configuration."""
        # Convert headers dict to sorted tuple for hashability
        headers_key = tuple(sorted(headers.items()))
        
        # Extract timeout values
        # httpx.Timeout has connect, read, write, and pool attributes
        timeout_key = (
            getattr(timeout, 'connect', None),
            getattr(timeout, 'read', None),
            getattr(timeout, 'write', None),
            getattr(timeout, 'pool', None),
        )
        
        # Extract limits values
        limits_key = (
            limits.max_connections,
            limits.max_keepalive_connections,
            limits.keepalive_expiry,
        )
        
        # For transport, use http2 flag since transport is created with http2 parameter
        # We don't need to include the transport object itself in the key
        transport_key = http2
        
        return (base_url, headers_key, timeout_key, limits_key, transport_key)
    
    async def get_or_create_client(
        self,
        base_url: str,
        headers: Dict[str, str],
        timeout: httpx.Timeout,
        limits: httpx.Limits,
        http2: bool,
        transport: Optional[AsyncHTTPTransport]
    ) -> httpx.AsyncClient:
        """Get or create a shared httpx.AsyncClient instance for the current event loop."""
        # Get current event loop ID to ensure clients are loop-specific
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # No running loop, use a sentinel value
            loop_id = None
        
        cache_key = self._make_cache_key(base_url, headers, timeout, limits, http2, transport)
        
        # Thread-safe cache check
        with self._thread_lock:
            # Get or create loop-specific cache
            if loop_id not in self._cache:
                self._cache[loop_id] = {}
            
            loop_cache = self._cache[loop_id]
            
            # Check if client exists and is still open
            if cache_key in loop_cache:
                client = loop_cache[cache_key]
                # Verify client is still valid (not closed)
                if not getattr(client, 'is_closed', False):
                    return client
                # Client is closed, remove from cache
                del loop_cache[cache_key]
        
        # Async-safe client creation
        async with self._async_lock:
            # Double-check after acquiring async lock
            with self._thread_lock:
                if loop_id in self._cache and cache_key in self._cache[loop_id]:
                    return self._cache[loop_id][cache_key]
            
            # Create new client bound to current event loop
            if transport:
                client = httpx.AsyncClient(
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                    transport=transport,
                    limits=limits,
                )
            else:
                client = httpx.AsyncClient(
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                    limits=limits,
                )
            
            # Thread-safe cache update
            with self._thread_lock:
                if loop_id not in self._cache:
                    self._cache[loop_id] = {}
                self._cache[loop_id][cache_key] = client
            
            return client
    
    async def cleanup_all(self):
        """Close all cached clients and clear the cache."""
        # Get all clients thread-safely
        clients_to_close = []
        with self._thread_lock:
            for loop_cache in self._cache.values():
                clients_to_close.extend(loop_cache.values())
            self._cache.clear()
        
        # Close clients async-safely
        async with self._async_lock:
            for client in clients_to_close:
                try:
                    if not getattr(client, 'is_closed', False):
                        await client.aclose()
                except Exception:
                    # Ignore errors during cleanup
                    pass
    
    async def cleanup_loop(self, loop_id: Optional[int] = None):
        """Clean up clients for a specific event loop (or current loop if None)."""
        if loop_id is None:
            try:
                loop = asyncio.get_running_loop()
                loop_id = id(loop)
            except RuntimeError:
                return
        
        # Get clients to close thread-safely
        clients_to_close = []
        with self._thread_lock:
            if loop_id in self._cache:
                clients_to_close = list(self._cache[loop_id].values())
                del self._cache[loop_id]
        
        # Close clients async-safely
        async with self._async_lock:
            for client in clients_to_close:
                try:
                    if not getattr(client, 'is_closed', False):
                        await client.aclose()
                except Exception:
                    pass


# Global shared client factory instance
_shared_client_factory = _SharedClientFactory()


def _cleanup_shared_clients():
    """Cleanup function registered with atexit to close all shared clients."""
    try:
        # Try to get existing event loop
        try:
            loop = asyncio.get_running_loop()
            # Loop is running, we can't easily schedule async cleanup from sync context
            # In this case, we'll just skip cleanup (connections will close on process exit)
            return
        except RuntimeError:
            # No running loop, try to get or create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Loop is closed, create a new one
                    asyncio.run(_shared_client_factory.cleanup_all())
                else:
                    # Loop exists but not running, run cleanup
                    loop.run_until_complete(_shared_client_factory.cleanup_all())
            except RuntimeError:
                # No event loop exists, create a new one
                asyncio.run(_shared_client_factory.cleanup_all())
    except Exception:
        # Ignore all errors during cleanup - process is exiting anyway
        pass


# Register cleanup on process exit
atexit.register(_cleanup_shared_clients)

_synchronizer = Synchronizer()

if not hasattr(_synchronizer, "sync"):
    if hasattr(_synchronizer, "wrap"):
        _synchronizer.sync = _synchronizer.wrap
    if hasattr(_synchronizer, "create_blocking"):
        _synchronizer.sync = _synchronizer.create_blocking
    else:
        raise ImportError(f"Your version of 'synchronicity' ({version('synchronicity')}) is incompatible.")

# Automatically apply nest_asyncio in Jupyter contexts
# This must happen before BioLMApi wrapper is created
_ensure_nest_asyncio_for_jupyter()

def type_check(param_types: Dict[str, Any]):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for param, expected_type in param_types.items():
                value = kwargs.get(param)
                if value is None and len(args) > 0:
                    arg_names = func.__code__.co_varnames
                    if param in arg_names:
                        idx = arg_names.index(param)
                        if idx < len(args):
                            value = args[idx]
                if value is not None:
                    # Allow tuple of types or single type
                    if not isinstance(expected_type, tuple):
                        expected_types = (expected_type,)
                    else:
                        expected_types = expected_type
                    if not isinstance(value, expected_types):
                        # Special case: 'items' also accepts iterables (generators, iterators)
                        # but not str, bytes, or dict (those are single-item types)
                        if (
                            param == "items"
                            and isinstance(value, Iterable)
                            and not isinstance(value, (str, bytes, dict))
                        ):
                            pass
                        else:
                            type_names = ", ".join([t.__name__ for t in expected_types])
                            raise TypeError(
                                f"Parameter '{param}' must be of type {type_names}, got {type(value).__name__}"
                            )
                    # Check for empty list/tuple
                    # if isinstance(value, (list, tuple)) and len(value) == 0:
                    #     raise ValueError(
                    #         f"Parameter '{param}' must not be an empty {type(value).__name__}"
                    #     )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class AsyncRateLimiter:
    def __init__(self, max_calls: int, period: float):
        self._max_calls = max_calls
        self._period = period
        self._lock = asyncio.Lock()
        self._calls = []

    @asynccontextmanager
    async def limit(self):
        async with self._lock:
            now = time.monotonic()
            # Remove calls outside the window
            self._calls = [t for t in self._calls if now - t < self._period]
            if len(self._calls) >= self._max_calls:
                sleep_time = self._period - (now - self._calls[0])
                await asyncio.sleep(max(0, sleep_time))
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._period]
            self._calls.append(time.monotonic())
        yield

def parse_rate_limit(rate: str):
    # e.g. "1000/second", "60/minute"
    if not rate:
        return None
    num, per = rate.strip().split("/")
    num = int(num)
    per = per.strip().lower()
    if per == "second":
        return num, 1.0
    elif per == "minute":
        return num, 60.0
    else:
        raise ValueError(f"Unknown rate period: {per}")

class CredentialsProvider:
    @staticmethod
    def get_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
        if api_key:
            return {"Authorization": f"Token {api_key}"}
        api_token = os.environ.get("BIOLM_TOKEN") or os.environ.get("BIOLMAI_TOKEN") or os.environ.get("BIOLM_TOKEN")
        if api_token:
            return {"Authorization": f"Token {api_token}"}
        if os.path.exists(ACCESS_TOK_PATH):
            with open(ACCESS_TOK_PATH) as f:
                creds = json.load(f)
            access = creds.get("access")
            refresh = creds.get("refresh")
            return {
                "Cookie": f"access={access};refresh={refresh}",
            }
        raise AssertionError("No credentials found. Set BIOLM_TOKEN (or BIOLMAI_TOKEN) or run `biolm login`.")


class HttpClient:

    def __init__(
        self, 
        base_url: str, 
        headers: Dict[str, str], 
        timeout: httpx.Timeout,
        compress_requests: bool = True,
        compress_threshold: int = 256,
        limits: Optional[httpx.Limits] = None,
        http2: bool = True
    ):
        self._base_url = base_url.rstrip("/") + "/"
        self._headers = headers
        # Add Accept-Encoding: gzip to headers to support compressed responses
        if 'Accept-Encoding' not in self._headers:
            self._headers['Accept-Encoding'] = 'gzip, deflate'
        self._timeout = timeout
        self._compress_requests = compress_requests
        self._compress_threshold = compress_threshold
        self._limits = limits or DEFAULT_LIMITS
        self._http2 = http2
        self._transport = None
        # Create transport with HTTP/2 enabled
        self._transport = AsyncHTTPTransport(http2=http2)

    def _hub_style_paths(self) -> bool:
        return self._base_url.rstrip("/").endswith("/api/v1")

    async def get_async_client(self) -> httpx.AsyncClient:
        """Get or create a shared httpx.AsyncClient instance from the factory."""
        return await _shared_client_factory.get_or_create_client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
            limits=self._limits,
            http2=self._http2,
            transport=self._transport
        )

    async def _post_with_retry(
        self, 
        client: httpx.AsyncClient, 
        endpoint: str, 
        **request_kwargs
    ) -> httpx.Response:
        """POST with retry logic for network errors (ReadError, ConnectError, NetworkError)."""
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return await client.post(endpoint, **request_kwargs)
            except (httpx.ReadError, httpx.ConnectError, httpx.NetworkError) as e:
                last_exception = e
                # Don't retry on the last attempt
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                # Last attempt failed, re-raise
                raise
            # Don't retry on HTTP status errors (4xx, 5xx) - these are not network errors
            except httpx.HTTPStatusError:
                raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise httpx.NetworkError("Failed to make request after retries")

    async def post(self, endpoint: str, payload: dict, extra_headers: Optional[dict] = None) -> httpx.Response:
        """POST with optional *extra_headers* added just for this request."""
        client = await self.get_async_client()
        # Remove leading slash; biolm-hub v1 routes omit trailing slashes.
        endpoint = endpoint.lstrip("/")
        if not self._hub_style_paths() and not endpoint.endswith("/"):
            endpoint += "/"

        headers = None
        if extra_headers:
            headers = {**client.headers, **extra_headers}

        # Check if we should compress
        if self._compress_requests:
            # Use the same JSON encoding as httpx to get accurate size
            json_bytes = json_dumps(payload, ensure_ascii=False).encode("utf-8")
            
            if len(json_bytes) > self._compress_threshold:
                # Compress in a thread pool to avoid blocking the event loop
                # Use compression level 6 (good balance between speed and compression ratio)
                # Use asyncio.to_thread() for Python 3.9+, fallback to run_in_executor for older versions
                try:
                    compressed_body = await asyncio.to_thread(gzip.compress, json_bytes, compresslevel=6)
                except AttributeError:
                    # Python < 3.9: use run_in_executor
                    loop = asyncio.get_event_loop()
                    compressed_body = await loop.run_in_executor(
                        None, functools.partial(gzip.compress, json_bytes, compresslevel=6)
                    )
                
                if headers is None:
                    # Start with client's default headers but ensure we override Content-Type
                    headers = dict(client.headers)
                else:
                    # Merge extra_headers with client headers, then override for compression
                    headers = {**client.headers, **headers}
                # Set headers for compressed request - these must override any defaults
                # When using content= (raw bytes), we need to explicitly set Content-Type
                # Important: Set these headers explicitly to ensure they override any defaults
                headers["Content-Type"] = "application/json"
                headers["Content-Encoding"] = "gzip"
                headers["Content-Length"] = str(len(compressed_body))
                # Use content= for raw bytes (httpx recommends this over data= for raw content)
                # The headers we provide should override client defaults
                r = await self._post_with_retry(
                    client, endpoint, content=compressed_body, headers=headers
                )
                return r
        
        # Default: send uncompressed JSON
        r = await self._post_with_retry(client, endpoint, json=payload, headers=headers)
        return r

    async def get(self, endpoint: str) -> httpx.Response:
        """GET with retry logic for network errors."""
        client = await self.get_async_client()
        endpoint = endpoint.lstrip("/")
        if not endpoint.endswith("/"):
            endpoint += "/"
        
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return await client.get(endpoint)
            except (httpx.ReadError, httpx.ConnectError, httpx.NetworkError) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except httpx.HTTPStatusError:
                raise
        
        if last_exception:
            raise last_exception
        raise httpx.NetworkError("Failed to make request after retries")


    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        """Close method for backward compatibility. 
        Shared clients are managed by _SharedClientFactory and cleaned up on process exit.
        """
        # No-op: shared clients are cleaned up by the factory
        pass


# Schema cache keyed by (model, action). Event-loop-agnostic so sync and async
# callers (and tests under pytest-xdist) share the same cache without alru_cache's
# "not safe to use across event loops" restriction.
_SCHEMA_CACHE: Dict[Tuple[str, str], Optional[dict]] = {}


class BioLMApiClient:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        raise_httpx: bool = False,
        unwrap_single: bool = True,
        semaphore: 'Optional[Union[int, asyncio.Semaphore]]' = 16,
        rate_limit: 'Optional[str]' = None,
        retry_error_batches: bool = True,
        compress_requests: bool = True,
        compress_threshold: int = 256,
        concurrent_batches: bool = True,  # When True and stop_on_error=False, process batches in parallel (up to semaphore limit)
        http2: bool = True,  # If False, use HTTP/1.1 so each request can use a separate connection (avoids server MAX_CONCURRENT_STREAMS cap)
    ):
        # Use base_url parameter if provided, otherwise use default from const
        final_base_url = base_url if base_url is not None else BIOLM_BASE_API_URL

        self.model_name = model_name
        self.base_url = final_base_url.rstrip("/") + "/"  # Ensure trailing slash
        self.timeout = timeout
        self.raise_httpx = raise_httpx
        self.unwrap_single = unwrap_single
        self._headers = CredentialsProvider.get_auth_headers(api_key)
        self._http_client = HttpClient(
            self.base_url, 
            self._headers, 
            self.timeout,
            compress_requests=compress_requests,
            compress_threshold=compress_threshold,
            http2=http2,
        )
        self._semaphore = None
        self._semaphore_arg = semaphore  # Lazy-init: None, int, or asyncio.Semaphore
        self._rate_limiter = None
        self._rate_limit_lock = None
        self._rate_limit_initialized = False
        self.retry_error_batches = retry_error_batches
        self._concurrent_batches = concurrent_batches

        # RPS limit
        if rate_limit:
            max_calls, period = parse_rate_limit(rate_limit)
            self._rate_limiter = AsyncRateLimiter(max_calls, period)
            self._rate_limit_initialized = True

    async def _get_semaphore(self):
        """Lazy-init semaphore: create on first use to avoid event-loop issues."""
        if self._semaphore is not None:
            return self._semaphore
        if self._semaphore_arg is None:
            return None
        if isinstance(self._semaphore_arg, asyncio.Semaphore):
            self._semaphore = self._semaphore_arg
        else:
            self._semaphore = asyncio.Semaphore(self._semaphore_arg)
        return self._semaphore

    async def _ensure_rate_limit(self, action: Optional[str] = None):
            if self._rate_limit_lock is None:
                self._rate_limit_lock = asyncio.Lock()
            if self._rate_limit_initialized:
                return
            async with self._rate_limit_lock:
                if self._rate_limit_initialized:
                    return
                if self._rate_limiter is None:
                    schema_action = (action or "encode").strip() or "encode"
                    schema = await self.schema(self.model_name, schema_action)
                    throttle_rate = schema.get("throttle_rate") if schema else None
                    if throttle_rate:
                        max_calls, period = parse_rate_limit(throttle_rate)
                        self._rate_limiter = AsyncRateLimiter(max_calls, period)
                self._rate_limit_initialized = True

    @asynccontextmanager
    async def _limit(self):
        """
         Usage:
            # Default: 2 concurrent requests (cancellation protection)
            # BioLMApiClient(...)
            # No throttling: BioLMApiClient(..., semaphore=None)
            # Custom concurrency limit: BioLMApiClient(..., semaphore=5)
            # User's own semaphore: BioLMApiClient(..., semaphore=my_semaphore)
            # RPS limit: BioLMApiClient(..., rate_limit="1000/second")
            # Both: BioLMApiClient(..., semaphore=5, rate_limit="1000/second")
        """
        sem = await self._get_semaphore()
        if sem:
            async with sem:
                if self._rate_limiter:
                    async with self._rate_limiter.limit():
                        yield
                else:
                    yield
        elif self._rate_limiter:
            async with self._rate_limiter.limit():
                yield
        else:
            yield

    async def schema(
        self,
        model: str,
        action: str,
    ) -> Optional[dict]:
        """
        Fetch the JSON schema for a given model and action, with caching.
        Returns the schema dict if successful, else None.
        Uses a module-level cache keyed by (model, action) so it is safe to use
        from any event loop (sync wrapper, async tests, pytest-xdist workers).
        """
        key = (model, action)
        if key in _SCHEMA_CACHE:
            return _SCHEMA_CACHE[key]
        endpoint = f"schema/{model}/{action}/"
        try:
            resp = await self._http_client.get(endpoint)
            if resp.status_code == 200:
                result = resp.json()
            else:
                result = None
        except Exception:
            result = None
        _SCHEMA_CACHE[key] = result
        return result

    @staticmethod
    def extract_max_items(schema: dict) -> Optional[int]:
        """
        Extracts the 'maxItems' value for the 'items' key from the schema.
        Returns the integer value if found, else None.
        """
        try:
            props = schema.get('properties', {})
            items_schema = props.get('items', {})
            max_items = items_schema.get('maxItems')
            if isinstance(max_items, int):
                return max_items
        except Exception:
            pass
        return None

    async def _get_max_batch_size(self, model: str, action: str) -> Optional[int]:
        schema = await self.schema(model, action)
        if schema:
            return self.extract_max_items(schema)
        return None

    async def _retry_batch_individually(
        self, func: str, batch: List[dict], params: Optional[dict], raw: bool
    ) -> list:
        """Retry a failed batch by calling the API once per item. Returns list of results."""
        out = []
        for item in batch:
            single_result = await self.call(func, [item], params=params, raw=raw)
            if isinstance(single_result, list) and len(single_result) == 1:
                out.append(single_result[0])
            else:
                out.append(single_result)
        return out

    async def _process_single_batch(
        self,
        func: str,
        batch: List[dict],
        params: Optional[dict],
        raw: bool,
    ):
        """Process one batch: call API, optionally retry individually on error. Returns batch-level results."""
        batch_results = await self.call(func, batch, params=params, raw=raw)
        if (
            self.retry_error_batches
            and isinstance(batch_results, dict)
            and ('error' in batch_results or 'status_code' in batch_results)
        ):
            batch_results = await self._retry_batch_individually(func, batch, params, raw)
        return batch_results

    async def _fetch_rps_limit_async(self) -> Optional[int]:
        return None
        # Not implemented yet
        try:
            # Reuse existing http_client instead of creating a new client
            resp = await self._http_client.get(f"/{self.model_name}/")
            if resp.status_code == 200:
                meta = resp.json()
                return meta.get("rps_limit") or meta.get("max_rps") or meta.get("requests_per_second")
        except Exception:
            pass
        return None

    async def _api_call(
        self, endpoint: str, payload: dict, raw: bool = False
    ) -> Union[dict, Tuple[Any, httpx.Response]]:
        parts = endpoint.rstrip("/").split("/")
        action = parts[-1] if parts else "encode"
        await self._ensure_rate_limit(action=action)
        async with self._limit():
            resp = await self._http_client.post(endpoint, payload)
        content_type = resp.headers.get("Content-Type", "")

        assert hasattr(resp, 'status_code') or hasattr(resp, 'status') or 'status' in resp or 'status_code' in resp

        # Read response content once and parse as JSON or use as text
        # Note: resp.json() and resp.text both consume the body stream, so we read text first, then parse
        # httpx automatically decompresses responses with Content-Encoding: gzip if Accept-Encoding was set
        # But we'll handle it explicitly to be safe
        resp_json = None
        resp_text = None
        try:
            # Check if response is compressed
            content_encoding = resp.headers.get('Content-Encoding', '').lower()
            if content_encoding == 'gzip':
                # Response is compressed - httpx should have decompressed it automatically,
                # but if not, we'll handle it
                try:
                    # Try reading as text first (httpx should have decompressed)
                    resp_text = resp.text
                except Exception:
                    # If that fails, read raw bytes and decompress manually
                    try:
                        raw_bytes = resp.content
                        # Decompress in thread pool to avoid blocking
                        try:
                            decompressed = await asyncio.to_thread(gzip.decompress, raw_bytes)
                        except AttributeError:
                            loop = asyncio.get_event_loop()
                            decompressed = await loop.run_in_executor(None, gzip.decompress, raw_bytes)
                        resp_text = decompressed.decode('utf-8')
                    except Exception:
                        resp_text = ''
            else:
                # Not compressed, read normally
                resp_text = resp.text
            
            # Try to parse the text as JSON
            if resp_text:
                try:
                    resp_json = json.loads(resp_text)
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, use as text
                    resp_json = resp_text
            else:
                resp_json = ''
        except Exception:
            # If reading text fails, try json() as fallback (may also fail if body already consumed)
            try:
                resp_json = resp.json()
            except Exception:
                resp_text = ''
                resp_json = ''

        assert resp.status_code
        # Check for errors: non-200 status OR top-level error/detail/description keys
        # Note: v3 can return validation errors (200 status) with item-specific errors
        has_error_key = isinstance(resp_json, dict) and ('error' in resp_json or 'detail' in resp_json or 'description' in resp_json)
        is_error_status = resp.status_code >= 400
        
        # Check if response has both error and results (partial success scenario)
        # In this case, we return the response as-is so batch processing can handle item-level errors
        has_results = isinstance(resp_json, dict) and 'results' in resp_json
        
        if is_error_status or (has_error_key and not has_results):
            if 'application/json' in content_type:
                try:
                    # Copy to avoid mutating original response
                    if isinstance(resp_json, dict):
                        error_json = resp_json.copy()
                    else:
                        error_json = resp_json
                    # If the API already returns a dict with "error" or similar, normalize it
                    if isinstance(error_json, (dict, list)):
                        # Normalize error keys - ensure "error" key exists
                        if isinstance(error_json, dict) and 'error' not in error_json:
                            if 'detail' in error_json:
                                error_json['error'] = error_json['detail']
                            elif 'description' in error_json:
                                error_json['error'] = error_json['description']
                        # Handle empty error strings - provide fallback message
                        elif isinstance(error_json, dict) and 'error' in error_json:
                            error_value = error_json.get('error')
                            if not error_value or (isinstance(error_value, str) and not error_value.strip()):
                                # Empty or whitespace-only error string - use fallback
                                error_json['error'] = (
                                    error_json.get('detail') or 
                                    error_json.get('description') or 
                                    f"API returned error status {resp.status_code}"
                                )
                        DEFAULT_STATUS_CODE = 502
                        stat = error_json.get('status', DEFAULT_STATUS_CODE)
                        error_json['status_code'] = resp.status_code or error_json.get('status_code', stat)
                        if raw:
                            return (error_json, resp)
                        if self.raise_httpx:
                            # Use stored resp_text if available, otherwise use error_json as string
                            error_msg = resp_text if resp_text is not None else str(resp_json)
                            raise httpx.HTTPStatusError(message=error_msg, request=resp.request, response=resp)
                        return error_json
                    else:
                        # If the JSON is not a dict or list, wrap it
                        error_info = {'error': error_json, 'status_code': resp.status_code}
                except Exception:
                    # Use stored resp_text if available, otherwise use fallback message
                    error_text = resp_text if resp_text is not None else f"API returned error status {resp.status_code}"
                    error_info = {'error': error_text, 'status_code': resp.status_code}
            else:
                # Use stored resp_text if available, otherwise use fallback message
                error_text = resp_text if resp_text is not None else f"API returned error status {resp.status_code}"
                error_info = {'error': error_text, 'status_code': resp.status_code}
            if raw:
                return (error_info, resp)
            if self.raise_httpx:
                # Use stored resp_text if available, otherwise use error_info error as string
                error_msg = resp_text if resp_text is not None else str(error_info.get('error', ''))
                raise httpx.HTTPStatusError(message=error_msg, request=resp.request, response=resp)
            return error_info

        # Success response (200 status, no error keys)
        # Use the already-parsed resp_json instead of calling resp.json() again
        if isinstance(resp_json, dict):
            data = resp_json
        elif 'application/json' in content_type:
            # If resp_json is not a dict but content-type says JSON, something went wrong
            # Use stored resp_text if available, otherwise use fallback message
            error_msg = resp_text if resp_text is not None else f"Failed to parse JSON response (status {resp.status_code})"
            data = {"error": error_msg, "status_code": resp.status_code}
        else:
            # Use stored resp_text if available, otherwise use fallback message
            error_msg = resp_text if resp_text is not None else f"Failed to read response body (status {resp.status_code})"
            data = {"error": error_msg, "status_code": resp.status_code}
        
        # If response has both error and results, it's a partial success scenario
        # Add status_code and return as-is so batch processing can handle item-level errors
        if isinstance(data, dict) and 'error' in data and 'results' in data:
            # Partial success: some items have errors, some have results
            if 'status_code' not in data:
                data['status_code'] = resp.status_code
            return (data, resp) if raw else data
        
        # Pure success response - return as-is (will be formatted by _format_result)
        return (data, resp) if raw else data

    async def call(self, func: str, items: List[dict], params: Optional[dict] = None, raw: bool = False):
        if not items:
            return items

        endpoint = f"{self.model_name}/{func}/"
        endpoint = endpoint.lstrip("/")
        if self.base_url.rstrip("/").endswith("/api/v1"):
            endpoint = endpoint.rstrip("/")
        payload = {'items': items} if func != 'lookup' else {'query': items}
        if params:
            payload['params'] = params
        try:
            res = await self._api_call(endpoint, payload, raw=raw if func == 'lookup' else False)
        except Exception as e:
            if self.raise_httpx:
                raise
            res = self._format_exception(e, 0)
        res = self._format_result(res)
        if isinstance(res, dict) and ('error' in res or 'status_code' in res):
            return res
        elif isinstance(res, (list, tuple)):
            return list(res)
        else:
            return res

    async def _batch_call_autoschema_or_manual(
        self,
        func: str,
        items,
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        raw: bool = False,
        overwrite: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        if not items:
            return items

        is_lol, first_n, rest_iter = is_list_of_lists(items)

        # Check if file exists and overwrite is False
        if output == 'disk' and not overwrite:
            path = file_path or f"{self.model_name}_{func}_output.jsonl"
            if os.path.exists(path):
                # Read existing file and return its contents
                results = []
                async with aiofiles.open(path, 'r', encoding='utf-8') as file_handle:
                    async for line in file_handle:
                        line = line.strip()
                        if line:
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
                # Return in the same format as memory output would
                if is_lol:
                    return results
                return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results

        results = []

        def parse_validation_errors(batch_results, batch_size):
            """Parse validation errors with items__N__sequence keys and distribute to specific items.
            
            Returns a list of results, one per item in the batch.
            - Items with validation errors get item-specific error dicts
            - Items without errors get the batch error dict (batch-level failure)
            - If results array exists, valid items get their results
            """
            error_dict = batch_results.get('error', {})
            status_code = batch_results.get('status_code', 0)
            
            # Check if this is a validation error with item-specific errors
            is_validation_error = (
                status_code == 200 and 
                isinstance(error_dict, dict) and 
                any(key.startswith('items__') and '__sequence' in key for key in error_dict.keys())
            )
            
            if not is_validation_error:
                # Batch-level error (4xx/5xx): apply to all items
                return [batch_results] * batch_size
            
            # Parse which items have validation errors
            import re
            error_item_indices = set()
            for key in error_dict.keys():
                match = re.match(r'items__(\d+)__', key)
                if match:
                    error_item_indices.add(int(match.group(1)))
            
            # Check if response also has results (partial success scenario)
            if 'results' in batch_results:
                # Partial success: some items have errors, some have results
                results_list = batch_results.get('results', [])
                item_results = []
                for idx in range(batch_size):
                    if idx in error_item_indices:
                        # Item has validation error
                        item_error = {
                            'error': {k: v for k, v in error_dict.items() if f'items__{idx}__' in k},
                            'status_code': status_code
                        }
                        item_results.append(item_error)
                    elif idx < len(results_list):
                        # Item has successful result
                        item_results.append(results_list[idx])
                    else:
                        # Item index out of range - apply batch error as fallback
                        item_results.append(batch_results)
                return item_results
            else:
                # Full batch validation error: all items get error, but preserve item-specific error info
                item_results = []
                for idx in range(batch_size):
                    if idx in error_item_indices:
                        # Item has specific validation error
                        item_error = {
                            'error': {k: v for k, v in error_dict.items() if f'items__{idx}__' in k},
                            'status_code': status_code
                        }
                        item_results.append(item_error)
                    else:
                        # Item doesn't have specific error, but batch failed validation
                        # Apply the full error dict (batch-level validation failure)
                        item_results.append(batch_results)
                return item_results

        if is_lol:
            all_batches_iter = chain(first_n, rest_iter)
            batches_list = list(all_batches_iter)
            total_items = sum(len(b) for b in batches_list) if progress_callback else 0
            completed = 0
            def _lol_progress(batch_len: int) -> None:
                nonlocal completed
                if progress_callback:
                    completed += batch_len
                    progress_callback(completed, total_items)
            if output == 'disk':
                path = file_path or f"{self.model_name}_{func}_output.jsonl"
                use_concurrent_lol_disk = self._concurrent_batches and not stop_on_error
                if use_concurrent_lol_disk:
                    progress_lock_lol = asyncio.Lock()
                    completed_ref_lol_disk: list = [0]

                    async def process_batch_lol_disk(batch_idx: int, batch: list):
                        br = await self._process_single_batch(func, batch, params, raw)
                        if progress_callback:
                            async with progress_lock_lol:
                                completed_ref_lol_disk[0] += len(batch)
                                progress_callback(completed_ref_lol_disk[0], total_items)
                        return (batch_idx, batch, br)

                    gathered = await asyncio.gather(*[process_batch_lol_disk(i, b) for i, b in enumerate(batches_list)])
                    gathered.sort(key=lambda x: x[0])
                    async with aiofiles.open(path, 'w', encoding='utf-8') as file_handle:
                        for _batch_idx, batch, batch_results in gathered:
                            if isinstance(batch_results, list):
                                if func != "generate":
                                    assert len(batch_results) == len(batch), (
                                        f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                        "This is a contract violation."
                                    )
                                for res in batch_results:
                                    await file_handle.write(json.dumps(res) + '\n')
                            else:
                                item_results = parse_validation_errors(batch_results, len(batch))
                                for res in item_results:
                                    await file_handle.write(json.dumps(res) + '\n')
                            await file_handle.flush()
                else:
                    async with aiofiles.open(path, 'w', encoding='utf-8') as file_handle:
                        for batch in batches_list:
                            batch_results = await self.call(func, batch, params=params, raw=raw)
                            if (
                                self.retry_error_batches and
                                isinstance(batch_results, dict) and
                                ('error' in batch_results or 'status_code' in batch_results)
                            ):
                                batch_results = await self._retry_batch_individually(func, batch, params, raw)

                            if isinstance(batch_results, list):
                                # For 'generate' actions, models may return multiple results per item
                                # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                                if func != "generate":
                                    assert len(batch_results) == len(batch), (
                                        f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                        "This is a contract violation."
                                    )
                                for res in batch_results:
                                    await file_handle.write(json.dumps(res) + '\n')
                            else:
                                # Parse validation errors to distribute to specific items
                                item_results = parse_validation_errors(batch_results, len(batch))
                                for res in item_results:
                                    await file_handle.write(json.dumps(res) + '\n')
                            await file_handle.flush()
                            _lol_progress(len(batch))
                            if stop_on_error and (
                                (isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results)) or
                                (isinstance(batch_results, list) and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results))
                            ):
                                break
                return
            else:
                use_concurrent_lol_mem = self._concurrent_batches and not stop_on_error
                if use_concurrent_lol_mem:
                    progress_lock = asyncio.Lock()
                    completed_ref_lol: list = [0]

                    async def process_batch_lol_mem(batch_idx: int, batch: list):
                        br = await self._process_single_batch(func, batch, params, raw)
                        if progress_callback:
                            async with progress_lock:
                                completed_ref_lol[0] += len(batch)
                                progress_callback(completed_ref_lol[0], total_items)
                        return (batch_idx, batch, br)

                    gathered = await asyncio.gather(*[process_batch_lol_mem(i, b) for i, b in enumerate(batches_list)])
                    gathered.sort(key=lambda x: x[0])
                    for _batch_idx, batch, batch_results in gathered:
                        if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                            item_results = parse_validation_errors(batch_results, len(batch))
                            results.extend(item_results)
                        elif isinstance(batch_results, list):
                            if func != "generate":
                                assert len(batch_results) == len(batch), (
                                    f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                    "This is a contract violation."
                                )
                            results.extend(batch_results)
                        else:
                            results.append(batch_results)
                    return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results
                for batch in batches_list:
                    batch_results = await self.call(func, batch, params=params, raw=raw)
                    if (
                        self.retry_error_batches and
                        isinstance(batch_results, dict) and
                        ('error' in batch_results or 'status_code' in batch_results)
                    ):
                        batch_results = await self._retry_batch_individually(func, batch, params, raw)
                    if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                        # Parse validation errors to distribute to specific items
                        item_results = parse_validation_errors(batch_results, len(batch))
                        results.extend(item_results)
                        _lol_progress(len(batch))
                        if stop_on_error:
                            break
                    elif isinstance(batch_results, list):
                        # For 'generate' actions, models may return multiple results per item
                        # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                        if func != "generate":
                            assert len(batch_results) == len(batch), (
                                f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                "This is a contract violation."
                            )
                        results.extend(batch_results)
                    else:
                        results.append(batch_results)
                    _lol_progress(len(batch))
                    if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                        break
                return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results

        all_items = chain(first_n, rest_iter)
        if progress_callback is not None:
            all_items = list(all_items)
            total_items = len(all_items)
        else:
            total_items = 0
        completed_flat = 0

        def _flat_progress(batch_len: int) -> None:
            nonlocal completed_flat
            if progress_callback:
                completed_flat += batch_len
                progress_callback(completed_flat, total_items)

        max_batch = await self._get_max_batch_size(self.model_name, func) or 1

        if output == 'disk':
            path = file_path or f"{self.model_name}_{func}_output.jsonl"
            use_concurrent_disk = self._concurrent_batches and not stop_on_error
            if use_concurrent_disk:
                batches = list(batch_iterable(all_items, max_batch))
                progress_lock_disk = asyncio.Lock()
                completed_flat_ref_disk: list = [0]

                async def process_batch_disk(batch_idx: int, batch: list):
                    br = await self._process_single_batch(func, batch, params, raw)
                    if progress_callback:
                        async with progress_lock_disk:
                            completed_flat_ref_disk[0] += len(batch)
                            progress_callback(completed_flat_ref_disk[0], total_items)
                    return (batch_idx, batch, br)

                gathered = await asyncio.gather(*[process_batch_disk(i, b) for i, b in enumerate(batches)])
                gathered.sort(key=lambda x: x[0])
                async with aiofiles.open(path, 'w', encoding='utf-8') as file_handle:
                    for _batch_idx, batch, batch_results in gathered:
                        if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                            item_results = parse_validation_errors(batch_results, len(batch))
                            for res in item_results:
                                to_dump = res[0] if (raw and isinstance(res, tuple)) else res
                                await file_handle.write(json.dumps(to_dump) + '\n')
                        else:
                            if not isinstance(batch_results, list):
                                batch_results = [batch_results]
                            if func != "generate":
                                assert len(batch_results) == len(batch), (
                                    f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                    "This is a contract violation."
                                )
                            for res in batch_results:
                                to_dump = res[0] if (raw and isinstance(res, tuple)) else res
                                await file_handle.write(json.dumps(to_dump) + '\n')
                        await file_handle.flush()
            else:
                async with aiofiles.open(path, 'w', encoding='utf-8') as file_handle:
                    for batch in batch_iterable(all_items, max_batch):
                        batch_results = await self.call(func, batch, params=params, raw=raw)

                        if (
                            self.retry_error_batches and
                            isinstance(batch_results, dict) and
                            ('error' in batch_results or 'status_code' in batch_results)
                        ):
                            batch_results = await self._retry_batch_individually(func, batch, params, raw)
                            # After retry, always treat as list
                            for res in batch_results:
                                to_dump = res[0] if (raw and isinstance(res, tuple)) else res
                                await file_handle.write(json.dumps(to_dump) + '\n')
                            await file_handle.flush()
                            if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                                break
                            continue  # move to next batch

                        if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                            for _ in batch:
                                await file_handle.write(json.dumps(batch_results) + '\n')
                            await file_handle.flush()
                            if stop_on_error:
                                break
                        else:
                            if not isinstance(batch_results, list):
                                batch_results = [batch_results]
                            # For 'generate' actions, models may return multiple results per item
                            # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                            if func != "generate":
                                assert len(batch_results) == len(batch), (
                                    f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                    "This is a contract violation."
                                )
                            for res in batch_results:
                                to_dump = res[0] if (raw and isinstance(res, tuple)) else res
                                await file_handle.write(json.dumps(to_dump) + '\n')
                            await file_handle.flush()
                            if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                                break
                            _flat_progress(len(batch))

            return
        else:
            use_concurrent = self._concurrent_batches and not stop_on_error
            if use_concurrent:
                batches = list(batch_iterable(all_items, max_batch))
                progress_lock_flat = asyncio.Lock()
                completed_flat_ref_mem: list = [0]

                async def process_batch(batch_idx: int, batch: list):
                    br = await self._process_single_batch(func, batch, params, raw)
                    if progress_callback:
                        async with progress_lock_flat:
                            completed_flat_ref_mem[0] += len(batch)
                            progress_callback(completed_flat_ref_mem[0], total_items)
                    return (batch_idx, batch, br)

                gathered = await asyncio.gather(*[process_batch(i, b) for i, b in enumerate(batches)])
                gathered.sort(key=lambda x: x[0])
                for _batch_idx, batch, batch_results in gathered:
                    if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                        item_results = parse_validation_errors(batch_results, len(batch))
                        results.extend(item_results)
                    elif isinstance(batch_results, list):
                        results.extend(batch_results)
                    else:
                        if not isinstance(batch_results, list):
                            batch_results = [batch_results]
                        if func != "generate":
                            assert len(batch_results) == len(batch), (
                                f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                "This is a contract violation."
                            )
                        results.extend(batch_results)
                return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results

            for batch_idx, batch in enumerate(batch_iterable(all_items, max_batch)):
                batch_results = await self.call(func, batch, params=params, raw=raw)

                if (
                    self.retry_error_batches and
                    isinstance(batch_results, dict) and
                    ('error' in batch_results or 'status_code' in batch_results)
                ):
                    batch_results = await self._retry_batch_individually(func, batch, params, raw)
                    results.extend(batch_results)
                    if stop_on_error and any(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                        break
                    continue  # move to next batch


                if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                    # Parse validation errors to distribute to specific items
                    item_results = parse_validation_errors(batch_results, len(batch))
                    results.extend(item_results)
                    if stop_on_error:
                        break
                elif isinstance(batch_results, list):
                    # Successful batch - results is already a list
                    results.extend(batch_results)
                else:
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                    # For 'generate' actions, models may return multiple results per item
                    # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                    if func != "generate":
                        assert len(batch_results) == len(batch), (
                            f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                            "This is a contract violation."
                        )
                    results.extend(batch_results)
                    if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                        break
                _flat_progress(len(batch))

            return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results

    @staticmethod
    def _format_result(res: Union[dict, List[dict], Tuple[dict, int]]) -> Union[dict, List[dict], Tuple[dict, int]]:
        if isinstance(res, dict) and 'results' in res:
            return res['results']
        elif isinstance(res, list):
            if all(isinstance(x, dict) for x in res):
                return res
            raise ValueError("Unexpected response format")
        elif isinstance(res, dict) and ('error' in res or 'status_code' in res):
            return res
        return res


    def _format_exception(self, exc: Exception, index: int) -> dict:
        """Format an exception as an error dict, with fallback for empty error messages."""
        error_msg = str(exc)
        
        # Handle empty or whitespace-only error messages
        if not error_msg or not error_msg.strip():
            # Check if it's a known httpx exception type
            exc_type = type(exc)
            if exc_type in HTTPX_EXCEPTION_MESSAGES:
                error_msg = HTTPX_EXCEPTION_MESSAGES[exc_type]
            else:
                # Fallback: use exception class name
                error_msg = f"{exc_type.__name__}: Network or connection error occurred"
        
        return {"error": error_msg, "index": index}

    @staticmethod
    def _unwrap_single(result):
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def generate(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        overwrite: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        return await self._batch_call_autoschema_or_manual(
            "generate", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite,
            progress_callback=progress_callback
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def predict(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        overwrite: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        return await self._batch_call_autoschema_or_manual(
            "predict", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite,
            progress_callback=progress_callback
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def encode(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        overwrite: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        return await self._batch_call_autoschema_or_manual(
            "encode", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite,
            progress_callback=progress_callback
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def search(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        overwrite: bool = True,
    ):
        return await self._batch_call_autoschema_or_manual(
            "search", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def score(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        overwrite: bool = True,
    ):
        return await self._batch_call_autoschema_or_manual(
            "score", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite
        )

    async def lookup(
        self,
        query: Union[dict, List[dict]],
        *,
        raw: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
    ):
        items = query if isinstance(query, list) else [query]
        res = await self.call("lookup", items, params=None, raw=raw)
        if raw:
            single = len(items) == 1
            if single:
                data, resp = res
                return LookupResult(data, resp)
            return [LookupResult(r[0], r[1]) for r in res]
        return res

    async def shutdown(self):
        await self._http_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

# Log execution context for debugging (only if DEBUG env var is set)
if os.environ.get("DEBUG", '').upper().strip() in ('TRUE', '1'):
    context = _detect_execution_context()
    debug(f"BioLMApi sync wrapper created in context: {context}")

# Synchronous wrapper for compatibility
# Note: nest_asyncio is automatically applied in Jupyter contexts
# to enable this wrapper to work with Jupyter's event loop
@_synchronizer.sync
class BioLMApi(BioLMApiClient):
    pass
