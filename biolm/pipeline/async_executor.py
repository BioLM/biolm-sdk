"""
Async batch execution utilities with semaphore-based rate limiting.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, Callable, Optional, TypeVar

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

T = TypeVar("T")
R = TypeVar("R")


class AsyncBatchExecutor:
    """
    Execute async tasks in batches with rate limiting.

    Features:
    - Semaphore-based concurrency control
    - Progress tracking with tqdm
    - Error handling per item
    - Dynamic batch sizing

    Args:
        max_concurrent: Maximum number of concurrent tasks
        batch_size: Size of each batch (None for no batching)
        progress_desc: Description for progress bar
        show_progress: Whether to show progress bar

    Example:
        >>> executor = AsyncBatchExecutor(max_concurrent=10)
        >>> results = await executor.execute(items, process_func)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        batch_size: Optional[int] = None,
        progress_desc: str = "Processing",
        show_progress: bool = True,
    ):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.progress_desc = progress_desc
        self.show_progress = show_progress
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _execute_with_semaphore(
        self, func: Callable[[T], Coroutine[Any, Any, R]], item: T
    ) -> R:
        """Execute function with semaphore control."""
        async with self.semaphore:
            return await func(item)

    async def execute(
        self,
        items: list[T],
        func: Callable[[T], Coroutine[Any, Any, R]],
        return_exceptions: bool = False,
    ) -> list[R]:
        """
        Execute async function on all items with rate limiting.

        Args:
            items: List of items to process
            func: Async function to apply to each item
            return_exceptions: If True, return exceptions instead of raising

        Returns:
            List of results (same order as input)
        """
        if not items:
            return []

        tasks = [self._execute_with_semaphore(func, item) for item in items]

        if self.show_progress:
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=self.progress_desc,
                total=len(tasks),
                return_exceptions=return_exceptions,
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        return results

    async def execute_batched(
        self,
        items: list[T],
        batch_func: Callable[[list[T]], Coroutine[Any, Any, list[R]]],
        batch_size: Optional[int] = None,
        return_exceptions: bool = False,
    ) -> list[R]:
        """
        Execute async function on batches of items.

        Args:
            items: List of items to process
            batch_func: Async function that processes a batch and returns results
            batch_size: Size of each batch (uses self.batch_size if None)
            return_exceptions: If True, return exceptions instead of raising

        Returns:
            Flattened list of results
        """
        batch_size = batch_size or self.batch_size
        if batch_size is None:
            raise ValueError("batch_size must be specified")

        # Create batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        async def process_batch_with_semaphore(batch: list[T]) -> list[R]:
            async with self.semaphore:
                return await batch_func(batch)

        tasks = [process_batch_with_semaphore(batch) for batch in batches]

        if self.show_progress:
            batch_results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"{self.progress_desc} (batches)",
                total=len(batches),
                return_exceptions=return_exceptions,
            )
        else:
            batch_results = await asyncio.gather(
                *tasks, return_exceptions=return_exceptions
            )

        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception) and return_exceptions:
                results.append(batch_result)
            elif isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)

        return results


class CachingExecutor:
    """
    Executor with built-in caching support.

    Checks cache before execution and updates cache after completion.

    Args:
        executor: AsyncBatchExecutor instance
        cache_check_func: Function to check if result is cached (returns cached result or None)
        cache_store_func: Function to store result in cache

    Example:
        >>> def cache_check(item):
        ...     return datastore.get_prediction(item['sequence_id'], 'stability')
        >>>
        >>> def cache_store(item, result):
        ...     datastore.add_prediction(item['sequence_id'], 'stability', result)
        >>>
        >>> executor = CachingExecutor(
        ...     AsyncBatchExecutor(max_concurrent=10),
        ...     cache_check, cache_store
        ... )
        >>> results = await executor.execute_with_cache(items, process_func)
    """

    def __init__(
        self,
        executor: AsyncBatchExecutor,
        cache_check_func: Optional[Callable[[T], Optional[R]]] = None,
        cache_store_func: Optional[Callable[[T, R], None]] = None,
    ):
        self.executor = executor
        self.cache_check_func = cache_check_func
        self.cache_store_func = cache_store_func

    async def execute_with_cache(
        self,
        items: list[T],
        func: Callable[[T], Coroutine[Any, Any, R]],
        return_exceptions: bool = False,
    ) -> tuple[list[R], int, int]:
        """
        Execute with caching.

        Returns:
            Tuple of (results, cached_count, computed_count)
        """
        # Separate cached and uncached items
        uncached_items = []
        uncached_indices = []
        results = [None] * len(items)
        cached_count = 0

        for i, item in enumerate(items):
            if self.cache_check_func:
                cached_result = self.cache_check_func(item)
                if cached_result is not None:
                    results[i] = cached_result
                    cached_count += 1
                    continue

            uncached_items.append(item)
            uncached_indices.append(i)

        # Process uncached items
        if uncached_items:
            computed_results = await self.executor.execute(
                uncached_items, func, return_exceptions=return_exceptions
            )

            # Store computed results in cache and results list
            for item, result, idx in zip(
                uncached_items, computed_results, uncached_indices
            ):
                results[idx] = result

                if self.cache_store_func and not isinstance(result, Exception):
                    self.cache_store_func(item, result)

        computed_count = len(uncached_items)

        return results, cached_count, computed_count


class StreamingExecutor:
    """
    Executor that streams results as they complete.

    Useful for pipelines where downstream stages can start before all results are ready.

    Args:
        executor: AsyncBatchExecutor instance
        result_callback: Optional callback for each completed result

    Example:
        >>> async def on_result(item, result):
        ...     print(f"Completed: {item}")
        ...     downstream_queue.put(result)
        >>>
        >>> executor = StreamingExecutor(
        ...     AsyncBatchExecutor(max_concurrent=10),
        ...     result_callback=on_result
        ... )
        >>> await executor.execute_streaming(items, process_func)
    """

    def __init__(
        self,
        executor: AsyncBatchExecutor,
        result_callback: Optional[Callable[[T, R], Coroutine[Any, Any, None]]] = None,
    ):
        self.executor = executor
        self.result_callback = result_callback

    async def execute_streaming(
        self,
        items: list[T],
        func: Callable[[T], Coroutine[Any, Any, R]],
        return_exceptions: bool = False,
    ) -> list[R]:
        """
        Execute with streaming results.

        Results are passed to callback as they complete.

        Returns:
            List of results (order may differ from input)
        """

        async def wrapped_func(item: T) -> R:
            result = await func(item)

            if self.result_callback:
                await self.result_callback(item, result)

            return result

        return await self.executor.execute(
            items, wrapped_func, return_exceptions=return_exceptions
        )


# Helper functions for DataFrame-based processing


async def process_dataframe_async(
    df: pd.DataFrame,
    process_func: Callable[[pd.Series], Coroutine[Any, Any, Any]],
    result_column: str,
    max_concurrent: int = 10,
    show_progress: bool = True,
    progress_desc: str = "Processing",
) -> pd.DataFrame:
    """
    Process DataFrame rows asynchronously and add results as a new column.

    Args:
        df: Input DataFrame
        process_func: Async function to process each row (receives a Series)
        result_column: Name of column to store results
        max_concurrent: Maximum concurrent tasks
        show_progress: Whether to show progress bar
        progress_desc: Description for progress bar

    Returns:
        DataFrame with new result column

    Example:
        >>> async def predict_stability(row):
        ...     return await model.predict(row['sequence'])
        >>>
        >>> df = await process_dataframe_async(
        ...     df, predict_stability, 'stability_score'
        ... )
    """
    executor = AsyncBatchExecutor(
        max_concurrent=max_concurrent,
        progress_desc=progress_desc,
        show_progress=show_progress,
    )

    rows = [row for _, row in df.iterrows()]
    results = await executor.execute(rows, process_func)

    df = df.copy()
    df[result_column] = results

    return df


async def process_sequences_batched(
    sequences: list[str],
    batch_func: Callable[[list[str]], Coroutine[Any, Any, list[Any]]],
    batch_size: int = 32,
    max_concurrent: int = 5,
    show_progress: bool = True,
    progress_desc: str = "Processing sequences",
) -> list[Any]:
    """
    Process sequences in batches.

    Args:
        sequences: List of sequences
        batch_func: Async function that processes a batch of sequences
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent batches
        show_progress: Whether to show progress bar
        progress_desc: Description for progress bar

    Returns:
        List of results (one per sequence)

    Example:
        >>> async def predict_batch(seqs):
        ...     return await model.predict_batch(seqs)
        >>>
        >>> results = await process_sequences_batched(
        ...     sequences, predict_batch, batch_size=32
        ... )
    """
    executor = AsyncBatchExecutor(
        max_concurrent=max_concurrent,
        batch_size=batch_size,
        progress_desc=progress_desc,
        show_progress=show_progress,
    )

    return await executor.execute_batched(sequences, batch_func)
