"""Utility functions for BioLM SDK."""
from itertools import chain, islice
from typing import Any, List, Optional, Tuple, Union


def is_list_of_lists(items, check_n=10):
    # Accepts any iterable, checks first N items for list/tuple-ness
    # Returns (is_list_of_lists, first_n_items, rest_iter)
    if isinstance(items, (list, tuple)):
        if not items:
            return False, [], iter(())
        first_n = items[:check_n]
        is_lol = all(isinstance(x, (list, tuple)) for x in first_n)
        return is_lol, first_n, iter(items[check_n:])
    # For iterators/generators: consume first N, return rest (no tee - tee would
    # duplicate first N when caller chains first_n + rest)
    first_n = list(islice(items, check_n))
    is_lol = all(isinstance(x, (list, tuple)) for x in first_n) and bool(first_n)
    return is_lol, first_n, items


def prepare_items_for_api(
    items: Union[Any, List[Any]],
    type: Optional[str] = None,
    check_n: int = 10,
) -> Tuple[Union[List[dict], List[List[dict]], Any], bool]:
    """
    Normalize items for API calls. Supports list, tuple, single value, and iterables (e.g. generators).
    Returns (data, is_lol) where is_lol is True for list-of-lists; data is iterable of dicts or list of lists.
    """
    if isinstance(items, list):
        normalized = items
    elif isinstance(items, tuple):
        normalized = items
    elif isinstance(items, (str, bytes, dict)) or not hasattr(items, "__iter__"):
        normalized = [items]
    else:
        normalized = items  # generator, iterator, range, etc.

    is_lol, first_n, rest_iter = is_list_of_lists(normalized, check_n=check_n)
    is_reiterable = isinstance(normalized, (list, tuple))

    if is_lol:
        for batch in first_n:
            if not all(isinstance(x, dict) for x in batch):
                raise ValueError("All items in each batch must be dicts when passing a list of lists.")
        if type is not None:
            raise ValueError("Do not specify `type` when passing a list of lists of dicts for `items`.")
        return (list(first_n) + list(rest_iter), True)
    elif is_reiterable and all(isinstance(v, dict) for v in normalized):
        return (normalized, False)
    elif not is_reiterable and first_n and all(isinstance(v, dict) for v in first_n):
        return (chain(first_n, rest_iter), False)
    else:
        if type is None:
            raise ValueError("If `items` are not dicts, `type` must be specified.")
        if is_reiterable:
            return ([{type: v} for v in normalized], False)
        return (({type: v} for v in chain(first_n, rest_iter)), False)


def batch_iterable(iterable, batch_size):
    # Yields lists of up to batch_size from any iterable, deleting as we go
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
