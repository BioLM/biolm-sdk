"""BioLM convenience client and HTTP client re-exports."""
import logging

from typing import Optional, Union, List, Any

from biolm.core.http import BioLMApi, BioLMApiClient, CredentialsProvider, AsyncRateLimiter, parse_rate_limit
from biolm.core.utils import prepare_items_for_api, batch_iterable, is_list_of_lists

log = logging.getLogger("biolm_util")

__all__ = [
    "BioLM",
    "BioLMApi",
    "BioLMApiClient",
    "CredentialsProvider",
    "AsyncRateLimiter",
    "batch_iterable",
    "is_list_of_lists",
    "parse_rate_limit",
]


class BioLM:
    """
    Universal client for BioLM API.

    This is a convenience wrapper that creates a client, makes the request, and returns the result.
    For long-running operations or when making multiple requests, consider using `BioLMApiClient`
    (async) or `BioLMApi` (sync) directly with proper cleanup via context managers or shutdown().

    Args:
        entity (str): The entity name (model, database, calculation, etc).
        action (str): The action to perform (e.g., 'generate', 'encode', 'predict', 'search', 'finetune').
        type (str): The type of item (e.g., 'sequence', 'pdb', 'fasta_str').
        item (Union[Any, List[Any]]): The item(s) to process.
        params (Optional[dict]): Optional parameters for the action.
        raise_httpx (bool): Whether to raise HTTPX errors.
        stop_on_error (bool): Stop on first error if True.
        output (str): 'memory' or 'disk'.
        file_path (Optional[str]): Output file path if output='disk'.
        api_key (Optional[str]): API key for authentication.
        compress_requests (bool): Enable gzip compression for POST requests. Default: True.
        compress_threshold (int): Minimum payload size in bytes to trigger compression. Default: 256.
    """

    def __new__(
        cls,
        *,
        entity: str,
        action: str,
        type: Optional[str] = None,
        items: Union[Any, List[Any]],
        params: Optional[dict] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        self = super().__new__(cls)
        self.entity = entity
        self.action = action
        self.type = type
        self.items = items
        self.params = params
        self.api_key = api_key
        self._class_kwargs = kwargs
        return self.run()

    def run(self) -> Any:
        """Run the specified action on the entity with the given item(s)."""
        items_dicts, _ = prepare_items_for_api(self.items, type=self.type)

        unwrap_single = self._class_kwargs.pop('unwrap_single', True)

        action_kwargs = {k: v for k, v in dict(
            stop_on_error=self._class_kwargs.pop('stop_on_error', None),
            output=self._class_kwargs.pop('output', None),
            file_path=self._class_kwargs.pop('file_path', None),
            overwrite=self._class_kwargs.pop('overwrite', None),
        ).items() if v is not None}

        model = BioLMApi(
            self.entity,
            api_key=self.api_key,
            unwrap_single=unwrap_single,
            **self._class_kwargs,
        )

        action_map = {
            'generate': model.generate,
            'predict': model.predict,
            'encode': model.encode,
            'search': model.search,
            'score': model.score,
            'finetune': getattr(model, 'finetune', None),
            'lookup': model.lookup,
        }
        if self.action not in action_map or action_map[self.action] is None:
            raise ValueError(
                f"Action '{self.action}' is not amongst the available actions "
                f"{', '.join(action_map.keys())}."
            )

        method = action_map[self.action]
        kwargs = {
            'items': items_dicts,
            'params': self.params,
        }
        kwargs.update(action_kwargs)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        try:
            result = method(**kwargs)
        finally:
            try:
                if hasattr(model, 'shutdown'):
                    model.shutdown()
            except Exception:
                pass

        return result
