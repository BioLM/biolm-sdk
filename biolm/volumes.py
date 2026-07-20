"""Deprecated compatibility placeholder for BioLM runtime volumes."""
import warnings
from typing import Optional


_DEPRECATION_MESSAGE = (
    "Modal-backed volumes are runtime-managed and not a local SDK storage API; "
    "manage them via the BioLM console, Jupyter, or protocol export workflows."
)
_UNSUPPORTED_MESSAGE = (
    "Direct local volume management is unsupported. Modal-backed volumes are "
    "managed by BioLM runtimes; use the BioLM console, Jupyter, or protocol "
    "export workflows."
)


class Volume:
    """Deprecated placeholder retained for constructor and import compatibility.

    Args:
        name: Legacy volume name.
        api_key: Legacy API token argument.
    """

    def __init__(self, name: Optional[str] = None, api_key: Optional[str] = None):
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.name = name
        self._api_key = api_key

    def list(self):
        """Reject direct local volume listing."""
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)

    def create(self, name: str, **kwargs):
        """Reject direct local volume creation."""
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)

    def get(self, name: Optional[str] = None):
        """Reject direct local volume lookup."""
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)

    def delete(self, name: Optional[str] = None) -> bool:
        """Reject direct local volume deletion."""
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)
