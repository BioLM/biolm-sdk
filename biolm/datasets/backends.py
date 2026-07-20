"""Push/pull backend registry for datasets."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, runtime_checkable

from biolm.datasets.errors import BackendNotAvailableError

if TYPE_CHECKING:
    from biolm.datasets.dataset import Dataset

_BACKENDS: Dict[str, "DatasetPushPullBackend"] = {}
_LAZY_LOADERS: Dict[str, Callable[[], "DatasetPushPullBackend"]] = {}


@runtime_checkable
class DatasetPushPullBackend(Protocol):
    """Adapter that can push/pull a local dataset to a remote store."""

    name: str

    def push(self, dataset: "Dataset", **opts: Any) -> Dict[str, Any]:
        """Upload local dataset artifacts and metadata."""

    def pull(self, dataset_id: str, dest: Path, **opts: Any) -> Dict[str, Any]:
        """Download remote dataset into *dest*; return status info."""


def register_backend(name: str, backend: DatasetPushPullBackend) -> None:
    """Register a push/pull backend under *name*."""
    _BACKENDS[name] = backend


def register_lazy_backend(name: str, loader: Callable[[], DatasetPushPullBackend]) -> None:
    """Register a factory that imports/builds a backend on first use."""
    _LAZY_LOADERS[name] = loader


def get_backend(name: str) -> DatasetPushPullBackend:
    """Return a registered backend, lazy-loading known plugins when needed."""
    if name in _BACKENDS:
        return _BACKENDS[name]

    if name in _LAZY_LOADERS:
        backend = _LAZY_LOADERS[name]()
        register_backend(name, backend)
        return backend

    if name == "mlflow":
        try:
            from biolm.plugins.mlflow.dataset_backend import MLflowDatasetBackend

            backend = MLflowDatasetBackend()
            register_backend("mlflow", backend)
            return backend
        except ImportError as exc:
            raise BackendNotAvailableError(
                "MLflow dataset backend is not available. "
                "Install with: pip install 'biolm-sdk[mlflow]'"
            ) from exc

    raise BackendNotAvailableError(
        f"Unknown dataset backend '{name}'. "
        "Known backends are registered via plugins (e.g. 'mlflow')."
    )


def clear_backends() -> None:
    """Clear registered backends (for tests)."""
    _BACKENDS.clear()
