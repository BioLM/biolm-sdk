"""Local dataset exceptions."""
from __future__ import annotations


class DatasetError(Exception):
    """Base error for local dataset operations."""


class DatasetNotFoundError(DatasetError):
    """Dataset id or path could not be resolved."""


class DatasetExistsError(DatasetError):
    """dataset.yaml already exists and force was not set."""


class DuplicateDatasetIdError(DatasetError):
    """The same dataset id was found under multiple discovery roots."""


class DatasetSchemaError(DatasetError):
    """dataset.yaml is missing required fields or is invalid."""


class BackendNotAvailableError(DatasetError):
    """Requested push/pull backend is not installed or not registered."""
