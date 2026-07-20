"""Workspace management for BioLM.

The workspace identity is an account + environment pair. Prefer
:class:`biolm.platform.PlatformClient` for listing, switching, and creating
workspaces. This module re-exports the public types for compatibility.
"""
from biolm.platform import (
    AmbiguousWorkspaceError,
    PlatformClient,
    PlatformError,
    Workspace,
    WorkspaceNotFoundError,
)

__all__ = [
    "AmbiguousWorkspaceError",
    "PlatformClient",
    "PlatformError",
    "Workspace",
    "WorkspaceNotFoundError",
]
