"""Legacy modules - deprecated but kept for backward compatibility."""

import warnings

warnings.warn(
    "The biolmai.core.legacy module is deprecated. "
    "Please update your imports to use the new module structure.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []
