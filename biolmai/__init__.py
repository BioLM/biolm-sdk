"""Deprecated compatibility shim — use `import biolm` instead."""
import warnings

warnings.warn(
    "The biolmai package has been renamed to biolm. "
    "Please use `import biolm` and `pip install biolm-sdk`. "
    "Support for `import biolmai` will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from biolm import *  # noqa: F403

import biolm as _biolm

__version__ = _biolm.__version__
__all__ = _biolm.__all__
