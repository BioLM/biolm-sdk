"""Import hook that aliases ``biolmai.*`` submodules to ``biolm.*``."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "biolmai."
_BIOLM_PREFIX = "biolm"
_INSTALLED = False


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, biolmai_name: str, biolm_name: str) -> None:
        self._biolmai_name = biolmai_name
        self._biolm_name = biolm_name

    def create_module(self, spec):  # noqa: ANN001
        return None

    def exec_module(self, module) -> None:  # noqa: ANN001
        biolm_mod = importlib.import_module(self._biolm_name)
        sys.modules[self._biolmai_name] = biolm_mod
        module.__dict__.update(biolm_mod.__dict__)


class _BiolmaiAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname == "biolmai" or not fullname.startswith(_PREFIX):
            return None

        biolm_name = _BIOLM_PREFIX + fullname[len("biolmai") :]
        try:
            biolm_spec = importlib.util.find_spec(biolm_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            return None
        if biolm_spec is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(fullname, biolm_name),
            is_package=bool(biolm_spec.submodule_search_locations),
        )


def install_shim() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    sys.meta_path.insert(0, _BiolmaiAliasFinder())
    _INSTALLED = True
