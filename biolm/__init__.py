"""Top-level package for BioLM."""
__author__ = """Nikhil Haas"""
__email__ = "nikhil@biolm.ai"
__version__ = '1.2.0'

from biolm.core.http import BioLMApi, BioLMApiClient
from biolm.client import BioLM
from biolm.models import Model, predict, encode, generate
from biolm.protocols import Protocol
from biolm.finetune import Finetune
from biolm.protocol_runs import (
    ProtocolClient,
    ProtocolRun,
    ProtocolRunError,
    ProtocolNotFoundError,
)
from biolm.platform import (
    AmbiguousWorkspaceError,
    PlatformClient,
    PlatformError,
    Workspace,
    WorkspaceNotFoundError,
)
from biolm.volumes import Volume
from biolm.models.examples import get_example, list_models
from biolm.io import (
    load_fasta,
    to_fasta,
    load_csv,
    to_csv,
    load_pdb,
    to_pdb,
    load_json,
    to_json,
)

try:
    from biolm import pipeline
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

from typing import Optional, Union, List, Any

__all__ = [
    'BioLM',
    'biolm',
    'BioLMApi',
    'BioLMApiClient',
    'Model',
    'Protocol',
    'Finetune',
    'Workspace',
    'PlatformClient',
    'PlatformError',
    'WorkspaceNotFoundError',
    'AmbiguousWorkspaceError',
    'Volume',
    'ProtocolClient',
    'ProtocolRun',
    'ProtocolRunError',
    'ProtocolNotFoundError',
    'run_protocol',
    'predict',
    'encode',
    'generate',
    'get_example',
    'list_models',
    'load_fasta',
    'to_fasta',
    'load_csv',
    'to_csv',
    'load_pdb',
    'to_pdb',
    'load_json',
    'to_json',
]
if _HAS_PIPELINE:
    __all__.append('pipeline')


def run_protocol(
    slug: str,
    inputs: dict,
    *,
    run_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 3600.0,
    show_progress: bool = True,
    poll_interval: float = 5.0,
) -> dict:
    """Submit a BioLM protocol run and block until results are ready."""
    client = ProtocolClient(api_key=api_key, base_url=base_url)
    return client.run_and_wait(
        slug,
        inputs,
        run_name=run_name,
        timeout=timeout,
        show_progress=show_progress,
        poll_interval=poll_interval,
    )


def biolm(
    *,
    entity: str,
    action: str,
    type: Optional[str] = None,
    items: Union[Any, List[Any]],
    params: Optional[dict] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Any:
    """Call a BioLM model in one step (sync, blocking).

    Wraps :class:`biolm.client.BioLM` and returns the API result. Single-item
    calls return a dict; batch calls return a list.

    Args:
        entity: Model slug (e.g. ``"esm2-8m"``, ``"esmfold"``).
        action: Model action (e.g. ``"encode"``, ``"predict"``, ``"generate"``).
        type: Item type when ``items`` are plain strings (e.g. ``"sequence"``).
        items: One item, a list of items, or a generator of items.
        params: Optional action-specific parameters.
        api_key: Optional API token; defaults to ``BIOLM_TOKEN``.
        **kwargs: Passed through to :class:`biolm.client.BioLM`.

    Returns:
        A single result dict or a list of result dicts (one per input item).
    """
    return BioLM(
        entity=entity,
        action=action,
        type=type,
        items=items,
        params=params,
        api_key=api_key,
        **kwargs
    )
