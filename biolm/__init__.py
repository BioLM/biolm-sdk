"""Top-level package for BioLM."""
__author__ = """Nikhil Haas"""
__email__ = "nikhil@biolm.ai"
__version__ = '1.0.0'

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
from biolm.workspaces import Workspace
from biolm.volumes import Volume
from biolm.examples import get_example, list_models
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
) -> dict:
    """Submit a BioLM protocol run and block until results are ready."""
    client = ProtocolClient(api_key=api_key, base_url=base_url)
    return client.run_and_wait(
        slug,
        inputs,
        run_name=run_name,
        timeout=timeout,
        show_progress=show_progress,
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
    """Top-level convenience function that wraps the BioLM class and returns the result."""
    return BioLM(
        entity=entity,
        action=action,
        type=type,
        items=items,
        params=params,
        api_key=api_key,
        **kwargs
    )
