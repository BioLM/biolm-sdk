"""sf.protocols — protocol run integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from biolm.protocol_runs import ProtocolClient
from biolm.seqframe.importers import from_protocol

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame


class ProtocolsNamespace:
    def __init__(self, sf: "SeqFrame"):
        self._sf = sf

    def run(
        self,
        slug: str,
        inputs: Dict[str, Any],
        *,
        join_on: str = "id",
        version: Optional[int] = None,
        run_name: Optional[str] = None,
        timeout: float = 3600.0,
        show_progress: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        output_dir: str = ".",
    ) -> "SeqFrame":
        """Submit a protocol run and merge results back into this SeqFrame."""
        client = ProtocolClient(api_key=api_key, base_url=base_url)
        run = client.submit(slug, inputs, version=version, run_name=run_name)
        run.wait(timeout=timeout, show_progress=show_progress)
        result_sf = from_protocol(run, output_dir=output_dir)
        return self._sf.query.join(result_sf, on=join_on, how="left")
