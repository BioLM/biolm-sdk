"""sf.models — model inference enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from biolm.seqframe.enrichment import ApiEnrichmentBackend, EnrichmentBackend

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame


class ModelsNamespace:
    def __init__(self, sf: "SeqFrame", backend: Optional[EnrichmentBackend] = None):
        self._sf = sf
        self._backend = backend or ApiEnrichmentBackend()

    def predict(
        self,
        model: str,
        *,
        column: Optional[str] = None,
        batch_size: int = 32,
        action: str = "predict",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "SeqFrame":
        col = column or f"{model.replace('-', '_')}_{action}"
        return self._backend.predict(
            self._sf,
            model=model,
            column=col,
            batch_size=batch_size,
            action=action,
            api_key=api_key,
            **kwargs,
        )

    def embed(
        self,
        model: str,
        *,
        column: Optional[str] = None,
        layer: Optional[int] = None,
        batch_size: int = 32,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "SeqFrame":
        col = column or f"{model.replace('-', '_')}_embedding"
        return self._backend.embed(
            self._sf,
            model=model,
            column=col,
            layer=layer,
            batch_size=batch_size,
            api_key=api_key,
            **kwargs,
        )
