"""Enrichment backends for SeqFrame model operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

import pandas as pd

from biolm.core.utils import batch_iterable

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame


class EnrichmentBackend(Protocol):
    def predict(
        self,
        sf: "SeqFrame",
        *,
        model: str,
        column: str,
        batch_size: int = 32,
        action: str = "predict",
        api_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SeqFrame": ...

    def embed(
        self,
        sf: "SeqFrame",
        *,
        model: str,
        column: str,
        layer: Optional[int] = None,
        batch_size: int = 32,
        api_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SeqFrame": ...


def _extract_scalar(result: Dict[str, Any], action: str) -> Any:
    if not isinstance(result, dict):
        return result
    for key in ("value", "score", "prediction", "log_prob", "embedding"):
        if key in result:
            return result[key]
    if action == "encode" and "embeddings" in result:
        return result["embeddings"]
    if len(result) == 1:
        return next(iter(result.values()))
    return result


_BIOLM_API_CTOR_KEYS = frozenset(
    {
        "base_url",
        "timeout",
        "raise_httpx",
        "unwrap_single",
        "semaphore",
        "rate_limit",
        "retry_error_batches",
        "compress_requests",
        "compress_threshold",
        "concurrent_batches",
        "http2",
    }
)


class ApiEnrichmentBackend:
    """v0 enrichment via sync BioLM API calls."""

    def predict(
        self,
        sf: "SeqFrame",
        *,
        model: str,
        column: str,
        batch_size: int = 32,
        action: str = "predict",
        api_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SeqFrame":
        from biolm.core.http import BioLMApi

        df = sf.collect()
        id_col = sf.schema.id_column
        seq_col = sf.schema.sequence_column
        values: List[Any] = []
        ids: List[str] = []

        ctor_kwargs = {k: v for k, v in kwargs.items() if k in _BIOLM_API_CTOR_KEYS}
        unknown = set(kwargs) - _BIOLM_API_CTOR_KEYS
        if unknown:
            raise TypeError(
                f"Unexpected keyword arguments for models.predict/embed: {sorted(unknown)}. "
                f"Pass model call options via params=; BioLMApi ctor options: "
                f"{sorted(_BIOLM_API_CTOR_KEYS)}."
            )

        api = BioLMApi(model, api_key=api_key, **ctor_kwargs)
        method = getattr(api, action, None)
        if method is None:
            raise ValueError(f"Unsupported BioLM action: {action!r}")
        try:
            for batch in batch_iterable(df.to_dict(orient="records"), batch_size):
                items = [{seq_col: row[seq_col]} for row in batch]
                call_kwargs: Dict[str, Any] = {"items": items}
                if params is not None:
                    call_kwargs["params"] = params
                results = method(**call_kwargs)
                if not isinstance(results, list):
                    results = [results]
                for row, result in zip(batch, results):
                    ids.append(str(row[id_col]))
                    values.append(_extract_scalar(result, action))
        finally:
            api.shutdown()

        out = pd.DataFrame({id_col: ids, column: values})
        return sf.merge_columns(out, on=id_col)

    def embed(
        self,
        sf: "SeqFrame",
        *,
        model: str,
        column: str,
        layer: Optional[int] = None,
        batch_size: int = 32,
        api_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SeqFrame":
        call_params = dict(params or {})
        if layer is not None:
            call_params["layer"] = layer
        return self.predict(
            sf,
            model=model,
            column=column,
            batch_size=batch_size,
            action="encode",
            api_key=api_key,
            params=call_params or None,
            **kwargs,
        )


class PipelineEnrichmentBackend:
    """Placeholder for future pipeline-backed enrichment with DuckDB caching."""

    def predict(self, sf: "SeqFrame", **kwargs: Any) -> "SeqFrame":
        raise NotImplementedError(
            "Pipeline enrichment backend is not yet implemented. "
            "Use the default API backend or ApiEnrichmentBackend."
        )

    def embed(self, sf: "SeqFrame", **kwargs: Any) -> "SeqFrame":
        raise NotImplementedError(
            "Pipeline enrichment backend is not yet implemented. "
            "Use the default API backend or ApiEnrichmentBackend."
        )
