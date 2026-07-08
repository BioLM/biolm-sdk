"""
Data-driven pipeline implementations.

DataPipeline: Load sequences from files/lists and run predictions
SingleStepPipeline: Simplified single-step prediction pipeline
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from biolm.client import BioLMApiClient  # Use async client directly
from biolm.pipeline.base import BasePipeline, InputSchema, Stage, StageResult, WorkingSet
from biolm.pipeline.datastore_duckdb import DuckDBDataStore as DataStore
from biolm.pipeline.filters import BaseFilter


@dataclass
class ExtractionSpec:
    """Specification for extracting a value from an API response (with reduction).

    Use when you need to apply a reduction (mean, max, min, sum) to an
    array-valued response key. For simple scalar extractions, pass a plain
    string to ``extractions`` instead.

    Args:
        response_key: Key in API response dict, e.g. "plddt"
        reduction: Optional reduction for array values: "mean", "max", "min", "sum"
    """

    response_key: str
    reduction: Optional[str] = None


@dataclass
class _ResolvedExtraction:
    """Internal: fully resolved extraction with output column name.

    Created by PredictionStage.__init__ from the user-facing ``extractions``
    and ``columns`` parameters.

    Attributes:
        response_key: Key to read from the API response dict.
        column: Output column name (DuckDB ``prediction_type`` + DataFrame column).
        reduction: Optional reduction for array values.
    """

    response_key: str
    column: str
    reduction: Optional[str] = None


class PipelineAPIAuthError(RuntimeError):
    """Raised when the BioLM API returns 401/402 (auth/billing) during a stage.

    These errors are not retriable per-item — every batch will hit the same
    failure — so the pipeline fails fast instead of producing a silent empty
    result.  ``skip_on_error=True`` does NOT swallow this; callers always see
    an unambiguous failure with the upstream error payload.
    """

    def __init__(self, status_code: int, payload: Any, model_name: str):
        self.status_code = status_code
        self.payload = payload
        self.model_name = model_name
        # Surface only the customer-meaningful fields in the message text
        # (full payload is preserved on `self.payload` for programmatic access).
        if isinstance(payload, dict):
            short = {k: payload[k] for k in ("error", "code") if k in payload}
        else:
            short = payload
        msg = (
            f"BioLM API returned {status_code} for model {model_name!r}. "
            f"Pipeline is failing fast because retrying per-item would amplify "
            f"this auth/billing error.  Upstream: {short!r}"
        )
        super().__init__(msg)


_FATAL_API_STATUS_CODES = frozenset({401, 402, 403})


def _maybe_raise_fatal_api_error(result: Any, model_name: str) -> None:
    """Raise PipelineAPIAuthError if *result* is a BioLM error dict with a
    fatal-class status code (auth/billing/forbidden).  Per-item-level retry
    won't help — short-circuit to a typed exception."""
    if not isinstance(result, dict):
        return
    sc = result.get("status_code")
    if sc in _FATAL_API_STATUS_CODES:
        raise PipelineAPIAuthError(int(sc), result, model_name)


# SQL keyword denylist for free-text WHERE clauses.  This is defense-in-depth
# only — DO NOT pass untrusted user input to query_results / get_merged_results.
# DuckDB has powerful built-ins (ATTACH, COPY, read_csv_auto, read_parquet,
# read_json_auto, glob, PRAGMA, INSTALL, LOAD) that can read arbitrary files
# off disk or attach external databases; we block them here as a safety net,
# but the only actually safe input is one constructed by the SDK consumer.
_SQL_WHERE_DENY_KEYWORDS = (
    # DDL/DML
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "EXEC",
    # DuckDB extensions / file/network access
    "ATTACH", "DETACH", "COPY", "PRAGMA", "INSTALL", "LOAD", "EXPORT", "IMPORT",
    # Functions that touch the filesystem or remote URLs
    "READ_CSV", "READ_CSV_AUTO", "READ_JSON", "READ_JSON_AUTO", "READ_PARQUET",
    "READ_TEXT", "READ_BLOB", "GLOB", "PARQUET_SCAN", "ICEBERG_SCAN", "DELTA_SCAN",
    # Data exfiltration / cross-origin
    "UNION", "INTERSECT", "EXCEPT",
)


def _validate_sql_where(expr: str, param_name: str) -> None:
    """Defense-in-depth check for free-text WHERE clauses passed to DuckDB.

    This is **not** a sandbox — DuckDB exposes too many file-system entrypoints
    for a keyword blocklist to be airtight.  Use only with input you control.
    Untrusted input must be parameterized at the call site (DuckDB ``?``
    placeholders) or rejected before reaching this layer.
    """
    if ";" in expr:
        raise ValueError(f"{param_name} must not contain semicolons")
    if "--" in expr or "/*" in expr or "*/" in expr:
        raise ValueError(f"{param_name} must not contain SQL comment markers")
    upper = expr.upper()
    # Substring match (not token-split) so `INSERT/**/INTO`, `IN/*x*/SERT`,
    # `read_csv_auto(...)`, `ATTACH '...'` are all caught.
    for kw in _SQL_WHERE_DENY_KEYWORDS:
        if kw in upper:
            raise ValueError(
                f"{param_name} must not contain SQL keyword {kw!r}.  "
                f"This API accepts only column-level boolean expressions; "
                f"pass values via parameter substitution instead of inline SQL."
            )


def _resolve_extractions(
    extractions: Union[str, list[Union[str, ExtractionSpec]]],
    columns: Optional[Union[str, dict[str, str]]] = None,
) -> list[_ResolvedExtraction]:
    """Normalize user-facing extractions + columns into _ResolvedExtraction list.

    Examples::

        _resolve_extractions("prediction", "tm")
        → [_ResolvedExtraction("prediction", "tm")]

        _resolve_extractions("prediction")
        → [_ResolvedExtraction("prediction", "prediction")]

        _resolve_extractions(["mean_plddt", "ptm"], {"mean_plddt": "plddt"})
        → [_ResolvedExtraction("mean_plddt", "plddt"), _ResolvedExtraction("ptm", "ptm")]

        _resolve_extractions([ExtractionSpec("plddt", reduction="mean")], {"plddt": "plddt_mean"})
        → [_ResolvedExtraction("plddt", "plddt_mean", "mean")]
    """
    # Normalize extractions to list of (response_key, reduction)
    specs: list[tuple[str, Optional[str]]] = []
    if isinstance(extractions, str):
        specs = [(extractions, None)]
    elif isinstance(extractions, list):
        for item in extractions:
            if isinstance(item, str):
                specs.append((item, None))
            elif isinstance(item, ExtractionSpec):
                specs.append((item.response_key, item.reduction))
            else:
                raise TypeError(
                    f"extractions list items must be str or ExtractionSpec, got {type(item)}"
                )
    else:
        raise TypeError(
            f"extractions must be str or list, got {type(extractions)}"
        )

    # Build column mapping
    col_map: dict[str, str] = {}
    if columns is None:
        pass  # identity: response_key = column name
    elif isinstance(columns, str):
        if len(specs) != 1:
            raise ValueError(
                f"columns='{columns}' (str) can only be used with a single extraction, "
                f"got {len(specs)}. Use a dict for multiple extractions."
            )
        col_map[specs[0][0]] = columns
    elif isinstance(columns, dict):
        col_map = dict(columns)
    else:
        raise TypeError(f"columns must be str, dict, or None, got {type(columns)}")

    # validate that all keys in columns dict appear in extractions.
    if isinstance(columns, dict):
        extraction_keys = {key for key, _ in specs}
        unrecognized = [k for k in columns if k not in extraction_keys]
        if unrecognized:
            raise ValueError(
                f"columns dict contains key(s) {unrecognized} that do not appear in "
                f"extractions. Valid keys are: {sorted(extraction_keys)}. "
                "Check for typos in column key names."
            )

    # Resolve
    resolved = []
    for response_key, reduction in specs:
        column = col_map.get(response_key, response_key)
        resolved.append(_ResolvedExtraction(response_key, column, reduction))

    # check for duplicate response_key values.
    seen_keys: dict[str, int] = {}
    for i, spec in enumerate(resolved):
        if spec.response_key in seen_keys:
            raise ValueError(
                f"Duplicate response_key '{spec.response_key}' in extractions at "
                f"positions {seen_keys[spec.response_key]} and {i}. "
                "Each extraction key must appear at most once."
            )
        seen_keys[spec.response_key] = i

    return resolved


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

# Return type for embedding extractors: list of (array, optional_layer_number)
EmbeddingResult = List[Tuple[np.ndarray, Optional[int]]]


@dataclass
class EmbeddingSpec:
    """Declarative specification for extracting embeddings from API responses.

    Covers common response formats without writing a custom function.

    Args:
        key: Response dict key containing the embedding data (e.g.
            ``"embedding"``, ``"seqcoding"``, ``"embeddings"``).
        layer: Which layer to extract when the response contains multiple
            layers (list of ``{layer: int, embedding: [...]}`` dicts).
            ``None`` stores all layers; an ``int`` stores only that layer.
        reduction: Reduce per-token 2-D embeddings to a single vector:
            ``"mean"``, ``"first"``, ``"last"``, ``"sum"``.
            ``None`` stores the full array as-is.

    Examples::

        # ablang2 returns {"seqcoding": [float, ...]}
        EmbeddingSpec(key="seqcoding")

        # esm2-8m returns {"embeddings": [{embedding: [...], layer: 33}]}
        # Store only layer 33:
        EmbeddingSpec(key="embeddings", layer=33)

        # Per-residue → mean-pool:
        EmbeddingSpec(key="embedding", reduction="mean")
    """

    key: str
    layer: Optional[int] = None
    reduction: Optional[str] = None

    def __call__(self, result: dict) -> EmbeddingResult:
        """Extract embeddings from an API response dict."""
        if not isinstance(result, dict):
            return []
        val = result.get(self.key)
        if val is None:
            # B15 fix: warn on wrong key so misconfigured EmbeddingSpec surfaces immediately
            logger.warning(
                "EmbeddingSpec: key %r not found in API response — "
                "embedding will not be stored. Available keys: %s",
                self.key,
                list(result.keys()) if isinstance(result, dict) else type(result),
            )
            return []

        # Case 1: flat list of floats or a nested numeric array
        if isinstance(val, (list, np.ndarray)):
            arr = np.array(val)
            if arr.size == 0:
                return []

            # Check if it's a list of dicts (multi-layer format)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return self._extract_layers(val)

            # Apply reduction if requested (e.g. per-token 2-D → 1-D)
            arr = self._apply_reduction(arr)
            return [(arr, None)]

        return []

    def _extract_layers(self, items: list[dict]) -> EmbeddingResult:
        """Handle list-of-dicts format: [{layer: 0, embedding: [...]}, ...]."""
        results: EmbeddingResult = []
        for item in items:
            layer_num = item.get("layer")
            emb_data = item.get("embedding")
            if emb_data is None:
                continue
            if self.layer is not None and layer_num != self.layer:
                continue
            arr = np.array(emb_data)
            if arr.size > 0:
                arr = self._apply_reduction(arr)
                results.append((arr, layer_num))
        return results

    def _apply_reduction(self, arr: np.ndarray) -> np.ndarray:
        """Reduce a 2-D array to 1-D if a reduction is set."""
        if self.reduction is None or arr.ndim < 2:
            return arr
        if self.reduction == "mean":
            return arr.mean(axis=0)
        elif self.reduction == "first":
            return arr[0]
        elif self.reduction == "last":
            return arr[-1]
        elif self.reduction == "sum":
            return arr.sum(axis=0)
        return arr


@dataclass
class StructureSpec:
    """Specification for extracting and storing a structure from an API response.

    Args:
        key: Response dict key containing the structure string (e.g. "pdb", "cif", "pdbs").
        format: Structure format — "pdb" or "cif". Auto-detected from key if None.
        plddt_key: Optional response key for a confidence score to store alongside.
        index: For list-valued keys (e.g. AF2 "pdbs"), which element to store (default 0).
    """

    key: str
    format: Optional[str] = None
    plddt_key: Optional[str] = None
    index: Optional[int] = 0

    def detect_format(self) -> str:
        if self.format:
            return self.format
        return "pdb" if "pdb" in self.key.lower() else "cif"


@dataclass
class MatrixExtractionSpec:
    """Flattens a per-mutation response into individual prediction rows.

    Each mutation becomes a separate prediction row with prediction_type
    formatted as '{prefix}_{label}' (e.g., 'ddg_M1A').

    Two modes:
        - **Matrix mode** (SPURS): 2D array + row/col labels.
        - **List mode** (ThermoMPNN): list of dicts with mutation name + value.

    Args:
        prefix: Prediction type prefix (e.g. "ddg").
        values_key: Dot-path to 2D array in response (matrix mode).
        row_labels_key: Dot-path to row labels (position labels).
        col_labels_key: Dot-path to column labels (amino acid labels).
        mutation_key: Dict key for mutation name (list mode). If set, uses list mode.
        value_key: Dict key for the numeric value (list mode).
    """

    prefix: str = "ddg"
    # Matrix mode (SPURS)
    values_key: str = "ddG_matrix.values"
    row_labels_key: str = "ddG_matrix.residue_axis"
    col_labels_key: str = "ddG_matrix.amino_acid_axis"
    # List mode (ThermoMPNN)
    mutation_key: Optional[str] = None
    value_key: Optional[str] = None


def _dot_access(obj: dict, path: str):
    """Access nested dict keys via dot-separated path (e.g. 'ddG_matrix.values')."""
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            raise KeyError(f"Cannot access '{part}' on {type(obj)}")
    return obj


class PredictionStage(Stage):
    """
    Generic prediction stage using BioLM API.

    Uses ``merge_mode = "union"`` so that when multiple prediction stages run
    in parallel, sequences processed by *any* stage survive the merge — i.e.,
    independent predictions do not cancel each other out.

    Args:
        name: Stage name
        model_name: BioLM model name (e.g., 'esmfold', 'esm2', 'temberture-regression')
        action: API action ('predict', 'encode', 'score')
        prediction_type: Type of prediction for caching (e.g., 'structure', 'stability', 'embedding')
        params: Optional parameters for the API call
        batch_size: Number of sequences per pipeline batch (default 32).
            Each pipeline batch becomes one SDK call, which the SDK may
            further split by the model's ``maxItems`` schema limit.
        max_concurrent: Maximum pipeline batches in flight at once (default 5).
            Controls how many batches are dispatched concurrently at the
            pipeline level.  Higher values keep the API more saturated but
            use more memory (up to ``max_concurrent * batch_size`` items
            plus their responses in memory at once).
        max_connections: Maximum concurrent HTTP connections to the API
            (default 10).  This is the SDK-level semaphore that throttles
            the actual HTTP requests.  Each pipeline batch may be split
            into multiple sub-requests by the model's ``maxItems`` limit;
            ``max_connections`` caps how many of those sub-requests run
            simultaneously across all in-flight batches.
        structure_output: Store the structure from this model's response.
        structure_input: Inject structures from upstream models into API items.
            Maps API field name → source model name (e.g. ``{"pdb": "esmfold"}``).
        matrix_extraction: Flatten a per-mutation response into individual
            prediction rows (e.g. for DMS heatmaps).
    """

    merge_mode = "union"

    def __init__(
        self,
        name: str,
        model_name: str,
        action: str = "predict",
        params: Optional[dict] = None,
        batch_size: int = 32,
        max_concurrent: int = 5,
        max_connections: int = 10,
        item_columns: Optional[dict[str, str]] = None,
        extractions: Optional[Union[str, list[Union[str, ExtractionSpec]]]] = None,
        columns: Optional[Union[str, dict[str, str]]] = None,
        embedding_extractor: Optional[Union[EmbeddingSpec, Callable[[dict], Any]]] = None,
        structure_output: Optional[StructureSpec] = None,
        structure_input: Optional[dict[str, str]] = None,
        matrix_extraction: Optional[MatrixExtractionSpec] = None,
        **kwargs,
    ):
        # Extract skip_on_error before passing to parent
        skip_on_error = kwargs.pop("skip_on_error", False)

        self.action = action
        self.params = params or {}
        self.batch_size = batch_size
        self.skip_on_error = skip_on_error
        self._max_connections = max_connections
        # item_columns: maps API field name → DataFrame column name.
        # E.g. {'H': 'heavy_chain', 'L': 'light_chain'} for abodybuilder3.
        # When None, defaults to {'sequence': 'sequence'}.
        self.item_columns = item_columns

        # Embedding extraction: user-defined function or EmbeddingSpec.
        self._embedding_extractor = embedding_extractor

        # Structure output/input specs
        self._structure_output = structure_output
        self._structure_input = structure_input
        self._matrix_extraction = matrix_extraction

        # --- Resolve extractions + columns into list[_ResolvedExtraction] ---
        self._resolved: list[_ResolvedExtraction] = []

        if action in ("predict", "score"):
            # extractions can be None when structure_output or matrix_extraction
            # is the primary output (no scalar extraction needed)
            if extractions is None and structure_output is None and matrix_extraction is None:
                raise ValueError(
                    f"PredictionStage '{name}': `extractions` is required. "
                    f"Set extractions='response_key' or "
                    f"extractions=[ExtractionSpec('key', reduction='mean')]. "
                    f"Check your model's API response to find the correct key."
                )
            if extractions is not None:
                self._resolved = _resolve_extractions(extractions, columns)

        if action == "encode" and self._embedding_extractor is None:
            raise ValueError(
                f"PredictionStage '{name}': `embedding_extractor` is required. "
                f"Set embedding_extractor=EmbeddingSpec('response_key') or "
                f"pass a callable. Check your model's API response for the "
                f"embedding key (e.g. 'embedding', 'seqcoding', 'embeddings')."
            )

        # prediction_type: first column name (used for cache checks, stage naming)
        self.prediction_type = (
            self._resolved[0].column if self._resolved else f"{model_name}_{action}"
        )

        # Cache key: auto-derived from model + action + sorted response keys
        response_keys = sorted({r.response_key for r in self._resolved})
        cache_key = (
            f"{model_name}::{action}::{','.join(response_keys)}"
            if response_keys
            else f"{model_name}_{action}"
        )

        super().__init__(
            name=name,
            cache_key=cache_key,
            model_name=model_name,
            max_concurrent=max_concurrent,
            **kwargs,
        )

    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        embedding_extractor_spec = None
        if isinstance(self._embedding_extractor, EmbeddingSpec):
            embedding_extractor_spec = {
                "type": "EmbeddingSpec",
                "key": self._embedding_extractor.key,
                "layer": self._embedding_extractor.layer,
                "reduction": self._embedding_extractor.reduction,
            }
        elif callable(self._embedding_extractor):
            raise NotImplementedError(
                f"PredictionStage '{self.name}': custom callable embedding_extractor "
                "cannot be serialized. Use EmbeddingSpec instead of a lambda/function."
            )
        return {
            "type": "PredictionStage",
            "name": self.name,
            "model_name": self.model_name,
            "action": self.action,
            "resolved": [
                {
                    "response_key": r.response_key,
                    "column": r.column,
                    "reduction": r.reduction,
                }
                for r in self._resolved
            ],
            "params": self.params,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "max_connections": self._max_connections,
            "item_columns": self.item_columns,
            "embedding_extractor": embedding_extractor_spec,
            "structure_output": (
                {"key": self._structure_output.key, "format": self._structure_output.format,
                 "plddt_key": self._structure_output.plddt_key, "index": self._structure_output.index}
                if self._structure_output else None
            ),
            "structure_input": self._structure_input,
            "matrix_extraction": (
                {"prefix": self._matrix_extraction.prefix,
                 "values_key": self._matrix_extraction.values_key,
                 "row_labels_key": self._matrix_extraction.row_labels_key,
                 "col_labels_key": self._matrix_extraction.col_labels_key,
                 "mutation_key": self._matrix_extraction.mutation_key,
                 "value_key": self._matrix_extraction.value_key}
                if self._matrix_extraction else None
            ),
            "skip_on_error": self.skip_on_error,
            "depends_on": self.depends_on,
        }

    async def process_streaming(self, df: pd.DataFrame, datastore: DataStore, **kwargs):
        """
        Process sequences and yield results as batches complete (streaming).

        Yields DataFrames as API batches complete instead of waiting for all results.
        This allows downstream stages to start processing immediately.
        """

        start_count = len(df)

        # Ensure sequence_ids are present (normally added by _get_initial_data)
        if "sequence_id" not in df.columns:
            df = df.copy()
            df["sequence_id"] = datastore.add_sequences_batch(df["sequence"].tolist())

        # Vectorized cache check: a sequence is uncached if ANY resolved column is missing.
        # For multi-extraction stages we must check every column — checking only the first
        # would silently skip sequences that have col[0] but not col[1].
        all_seq_ids = df["sequence_id"].tolist()
        uncached_id_set: set[int] = set()
        for spec in self._resolved:
            uncached_id_set |= set(
                datastore.get_uncached_sequence_ids(all_seq_ids, spec.column, self.model_name)
            )
        uncached_ids = list(uncached_id_set)
        uncached_mask = df["sequence_id"].isin(uncached_id_set)
        df_uncached = df[uncached_mask].copy()
        cached_count = start_count - len(df_uncached)

        print(f"  Cached: {cached_count}/{start_count}")
        print(f"  To compute: {len(df_uncached)} (streaming)")

        # Yield cached results first using bulk fetch (single JOIN, not N queries).
        # Failed-prediction rows are stored with value NULL; emitting them here
        # would feed NaN-laden rows to a downstream filter with keep_na=True.
        # Drop them from the cached yield — the streaming consumer should only
        # see successful predictions.
        if cached_count > 0:
            df_cached = df[~uncached_mask].copy()
            merge_specs = self._resolved
            cached_seq_ids = df_cached["sequence_id"].tolist()
            for spec in merge_specs:
                pred_df = datastore.get_predictions_bulk(
                    cached_seq_ids, spec.column, self.model_name
                )
                if not pred_df.empty:
                    val_map = dict(zip(pred_df["sequence_id"], pred_df["value"]))
                    df_cached[spec.column] = df_cached["sequence_id"].map(val_map)
            spec_cols = [s.column for s in merge_specs if s.column in df_cached.columns]
            if spec_cols:
                df_cached = df_cached.dropna(subset=spec_cols)
            if not df_cached.empty:
                yield df_cached

        if len(df_uncached) == 0:
            return

        # Pre-fetch upstream structures in one query per source model so
        # streaming has the same N→1 query reduction as the batch paths.
        struct_lookup: dict[str, dict[int, dict]] = {}
        if self._structure_input:
            uncached_seq_ids_for_struct = df_uncached["sequence_id"].tolist()
            for api_field, source_model in self._structure_input.items():
                struct_lookup[api_field] = datastore.get_structures_for_ids(
                    uncached_seq_ids_for_struct, source_model
                )

        _stream_semaphore = asyncio.Semaphore(self._max_connections)
        api = BioLMApiClient(
            self.model_name, semaphore=_stream_semaphore, retry_error_batches=True,
        )
        batch_size = self.batch_size
        pending_tasks: dict = {}
        remaining_tasks: set = set()
        try:
            for i in range(0, len(df_uncached), batch_size):
                batch_df = df_uncached.iloc[i : i + batch_size]
                batch_indices = batch_df.index

                if self.item_columns:
                    items = [
                        {
                            api_field: row[col]
                            for api_field, col in self.item_columns.items()
                        }
                        for _, row in batch_df.iterrows()
                    ]
                else:
                    items = [{"sequence": seq} for seq in batch_df["sequence"].tolist()]

                if struct_lookup:
                    for batch_idx, (_, row) in enumerate(batch_df.iterrows()):
                        seq_id = int(row["sequence_id"])
                        for api_field, struct_map in struct_lookup.items():
                            struct = struct_map.get(seq_id)
                            if struct and struct.get("structure_str"):
                                items[batch_idx][api_field] = struct["structure_str"]

                if self.action == "encode":
                    task = asyncio.create_task(api.encode(items=items, params=self.params))
                elif self.action == "score":
                    task = asyncio.create_task(api.score(items=items, params=self.params))
                else:
                    task = asyncio.create_task(api.predict(items=items, params=self.params))

                pending_tasks[task] = (batch_df, batch_indices)

            remaining_tasks = set(pending_tasks.keys())
            while remaining_tasks:
                # Wait for next batch to complete
                done, remaining_tasks = await asyncio.wait(
                    remaining_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    try:
                        results = await completed_task
                        # Some models (e.g. esmc score with 1-item batch) return a bare
                        # dict instead of a list — wrap it so the per-result loop works.
                        if isinstance(results, dict):
                            _maybe_raise_fatal_api_error(results, self.model_name)
                            results = [results]
                        pending_batch_df, batch_indices = pending_tasks[completed_task]
                        out_df = pending_batch_df.copy()

                        # Store results and add to DataFrame.
                        # Collect into batch_data and flush once (not per-row add_prediction).
                        batch_data = []
                        batch_emb_data = []
                        # B11 fix: warn on length mismatch; trailing unmatched rows in
                        # out_df remain with NaN prediction values (already correct since
                        # out_df = pending_batch_df.copy() contains all rows).
                        if len(results) != len(pending_batch_df):
                            logger.warning(
                                "process_streaming: API returned %d results for %d items "
                                "in batch — %d rows will be yielded with NaN predictions",
                                len(results),
                                len(pending_batch_df),
                                len(pending_batch_df) - len(results),
                            )
                        for (_, row), result in zip(pending_batch_df.iterrows(), results):
                            seq_id = int(row["sequence_id"])
                            idx = row.name
                            _maybe_raise_fatal_api_error(result, self.model_name)

                            if self.action in ("predict", "score"):
                                if isinstance(result, dict):
                                    for spec in self._resolved:
                                        val = self._extract_with_spec(result, spec)
                                        if val is not None:
                                            batch_data.append(
                                                {
                                                    "sequence_id": seq_id,
                                                    "prediction_type": spec.column,
                                                    "model_name": self.model_name,
                                                    "value": val,
                                                    "metadata": {"params": self.params},
                                                }
                                            )
                                            out_df.at[idx, spec.column] = val
                                    # Store structure if configured
                                    if self._structure_output is not None:
                                        self._store_structure(datastore, seq_id, result)
                                    # Extract matrix predictions if configured
                                    if self._matrix_extraction is not None:
                                        pairs = self._extract_matrix(result)
                                        for pred_type, mval in pairs:
                                            batch_data.append(
                                                {
                                                    "sequence_id": seq_id,
                                                    "prediction_type": pred_type,
                                                    "model_name": self.model_name,
                                                    "value": mval,
                                                    "metadata": {"params": self.params},
                                                }
                                            )

                            elif self.action == "encode":
                                batch_emb_data.extend(
                                    self._collect_embedding_rows(seq_id, result)
                                )

                        # Single batch insert per completed async task
                        if batch_data:
                            datastore.add_predictions_batch(batch_data)
                        if batch_emb_data:
                            datastore.add_embeddings_batch(batch_emb_data)

                        # Drop rows where spec columns are NaN — matches cached-path
                        # dropna at line 621 so streaming and resume produce identical output.
                        _spec_cols = [s.column for s in self._resolved if s.column in out_df.columns]
                        if _spec_cols:
                            out_df = out_df.dropna(subset=_spec_cols)
                        # Yield batch immediately!
                        yield out_df

                    except PipelineAPIAuthError:
                        # Auth/billing failures aren't survivable — every batch
                        # would hit the same upstream error.  Bubble out even
                        # when skip_on_error=True.
                        raise
                    except Exception as e:
                        if self.skip_on_error:
                            # use .get() to avoid KeyError if the task
                            # was already removed from pending_tasks in the normal path.
                            _pending_entry = pending_tasks.get(completed_task)
                            pending_batch_df = _pending_entry[0] if _pending_entry is not None else None
                            print(f"  Error processing batch (skipped): {e}")
                            # Write one NULL row per resolved column so that resume
                            # correctly detects all columns as present (failed).
                            _failed_seq_ids = (
                                pending_batch_df["sequence_id"].tolist()
                                if pending_batch_df is not None
                                else []
                            )
                            failed_batch = [
                                {
                                    "sequence_id": int(sid),
                                    "prediction_type": spec.column,
                                    "model_name": self.model_name,
                                    "value": None,
                                    "metadata": {
                                        "status": "failed",
                                        "error": str(e),
                                        "params": self.params,
                                    },
                                }
                                for sid in _failed_seq_ids
                                for spec in self._resolved
                            ]
                            if failed_batch:
                                datastore.add_predictions_batch(failed_batch)
                            # Don't yield failed batch - sequences are filtered out
                        else:
                            print(f"  Error processing batch: {e}")
                            raise
        finally:
            # Cancel any remaining in-flight tasks to prevent orphaned writes
            for task in remaining_tasks:
                task.cancel()
            if remaining_tasks:
                await asyncio.gather(*remaining_tasks, return_exceptions=True)
            await api.shutdown()

    @staticmethod
    def _extract_with_spec(
        result: dict, spec: Union[ExtractionSpec, _ResolvedExtraction]
    ) -> Optional[float]:
        """Extract a value from an API result using an extraction spec.

        Looks up spec.response_key in the result dict. If the value is a list/array
        and spec.reduction is set, applies the reduction (mean/max/min/sum).
        """
        val = result.get(spec.response_key)
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        # numpy arrays are not lists; check for both.
        if isinstance(val, (list, np.ndarray)) and (
            len(val) > 0 if hasattr(val, '__len__') else True
        ):
            # Array value — apply reduction only when the user explicitly requested one.
            # spec.reduction=None means "expect a scalar"; return None for array values
            # so the caller can surface an informative error rather than silently
            # collapsing the array to a mean the user didn't ask for.
            if spec.reduction is None:
                return None
            try:
                if isinstance(val, np.ndarray):
                    arr = [float(x) for x in val.flat if isinstance(x, (int, float, np.integer, np.floating))]
                else:
                    # Flatten nested lists (e.g. plddt: [[0.95, 0.92, ...]])
                    flat = []
                    for x in val:
                        if isinstance(x, (list, tuple)):
                            flat.extend(x)
                        else:
                            flat.append(x)
                    arr = [float(x) for x in flat if isinstance(x, (int, float))]
            except (TypeError, ValueError) as exc:
                logger.debug("_extract_with_spec failed for key '%s': %s", spec.response_key, exc)
                return None
            if not arr:
                return None
            reduction = spec.reduction
            if reduction == "mean":
                return float(np.mean(arr))
            elif reduction == "max":
                return float(np.max(arr))
            elif reduction == "min":
                return float(np.min(arr))
            elif reduction == "sum":
                return float(np.sum(arr))
            else:
                raise ValueError(
                    f"Unknown reduction '{reduction}'. "
                    "Valid values: 'mean', 'max', 'min', 'sum'."
                )
        return None

    def _store_embeddings(self, datastore: DataStore, seq_id: int, result: Any):
        """Store embedding(s) from a single API result dict into the datastore.

        Uses ``self._embedding_extractor`` (``EmbeddingSpec`` or callable)
        to extract embeddings from the API response.
        """
        if not isinstance(result, dict):
            return

        extracted = self._embedding_extractor(result)

        for embedding, layer in extracted:
            if not isinstance(embedding, np.ndarray):
                embedding = np.asarray(embedding, dtype=float)
            if embedding.size > 0:
                datastore.add_embedding(seq_id, self.model_name, embedding, layer=layer)

    def _inject_structures(
        self,
        datastore: DataStore,
        items: list[dict],
        seq_ids: list,
    ) -> None:
        """Mutate ``items`` in place, injecting upstream structure strings.

        For each ``(api_field, source_model)`` in ``self._structure_input`` we
        do one bulk lookup against the datastore, then attach the structure
        string to each item that has a corresponding sequence_id.  Used by
        ``process``, ``process_ws``, and ``process_streaming``.
        """
        if not self._structure_input:
            return
        for api_field, source_model in self._structure_input.items():
            struct_map = datastore.get_structures_for_ids(
                [int(s) for s in seq_ids], source_model
            )
            for i, seq_id in enumerate(seq_ids):
                struct = struct_map.get(int(seq_id))
                if struct and struct.get("structure_str"):
                    items[i][api_field] = struct["structure_str"]

    def _collect_embedding_rows(
        self, seq_id: int, result: Any
    ) -> list[dict]:
        """Build the list of embedding-batch dicts for one API result.

        Shared helper used by ``process``, ``process_ws``, and ``process_streaming``
        to avoid three near-identical inline blocks.  Empty arrays produce a
        warning rather than silent drop so the caller can investigate model
        responses that should have produced data.
        """
        if not isinstance(result, dict):
            return []
        rows: list[dict] = []
        for embedding, layer in self._embedding_extractor(result):
            if not isinstance(embedding, np.ndarray):
                embedding = np.asarray(embedding, dtype=float)
            if embedding.size == 0:
                logger.warning(
                    "Stage %r received empty embedding for sequence_id=%d "
                    "(model=%s, layer=%s) — dropping",
                    self.name, seq_id, self.model_name, layer,
                )
                continue
            if embedding.ndim != 1:
                logger.warning(
                    "Stage %r: sequence_id=%d returned %dD embedding "
                    "(model=%s, layer=%s) — use reduction='mean' or 'sum' to "
                    "collapse to 1D; dropping to prevent DuckDB FLOAT[] type error",
                    self.name, seq_id, embedding.ndim, self.model_name, layer,
                )
                continue
            rows.append({
                "sequence_id": seq_id,
                "model_name": self.model_name,
                "embedding": embedding,
                "layer": layer,
            })
        return rows

    def _store_structure(self, datastore: DataStore, seq_id: int, result: dict):
        """Store a structure string from an API result into the datastore."""
        spec = self._structure_output
        val = result.get(spec.key)
        if val is None:
            # B14 fix: warn instead of silently returning so misconfigured keys are visible
            logger.warning(
                "_store_structure: key %r not found in API response for sequence_id %s "
                "(model=%s) — no structure stored; check structure_key configuration. "
                "Available keys: %s",
                spec.key,
                seq_id,
                self.model_name,
                list(result.keys()) if isinstance(result, dict) else type(result),
            )
            return
        # Handle list-valued keys (e.g. AF2 "pdbs")
        if isinstance(val, list):
            idx = spec.index if spec.index is not None else 0
            if idx < len(val):
                val = val[idx]
            else:
                return
        if not isinstance(val, str) or not val:
            return
        fmt = spec.detect_format()
        plddt = None
        if spec.plddt_key:
            plddt_val = result.get(spec.plddt_key)
            if isinstance(plddt_val, (int, float)):
                plddt = float(plddt_val)
        datastore.add_structure(seq_id, self.model_name, structure_str=val, format=fmt, plddt_mean=plddt)

    def _extract_matrix(self, result) -> list[tuple[str, float]]:
        """Extract per-mutation predictions from a matrix or list response.

        Returns list of (prediction_type, value) pairs.
        """
        spec = self._matrix_extraction
        if spec.mutation_key:
            # List mode (ThermoMPNN): [{mutation: "M1A", ddg: -0.5}, ...]
            items = result if isinstance(result, list) else [result]
            pairs = []
            for item in items:
                if isinstance(item, dict):
                    name = item.get(spec.mutation_key)
                    val = item.get(spec.value_key)
                    if name is not None and val is not None:
                        pairs.append((f"{spec.prefix}_{name}", float(val)))
            return pairs
        else:
            # Matrix mode (SPURS): 2D array + axis labels
            try:
                values = _dot_access(result, spec.values_key)
                rows = _dot_access(result, spec.row_labels_key)
                cols = _dot_access(result, spec.col_labels_key)
            except (KeyError, TypeError) as e:
                logger.warning("_extract_matrix: could not access matrix data: %s", e)
                return []
            pairs = []
            for i, row_label in enumerate(rows):
                # Use standard mutation notation: {WT}{pos}{MT} (e.g., "M1A").
                # If the row label already includes a position number (e.g., "M1"),
                # use it as-is. Otherwise, prepend the 1-based index.
                import re as _re_mod
                if _re_mod.search(r'\d', str(row_label)):
                    pos_label = row_label
                else:
                    pos_label = f"{row_label}{i + 1}"
                for j, col_label in enumerate(cols):
                    try:
                        pairs.append((f"{spec.prefix}_{pos_label}{col_label}", float(values[i][j])))
                    except (IndexError, TypeError, ValueError):
                        continue
            return pairs

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Process sequences through prediction model (legacy DataFrame path).

        Uses the same bounded-concurrency batching as ``process_ws`` — see
        that method's docstring for the design rationale.

        Performance: single DuckDB anti-join for cache detection, bounded-
        concurrency API dispatch, single JOIN query to merge predictions back.
        """
        start_count = len(df)

        # Ensure sequence_ids are present (normally added by _get_initial_data)
        if "sequence_id" not in df.columns:
            df = df.copy()
            df["sequence_id"] = datastore.add_sequences_batch(df["sequence"].tolist())

        # --- Cache check: encode stores to embeddings table (not predictions).
        # For predict/score, union uncached across ALL resolved columns so a
        # sequence missing any column is treated as uncached.
        # Bug H fix: scope the encode cache check by layer so that a sequence
        # with only layer=33 stored is NOT treated as cached when layer=None
        # (all-layers) was requested, and vice versa.
        _all_seq_ids = df["sequence_id"].tolist()
        if self.action == "encode":
            _emb_layer_legacy = (
                self._embedding_extractor.layer
                if isinstance(self._embedding_extractor, EmbeddingSpec)
                and self._embedding_extractor.layer is not None
                else None
            )
            if _emb_layer_legacy is not None:
                _cached_emb_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer = ?",
                        [self.model_name, _emb_layer_legacy],
                    ).df()["sequence_id"].tolist()
                )
            else:
                # layer=None means "no specific layer requested" — only treat as cached
                # if stored with layer IS NULL (i.e., the flat single-layer format).
                _cached_emb_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer IS NULL",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
            _uncached_set = {int(sid) for sid in _all_seq_ids if sid not in _cached_emb_ids}
        else:
            _uncached_set: set[int] = set()
            for _spec in self._resolved:
                _uncached_set |= set(
                    datastore.get_uncached_sequence_ids(_all_seq_ids, _spec.column, self.model_name)
                )
            # When structure_output is set, also check structures table so
            # sequences with cached scalars but no stored structure get re-dispatched.
            if self._structure_output:
                _cached_struct_ids_legacy = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM structures WHERE model_name = ?",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
                _uncached_set |= {int(sid) for sid in _all_seq_ids if sid not in _cached_struct_ids_legacy}
            if not self._resolved and self._matrix_extraction:
                _cached_matrix_ids_legacy = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM predictions WHERE model_name = ?",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
                _uncached_set |= {int(sid) for sid in _all_seq_ids if sid not in _cached_matrix_ids_legacy}
        uncached_ids = list(_uncached_set)
        uncached_mask = df["sequence_id"].isin(_uncached_set)
        df_uncached = df[uncached_mask]
        cached_count = start_count - len(df_uncached)

        print(f"  Cached: {cached_count}/{start_count}")
        print(f"  To compute: {len(df_uncached)}")

        if len(df_uncached) > 0:
            print(f"  Calling {self.model_name}.{self.action}...")

            seq_ids = df_uncached["sequence_id"].tolist()
            seq_id_to_seq = dict(
                zip(df_uncached["sequence_id"], df_uncached["sequence"])
            )

            # Build items list
            if self.item_columns:
                # validate all item_columns values exist in the DataFrame.
                missing_cols = [
                    col for col in self.item_columns.values()
                    if col not in df_uncached.columns
                ]
                if missing_cols:
                    raise ValueError(
                        f"PredictionStage '{self.name}': item_columns references column(s) "
                        f"{missing_cols} that are not present in the DataFrame. "
                        f"Available columns: {list(df_uncached.columns)}"
                    )
                all_items = [
                    {
                        api_field: row[col]
                        for api_field, col in self.item_columns.items()
                    }
                    for _, row in df_uncached.iterrows()
                ]
            else:
                all_items = [
                    {"sequence": seq} for seq in df_uncached["sequence"].tolist()
                ]

            self._inject_structures(datastore, all_items, seq_ids)

            # Defer API client creation until all synchronous prep has
            # succeeded.  Anything above this line that raises (item-column
            # validation, structure fetch, items construction) does so
            # without leaking a connection pool.
            _process_semaphore = asyncio.Semaphore(self._max_connections)
            api = BioLMApiClient(
                self.model_name, semaphore=_process_semaphore, retry_error_batches=True,
            )

            batch_size = self.batch_size
            n_batches = (len(all_items) + batch_size - 1) // batch_size
            flight_semaphore = asyncio.Semaphore(self.max_concurrent)
            # Mutable list: first successful API response is appended by _dispatch_batch
            # and validated after asyncio.gather to catch misspelled extraction keys.
            _preflight_capture: list = []

            async def _dispatch_batch(batch_idx, _items=all_items, _seq_ids=seq_ids):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(_items))
                batch_items = _items[start:end]
                batch_seq_ids = _seq_ids[start:end]

                try:
                    async with flight_semaphore:
                        if self.action == "encode":
                            results = await api.encode(
                                items=batch_items, params=self.params
                            )
                        elif self.action == "score":
                            results = await api.score(
                                items=batch_items, params=self.params
                            )
                        else:
                            results = await api.predict(
                                items=batch_items, params=self.params
                            )
                        # detect top-level error dict returned instead of a list.
                        if isinstance(results, dict) and (
                            "error" in results or "status_code" in results
                        ):
                            _maybe_raise_fatal_api_error(results, self.model_name)
                            logger.warning(
                                "API returned error dict for batch %d: %s", batch_idx, results
                            )
                            raise ValueError(str(results))
                        if not isinstance(results, list):
                            results = [results]

                    # Capture first valid dict response for post-gather key validation.
                    # asyncio is single-threaded so the check+append is atomic.
                    if (
                        not _preflight_capture
                        and results
                        and isinstance(results[0], dict)
                        and "error" not in results[0]
                        and "status_code" not in results[0]
                    ):
                        _preflight_capture.append(results[0])

                    batch_data = []
                    batch_emb_data = []
                    # B10 fix: guard against positional drift from length-mismatched responses;
                    # raising here routes through the existing except block which writes NULLs.
                    if len(results) != len(batch_seq_ids):
                        raise ValueError(
                            f"API returned {len(results)} results for {len(batch_seq_ids)} items — "
                            "length mismatch; treating batch as failed to prevent positional drift"
                        )
                    for seq_id, result in zip(batch_seq_ids, results):
                        if (
                            isinstance(result, dict)
                            and "error" in result
                            and "status_code" in result
                        ):
                            _maybe_raise_fatal_api_error(result, self.model_name)
                            err = result["error"]
                            err_str = str(err)
                            seq = seq_id_to_seq.get(seq_id, "?")
                            if (
                                "ACDEFGHIKLMNPQRSTVWY" in err_str
                                or "Residues can only" in err_str
                            ):
                                invalid = sorted(
                                    {c for c in str(seq) if c not in "ACDEFGHIKLMNPQRSTVWY"}
                                )
                                print(
                                    f"  WARNING: seq_id {seq_id} '{str(seq)[:20]}...' skipped — "
                                    f"non-standard residue(s) {invalid} not accepted by {self.model_name}"
                                )
                            else:
                                msg = (
                                    "; ".join(
                                        f"{k}: {v[0] if isinstance(v, list) else v}"
                                        for k, v in err.items()
                                    )
                                    if isinstance(err, dict)
                                    else str(err)[:120]
                                )
                                print(f"  WARNING: seq_id {seq_id} skipped — {msg}")
                            continue

                        if self.action in ("predict", "score"):
                            if isinstance(result, dict):
                                for spec in self._resolved:
                                    value = self._extract_with_spec(result, spec)
                                    if value is None:
                                        logger.warning(
                                            "seq_id %s: could not extract '%s' from response — skipping",
                                            seq_id,
                                            spec.response_key,
                                        )
                                        continue
                                    batch_data.append(
                                        {
                                            "sequence_id": seq_id,
                                            "prediction_type": spec.column,
                                            "model_name": self.model_name,
                                            "value": value,
                                            "metadata": {"params": self.params},
                                        }
                                    )
                                # Store structure if configured
                                if self._structure_output is not None:
                                    self._store_structure(datastore, seq_id, result)
                                # Extract matrix predictions if configured
                                if self._matrix_extraction is not None:
                                    pairs = self._extract_matrix(result)
                                    for pred_type, val in pairs:
                                        batch_data.append(
                                            {
                                                "sequence_id": seq_id,
                                                "prediction_type": pred_type,
                                                "model_name": self.model_name,
                                                "value": val,
                                                "metadata": {"params": self.params},
                                            }
                                        )
                        elif self.action == "encode":
                            batch_emb_data.extend(
                                self._collect_embedding_rows(seq_id, result)
                            )

                    if batch_data:
                        datastore.add_predictions_batch(batch_data)
                    if batch_emb_data:
                        datastore.add_embeddings_batch(batch_emb_data)

                except PipelineAPIAuthError:
                    raise
                except Exception as e:
                    # Bug D fix: mark only THIS batch's sequences as failed (not all uncached).
                    if self.skip_on_error:
                        print(f"  Error in batch {batch_idx} (skipped): {e}")
                        failed = [
                            {
                                "sequence_id": sid,
                                "prediction_type": spec.column,
                                "model_name": self.model_name,
                                "value": None,
                                "metadata": {
                                    "status": "failed",
                                    "error": str(e),
                                    "params": self.params,
                                },
                            }
                            for sid in batch_seq_ids
                            for spec in self._resolved
                        ]
                        if failed:
                            try:
                                datastore.add_predictions_batch(failed)
                            except Exception as _fm_err:
                                # Non-serializable params or other metadata issue must
                                # not escape the skip_on_error boundary — just warn.
                                logger.warning(
                                    "Failed to write failure markers for batch %d: %s",
                                    batch_idx, _fm_err,
                                )
                    else:
                        raise

            try:
                # use return_exceptions=True so one failing batch
                # doesn't cancel all other in-flight tasks.
                _legacy_results = await asyncio.gather(
                    *[_dispatch_batch(i) for i in range(n_batches)],
                    return_exceptions=True,
                )
                _first_legacy_exc = None
                for _lr in _legacy_results:
                    # PipelineAPIAuthError (401/402/403) is fatal regardless of
                    # skip_on_error — auth/billing failures must surface, not
                    # silently produce empty results.
                    if isinstance(_lr, PipelineAPIAuthError):
                        _first_legacy_exc = _lr
                        break
                    if isinstance(_lr, Exception) and not self.skip_on_error:
                        _first_legacy_exc = _lr
                        break
                if _first_legacy_exc is not None:
                    raise _first_legacy_exc
                # Post-gather preflight: validate extraction keys against first captured
                # response.  Raises before the merge step so the caller sees a specific
                # error instead of a DataFrame of NaN columns for every sequence.
                if (
                    self.action in ("predict", "score")
                    and self._resolved
                    and _preflight_capture
                ):
                    _pf_r = _preflight_capture[0]
                    for _spec in self._resolved:
                        if _spec.response_key not in _pf_r:
                            raise ValueError(
                                f"PredictionStage preflight: extraction key "
                                f"{_spec.response_key!r} (column={_spec.column!r}) "
                                f"not found in API response for model "
                                f"{self.model_name!r}. "
                                f"Available keys: {list(_pf_r.keys())}. "
                                f"Full response sample: {_pf_r}"
                            )
            except Exception as e:
                print(f"  Error during prediction: {e}")
                raise
            finally:
                # always shut down the local API client
                await api.shutdown()

        # --- Vectorized result merge: single JOIN query, not N individual queries ---
        # Copy so we don't mutate the caller's DataFrame
        df = df.copy()
        if self.action in ("predict", "score"):
            # Determine which prediction_types to merge
            merge_specs = self._resolved

            # Materialize once — reused across all specs
            all_seq_ids = df["sequence_id"].tolist()

            for spec in merge_specs:
                pred_df = datastore.get_predictions_bulk(
                    all_seq_ids, spec.column, self.model_name
                )
                if not pred_df.empty:
                    # Map sequence_id → value for O(n) assignment instead of merge
                    val_map = dict(zip(pred_df["sequence_id"], pred_df["value"]))
                    df[spec.column] = df["sequence_id"].map(val_map)
                else:
                    df[spec.column] = None

            # Keep rows where ALL resolved columns have a value (not just the first).
            all_present = pd.Series(True, index=df.index)
            for _s in merge_specs:
                if _s.column in df.columns:
                    all_present &= df[_s.column].notna()
            df_out = df[all_present].copy()
            filtered_count = len(df) - len(df_out)

        elif self.action == "encode":
            # Bug I fix: scope the output gate by layer so sequences with embeddings
            # from a different layer are not incorrectly counted as having this layer.
            if _emb_layer_legacy is not None:
                emb_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer = ?",
                        [self.model_name, _emb_layer_legacy],
                    )
                    .df()["sequence_id"]
                    .tolist()
                )
            else:
                emb_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer IS NULL",
                        [self.model_name],
                    )
                    .df()["sequence_id"]
                    .tolist()
                )
            df_out = df[df["sequence_id"].isin(emb_ids)].copy()
            filtered_count = len(df) - len(df_out)

        else:
            df_out = df.copy()
            filtered_count = 0

        return df_out, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df_out),
            cached_count=cached_count,
            computed_count=len(df_uncached),
            filtered_count=filtered_count,
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Process sequences using WorkingSet with bounded-concurrency batching.

        **Batching strategy (two levels):**

        This method splits uncached sequences into pipeline-level batches of
        ``batch_size`` items (default 32) and keeps up to ``max_concurrent``
        batches in flight at once.  As each batch completes, its results are
        written to DuckDB immediately and a new batch is dispatched.

        Each pipeline batch is itself an SDK call that may be further split
        by the model's ``maxItems`` schema limit — the SDK handles that
        internally via ``asyncio.gather`` throttled by a semaphore.

        **Why bounded concurrency instead of all-at-once or sequential:**

        - *Sequential* (old behavior) leaves the API idle between batches.
          With 300 sequences and 200ms API latency, that's 10 idle round-trips.
        - *All-at-once* (``process_streaming`` style) has no backpressure —
          10k sequences creates 300+ in-flight tasks with all items and
          response payloads in memory simultaneously.
        - *Bounded concurrency* keeps the API saturated (``max_concurrent``
          batches in flight) while bounding memory to at most
          ``max_concurrent * batch_size`` items plus their results.  DuckDB
          writes happen as each batch lands, providing natural backpressure:
          a slow datastore slows down new batch dispatch.

        Steps:
            1. Cache check via anti-join (DuckDB)
            2. Fetch only uncached (sequence_id, sequence) pairs
            3. Dispatch batches with bounded concurrency
            4. Write results to DuckDB as each batch completes
            5. Return WorkingSet of IDs that have predictions
        """
        verbose = kwargs.get("verbose", True)
        input_count = len(ws)
        input_ids = list(ws.sequence_ids)

        # Cache check: encode → embeddings table; predict/score → predictions table.
        # For predict/score: union uncached across ALL resolved columns.
        if self.action == "encode":
            _emb_layer = (
                self._embedding_extractor.layer
                if isinstance(self._embedding_extractor, EmbeddingSpec) and self._embedding_extractor.layer is not None
                else None
            )
            if _emb_layer is not None:
                _cached_emb_ids_ws = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer = ?",
                        [self.model_name, _emb_layer],
                    ).df()["sequence_id"].tolist()
                )
            else:
                # layer=None means "no specific layer requested" — only treat as cached
                # if stored with layer IS NULL (i.e., the flat single-layer format).
                # previously matched ANY layer, now scoped to layer IS NULL.
                _cached_emb_ids_ws = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer IS NULL",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
            uncached_ids = [sid for sid in input_ids if sid not in _cached_emb_ids_ws]
        else:
            _uncached_ws_set: set[int] = set()
            for _spec in self._resolved:
                _uncached_ws_set |= set(
                    datastore.get_uncached_sequence_ids(input_ids, _spec.column, self.model_name)
                )
            # When structure_output is set, also check the structures table.
            # A sequence with cached scalar predictions but no stored structure
            # must be re-dispatched so the structure gets stored too.
            if self._structure_output:
                _cached_struct_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM structures WHERE model_name = ?",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
                _uncached_ws_set |= {sid for sid in input_ids if sid not in _cached_struct_ids}
            # For matrix_extraction (without scalar extractions), check predictions table.
            if not self._resolved and self._matrix_extraction:
                _cached_matrix_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM predictions WHERE model_name = ?",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
                _uncached_ws_set |= {sid for sid in input_ids if sid not in _cached_matrix_ids}
            uncached_ids = list(_uncached_ws_set)
        cached_count = input_count - len(uncached_ids)

        if verbose:
            print(f"  Cached: {cached_count}/{input_count}")
            print(f"  To compute: {len(uncached_ids)}")

        if uncached_ids:
            if verbose:
                print(f"  Calling {self.model_name}.{self.action}...")

            # Fetch (id, sequence) pairs — lightweight, no full DataFrame
            id_seq_pairs = datastore.get_sequences_for_ids(uncached_ids)
            seq_id_to_seq = dict(id_seq_pairs)

            all_seq_ids = [p[0] for p in id_seq_pairs]

            # Build items list — either from item_columns or default sequence
            if self.item_columns:
                col_names = list(self.item_columns.values())
                col_map = datastore.get_sequences_for_ids_with_columns(
                    all_seq_ids, col_names
                )
                if not col_map:
                    col_map = datastore.get_sequence_attributes_for_ids(
                        all_seq_ids, col_names
                    )
                # Bug G fix: raise early if any required column has None/NaN values.
                for api_field, col in self.item_columns.items():
                    null_sids = [
                        sid for sid in all_seq_ids
                        if col_map.get(sid, {}).get(col) in (None, "", "nan", "NaN")
                        or (
                            col_map.get(sid, {}).get(col) is not None
                            and str(col_map.get(sid, {}).get(col)).lower() == "nan"
                        )
                    ]
                    if null_sids:
                        raise ValueError(
                            f"Input column '{col}' has null/NaN values at row(s) "
                            f"{null_sids[:5]}{'...' if len(null_sids) > 5 else ''}. "
                            "Fill or drop these rows before running the pipeline."
                        )
                all_items = []
                for sid in all_seq_ids:
                    vals = col_map.get(sid, {})
                    item = {}
                    for api_field, col in self.item_columns.items():
                        v = vals.get(col)
                        if v is None or v == "":
                            print(
                                f"  WARNING: seq_id {sid} has "
                                f"NULL/empty value for column "
                                f"'{col}' (API field '{api_field}')"
                            )
                        item[api_field] = v or ""
                    all_items.append(item)
            else:
                # Guard: if item_columns is None, sequences must not be empty
                empty = [p[0] for p in id_seq_pairs if not p[1]]
                if empty:
                    raise ValueError(
                        f"Sequences with empty 'sequence' field found for sequence_ids "
                        f"{empty[:5]}. This pipeline uses multi-column input. Set "
                        "item_columns= to map the correct column to the API field "
                        "(e.g., item_columns={'sequence': 'heavy_chain'})."
                    )
                # Bug G fix: check for None/NaN in the default sequence column too.
                null_seqs = [
                    p[0] for p in id_seq_pairs
                    if p[1] is None or not str(p[1]).strip()
                    or str(p[1]).strip().lower() in ("nan", "none")
                ]
                if null_seqs:
                    raise ValueError(
                        f"Input column 'sequence' has null/NaN values at row(s) "
                        f"{null_seqs[:5]}{'...' if len(null_seqs) > 5 else ''}. "
                        "Fill or drop these rows before running the pipeline."
                    )
                all_items = [{"sequence": p[1]} for p in id_seq_pairs]

            self._inject_structures(datastore, all_items, all_seq_ids)

            # Defer API client creation until prep has succeeded so
            # validation/struct-fetch failures above don't leak the pool.
            connection_semaphore = asyncio.Semaphore(self._max_connections)
            api = BioLMApiClient(
                self.model_name, semaphore=connection_semaphore, retry_error_batches=True,
            )

            batch_size = self.batch_size
            n_batches = (len(all_items) + batch_size - 1) // batch_size
            flight_semaphore = asyncio.Semaphore(self.max_concurrent)
            # Mutable list: first successful API response captured by _dispatch_batch
            # and validated post-gather to detect misspelled extraction keys.
            _preflight_capture: list = []

            try:
                from tqdm.auto import tqdm as _tqdm_cls
            except ImportError:
                _tqdm_cls = None
            _pbar = (
                _tqdm_cls(total=n_batches, desc=f"  {self.model_name}", unit="batch")
                if _tqdm_cls and n_batches > 1
                else None
            )

            async def _dispatch_batch(batch_idx, _items=all_items, _seq_ids=all_seq_ids):
                """Send one batch to the API and store results in DuckDB."""
                start = batch_idx * batch_size
                end = min(start + batch_size, len(_items))
                batch_items = _items[start:end]
                batch_seq_ids = _seq_ids[start:end]

                try:
                    async with flight_semaphore:
                        if self.action == "encode":
                            results = await api.encode(
                                items=batch_items, params=self.params
                            )
                        elif self.action == "score":
                            results = await api.score(
                                items=batch_items, params=self.params
                            )
                        else:
                            results = await api.predict(
                                items=batch_items, params=self.params
                            )
                        # detect top-level error dict returned instead of a list.
                        # This happens when the API returns a single error response for the
                        # whole batch rather than per-item results.
                        if isinstance(results, dict) and (
                            "error" in results or "status_code" in results
                        ):
                            # 401/402/403 must surface as PipelineAPIAuthError so the
                            # post-gather loop can re-raise even when skip_on_error=True.
                            _maybe_raise_fatal_api_error(results, self.model_name)
                            logger.warning(
                                "API returned error dict for batch %d: %s", batch_idx, results
                            )
                            raise ValueError(str(results))
                        if not isinstance(results, list):
                            results = [results]

                    # Capture first valid dict response for post-gather key validation.
                    if (
                        not _preflight_capture
                        and results
                        and isinstance(results[0], dict)
                        and "error" not in results[0]
                        and "status_code" not in results[0]
                    ):
                        _preflight_capture.append(results[0])

                    # B10-mirror for process_ws: raise on length mismatch so the
                    # except block below writes NULL sentinel rows for all seq_ids
                    # in this batch, exactly as process() does at line 1194.
                    # Sequences that disappeared from the API response are recorded
                    # as failed rather than silently dropped from the WorkingSet.
                    if len(results) != len(batch_seq_ids):
                        raise ValueError(
                            f"API returned {len(results)} results for "
                            f"{len(batch_seq_ids)} items — length mismatch; "
                            "treating batch as failed to prevent positional drift"
                        )

                    # Process results and write to DuckDB immediately
                    batch_data = []
                    batch_emb_data = []
                    for seq_id, result in zip(batch_seq_ids, results):
                        if (
                            isinstance(result, dict)
                            and "error" in result
                            and "status_code" in result
                        ):
                            _maybe_raise_fatal_api_error(result, self.model_name)
                            err = result["error"]
                            err_str = str(err)
                            seq = seq_id_to_seq.get(seq_id, "?")
                            if (
                                "ACDEFGHIKLMNPQRSTVWY" in err_str
                                or "Residues can only" in err_str
                            ):
                                invalid = sorted(
                                    {
                                        c
                                        for c in str(seq)
                                        if c not in "ACDEFGHIKLMNPQRSTVWY"
                                    }
                                )
                                print(
                                    f"  WARNING: seq_id {seq_id} '{str(seq)[:20]}...' "
                                    f"skipped — non-standard residue(s) {invalid}"
                                )
                            else:
                                msg = (
                                    "; ".join(
                                        f"{k}: {v[0] if isinstance(v, list) else v}"
                                        for k, v in err.items()
                                    )
                                    if isinstance(err, dict)
                                    else str(err)[:120]
                                )
                                print(f"  WARNING: seq_id {seq_id} skipped — {msg}")
                            continue

                        if self.action in ("predict", "score"):
                            if isinstance(result, dict):
                                for spec in self._resolved:
                                    value = self._extract_with_spec(result, spec)
                                    # Skip None values — don't cache "no result" permanently;
                                    # the sequence will be retried on the next run.
                                    if value is None:
                                        print(
                                            f"  WARNING: seq_id {seq_id}: could not extract "
                                            f"'{spec.response_key}' from response — skipping"
                                        )
                                        continue
                                    batch_data.append(
                                        {
                                            "sequence_id": seq_id,
                                            "prediction_type": spec.column,
                                            "model_name": self.model_name,
                                            "value": value,
                                            "metadata": {"params": self.params},
                                        }
                                    )
                                # Store structure if configured
                                if self._structure_output is not None:
                                    self._store_structure(datastore, seq_id, result)
                                # Extract matrix predictions if configured
                                if self._matrix_extraction is not None:
                                    pairs = self._extract_matrix(result)
                                    for pred_type, val in pairs:
                                        batch_data.append(
                                            {
                                                "sequence_id": seq_id,
                                                "prediction_type": pred_type,
                                                "model_name": self.model_name,
                                                "value": val,
                                                "metadata": {"params": self.params},
                                            }
                                        )
                        elif self.action == "encode":
                            batch_emb_data.extend(
                                self._collect_embedding_rows(seq_id, result)
                            )

                    if batch_data:
                        datastore.add_predictions_batch(batch_data)
                    if batch_emb_data:
                        datastore.add_embeddings_batch(batch_emb_data)

                except PipelineAPIAuthError:
                    raise
                except Exception as e:
                    if self.skip_on_error:
                        print(f"  Error in batch {batch_idx} (skipped): {e}")
                        # Mark only THIS batch's sequences as failed (with NULL) so they
                        # are not silently retried forever but are flagged in the output.
                        failed = [
                            {
                                "sequence_id": sid,
                                "prediction_type": spec.column,
                                "model_name": self.model_name,
                                "value": None,
                                "metadata": {
                                    "status": "failed",
                                    "error": str(e),
                                    "params": self.params,
                                },
                            }
                            for sid in batch_seq_ids
                            for spec in self._resolved
                        ]
                        if failed:
                            try:
                                datastore.add_predictions_batch(failed)
                            except Exception as _fm_err:
                                # Non-serializable params or other metadata issue must
                                # not escape the skip_on_error boundary — just warn.
                                logger.warning(
                                    "Failed to write failure markers for batch %d: %s",
                                    batch_idx, _fm_err,
                                )
                    else:
                        raise
                finally:
                    if _pbar is not None:
                        _pbar.update(1)

            try:
                # use return_exceptions=True so one failing batch
                # doesn't cancel all other in-flight tasks — critical when
                # skip_on_error=True (each batch handles its own exception via
                # _dispatch_batch's except block), but also safer in general.
                gather_results = await asyncio.gather(
                    *[_dispatch_batch(i) for i in range(n_batches)],
                    return_exceptions=True,
                )
                if _pbar is not None:
                    _pbar.close()
                # Re-raise the first exception that was not swallowed by skip_on_error.
                # When skip_on_error=True, _dispatch_batch never re-raises, so any
                # Exception in gather_results means it propagated from outside the
                # try/except in _dispatch_batch (unexpected error).
                # PipelineAPIAuthError (401/402/403) is fatal regardless of
                # skip_on_error — auth/billing failures must surface, not silently
                # produce empty results.
                first_exc = None
                for gr in gather_results:
                    if isinstance(gr, PipelineAPIAuthError):
                        first_exc = gr
                        break
                    if isinstance(gr, Exception):
                        if not self.skip_on_error:
                            first_exc = gr
                            break
                if first_exc is not None:
                    raise first_exc
                # Post-gather preflight: validate extraction keys against the first
                # captured response.  Raises before WorkingSet construction so callers
                # see a specific error instead of an empty output set.
                if (
                    self.action in ("predict", "score")
                    and self._resolved
                    and _preflight_capture
                ):
                    _pf_r = _preflight_capture[0]
                    for _spec in self._resolved:
                        if _spec.response_key not in _pf_r:
                            raise ValueError(
                                f"PredictionStage preflight: extraction key "
                                f"{_spec.response_key!r} (column={_spec.column!r}) "
                                f"not found in API response for model "
                                f"{self.model_name!r}. "
                                f"Available keys: {list(_pf_r.keys())}. "
                                f"Full response sample: {_pf_r}"
                            )
                computed = len(uncached_ids)
                if verbose:
                    print(
                        f"  Completed: {computed} sequences "
                        f"in {n_batches} batches "
                        f"(max {self.max_concurrent} concurrent)"
                    )

            except Exception as e:
                if verbose:
                    print(f"  Error during prediction: {e}")
                raise
            finally:
                # shut down the local api client only. Do not touch self._api_client —
                # it is dead state in process_ws() (only used in legacy process() path).
                if api is not None:
                    await api.shutdown()

        # Return WorkingSet: IDs that have ALL resolved columns stored in DuckDB.
        # Intersect across columns so that a sequence missing any column is excluded.
        if self.action in ("predict", "score"):
            candidate_ids = set(input_ids)
            for spec in self._resolved:
                have_col = set(datastore.get_sequence_ids_with_prediction(
                    list(candidate_ids), spec.column, self.model_name
                ))
                candidate_ids &= have_col
                if not candidate_ids:
                    break
            # For structure-only or matrix-only stages (no scalar extractions),
            # gate on the structures/predictions table so sequences without
            # output don't pass through.
            if not self._resolved and self._structure_output:
                have_struct = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM structures WHERE model_name = ?",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
                candidate_ids &= have_struct
            if not self._resolved and self._matrix_extraction:
                have_matrix = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM predictions WHERE model_name = ?",
                        [self.model_name],
                    ).df()["sequence_id"].tolist()
                )
                candidate_ids &= have_matrix
            ws_out = WorkingSet.from_ids(candidate_ids)
        elif self.action == "encode":
            _emb_layer = (
                self._embedding_extractor.layer
                if isinstance(self._embedding_extractor, EmbeddingSpec) and self._embedding_extractor.layer is not None
                else None
            )
            if _emb_layer is not None:
                emb_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer = ?",
                        [self.model_name, _emb_layer],
                    ).df()["sequence_id"].tolist()
                )
            else:
                # output gate must match cache check — scope to layer IS NULL
                # when _emb_layer is None, not match ANY layer.
                emb_ids = set(
                    datastore.conn.execute(
                        "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ? AND layer IS NULL",
                        [self.model_name],
                    )
                    .df()["sequence_id"]
                    .tolist()
                )
            ws_out = WorkingSet.from_ids(set(input_ids) & emb_ids)
        else:
            ws_out = ws

        return ws_out, StageResult(
            stage_name=self.name,
            input_count=input_count,
            output_count=len(ws_out),
            cached_count=cached_count,
            computed_count=len(uncached_ids),
            filtered_count=input_count - len(ws_out),
        )


class FilterStage(Stage):
    """
    Generic filtering stage.

    Args:
        name: Stage name
        filter_func: Filter function or BaseFilter instance
    """

    def __init__(self, name: str, filter_func: Union["BaseFilter", Callable[..., Any]], **kwargs):
        super().__init__(name=name, **kwargs)
        self.filter_func = filter_func
        # Check if filter requires complete data (for streaming)
        self.requires_complete_data = getattr(
            filter_func,
            "requires_complete_data",
            True,  # Default: safe (batch)
        )
        # Cache for resample=False filters lives on the stage, not on the datastore.
        self._cached_ws_ids: Optional[list[int]] = None

    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        # raw callables don't have to_spec() — fail loudly instead of AttributeError.
        if not hasattr(self.filter_func, "to_spec"):
            raise NotImplementedError(
                f"FilterStage '{self.name}' uses a raw callable filter_func and cannot "
                "be serialized. Use a BaseFilter subclass instead."
            )
        spec = {
            "type": "FilterStage",
            "name": self.name,
            "filter_spec": self.filter_func.to_spec(),
            "depends_on": self.depends_on,
        }
        # include model_name if the stage tracks which model it filters on.
        if hasattr(self, "model_name") and self.model_name is not None:
            spec["model_name"] = self.model_name
        return spec

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Apply filter to DataFrame and return the filtered result."""
        start_count = len(df)

        print(f"  Applying filter: {self.filter_func}")
        if df.empty:
            df_filtered = df.copy()
        else:
            df_filtered = self.filter_func(
                df
            ).copy()  # .copy() prevents in-place mutation of input df

        filtered_count = start_count - len(df_filtered)

        print(f"  Filtered out: {filtered_count}/{start_count}")
        print(f"  Remaining: {len(df_filtered)}")

        # Persist filter results for resume support
        run_id = kwargs.get("run_id")
        if run_id and "sequence_id" in df_filtered.columns:
            datastore.save_filter_results(
                run_id, self.name, df_filtered["sequence_id"].tolist()
            )

        return df_filtered, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df_filtered),
            filtered_count=filtered_count,
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Apply filter using WorkingSet.

        Two execution paths — chosen at call time, not as a "fallback":

        1. **SQL-native** (zero materialization): filters that implement
           ``to_sql()`` return a complete SELECT scoped to the working set.
           DuckDB executes it directly; no DataFrame is ever created.

        2. **DataFrame-based**: filters that *cannot* be expressed in SQL
           (e.g. HammingDistanceFilter, CustomFilter) must materialize a
           DataFrame.  This is the correct path for those filters, not a
           fallback.
        """
        verbose = kwargs.get("verbose", True)
        input_count = len(ws)

        if verbose:
            print(f"  Applying filter: {self.filter_func}")

        if not ws:
            return ws, StageResult(
                stage_name=self.name,
                input_count=0,
                output_count=0,
                filtered_count=0,
            )

        # Determine execution path: SQL-native or DataFrame-based.
        # For prediction-backed filters (ThresholdFilter, RankingFilter), look up the
        # model_name from the column registry so the SQL is scoped to the correct model
        # and produces the same results as the DataFrame path.
        sql_query = None
        ws_table_name = "_filter_ws"
        if isinstance(self.filter_func, BaseFilter):
            col_name = getattr(self.filter_func, "column", None)
            model_name_for_sql: Optional[str] = None
            if col_name is not None:
                entry = datastore.get_column_registry_entry(col_name)
                if entry:
                    model_name_for_sql = entry.get("model_name")
            # Mint a unique ws_table name so two SQL filters at the same DAG
            # level don't race on a shared registration.
            ws_table_name = datastore.make_filter_ws_name()
            sql_query = self.filter_func.to_sql(
                ws_table=ws_table_name, model_name=model_name_for_sql
            )

        if sql_query is not None:
            surviving_ids = datastore.execute_filter_sql(
                list(ws.sequence_ids), sql_query, ws_table_name=ws_table_name
            )
            ws_out = WorkingSet.from_ids(surviving_ids)
        else:
            # DataFrame-based path — required for non-SQL filters
            # if resample=False and already have a working set cached, respect it.
            _resample = getattr(self.filter_func, "resample", True)
            if not _resample and self._cached_ws_ids is not None:
                cached_ids = self._cached_ws_ids
                # Intersect with current ws so we only keep IDs still in the pipeline.
                ws_out = WorkingSet.from_ids(
                    [sid for sid in cached_ids if sid in ws.sequence_ids]
                )
            else:
                df = datastore.materialize_working_set(ws)
                df_filtered = self.filter_func(df).copy()
                # raw callable filters must return a DataFrame with sequence_id.
                if not isinstance(self.filter_func, BaseFilter):
                    if "sequence_id" not in df_filtered.columns:
                        raise ValueError(
                            f"Filter function in stage '{self.name}' must return a "
                            "DataFrame with a 'sequence_id' column, but the returned "
                            f"DataFrame has columns: {list(df_filtered.columns)}"
                        )
                if "sequence_id" in df_filtered.columns:
                    ws_out = WorkingSet.from_ids(df_filtered["sequence_id"].tolist())
                    # Cache for resample=False on future invocations.
                    if not _resample:
                        self._cached_ws_ids = list(ws_out.sequence_ids)
                else:
                    # raise instead of silently passing all sequences through.
                    raise RuntimeError(
                        f"Filter stage '{self.name}': materialized DataFrame is missing "
                        "'sequence_id' column after filtering. The filter function must "
                        "return a DataFrame with 'sequence_id' column."
                    )

        filtered_count = input_count - len(ws_out)

        if verbose:
            print(f"  Filtered out: {filtered_count}/{input_count}")
            print(f"  Remaining: {len(ws_out)}")

        # Persist filter results for resume support
        run_id = kwargs.get("run_id")
        if run_id:
            datastore.save_filter_results(
                run_id, self.name, list(ws_out.sequence_ids)
            )

        return ws_out, StageResult(
            stage_name=self.name,
            input_count=input_count,
            output_count=len(ws_out),
            filtered_count=filtered_count,
        )


class CofoldingPredictionStage(Stage):
    """
    Prediction stage for co-folding models (Boltz2, Chai-1).

    Each sequence in the pipeline DataFrame is used as the **primary** entity
    (chain) in a multi-molecule folding request.  Additional static entities —
    ligands, cofactors, DNA/RNA, or other protein chains — are injected via
    ``static_entities`` and held constant for every sequence.

    The caller is responsible for providing the correct molecule field names and
    params for the target model.  Boltz2 and Chai-1 both expect an item shaped
    like ``{'molecules': [{'id': ..., 'type': ..., 'sequence': ...}, ...]}``.

    Args:
        name: Stage name.
        model_name: BioLM model slug (``'boltz2'``, ``'chai1'``).
        action: API action (default ``'predict'``).
        prediction_type: Column name written for the confidence score
            (default ``'structure'``).
        sequence_chain_id: Chain ID / name assigned to the pipeline's primary
            sequence in the molecules list.
        sequence_entity_type: Molecule type for the primary sequence
            (``'protein'``, ``'dna'``, ``'rna'``).
        static_entities: List of :class:`~biolm.pipeline.generative.FoldingEntity`
            objects appended to every molecules list after the primary chain.
        params: Model-specific params dict passed directly to the API
            (e.g. ``{'recycling_steps': 3, 'sampling_steps': 20}`` for Boltz).
        batch_size: Sequences per API call (default 1; co-folding models are
            typically limited to batch size 1).

    Example::

        from biolm.pipeline import FoldingEntity

        pipeline.add_cofolding_prediction(
            model_name='boltz2',
            static_entities=[
                FoldingEntity(id='L', entity_type='ligand', smiles='c1ccccc1'),
            ],
            params={'recycling_steps': 3, 'sampling_steps': 20},
            depends_on=['filter_top50'],
        )
    """

    merge_mode = "union"

    def __init__(
        self,
        name: str,
        model_name: str,
        action: str = "predict",
        prediction_type: str = "structure",
        sequence_chain_id: str = "A",
        sequence_entity_type: str = "protein",
        static_entities=None,  # List[FoldingEntity] — typed at call site
        params: Optional[dict] = None,
        batch_size: int = 1,
        item_columns: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(name=name, model_name=model_name, **kwargs)
        self.action = action
        self.prediction_type = prediction_type
        self.sequence_chain_id = sequence_chain_id
        self.sequence_entity_type = sequence_entity_type
        self.static_entities = static_entities or []
        self.params = params or {}
        self.batch_size = batch_size
        # item_columns maps API field names → DataFrame column names.
        # When None, defaults to reading row["sequence"] for the primary chain.
        self.item_columns = item_columns
        self._api_client = None

    def _build_item(self, sequence: str, row: Optional[dict] = None) -> dict:
        """Build the multi-molecule item dict for a single pipeline sequence.

        AP-10: When item_columns is set, builds the primary chain dict using
        the column mapping from ``row`` instead of the plain ``sequence`` string.
        """
        if self.item_columns is not None and row is not None:
            # Use item_columns mapping to build the primary chain dict.
            primary = {"id": self.sequence_chain_id, "type": self.sequence_entity_type}
            for api_field, col in self.item_columns.items():
                primary[api_field] = row.get(col, "")
        else:
            primary = {
                "id": self.sequence_chain_id,
                "type": self.sequence_entity_type,
                "sequence": sequence,
            }
        molecules = [primary]
        for entity in self.static_entities:
            mol: dict[str, Any] = {"id": entity.id, "type": entity.entity_type}
            if entity.sequence is not None:
                mol["sequence"] = entity.sequence
            if entity.smiles is not None:
                mol["smiles"] = entity.smiles
            if entity.ccd is not None:
                mol["ccd"] = entity.ccd
            molecules.append(mol)
        return {"molecules": molecules}

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Run co-folding prediction for each sequence in *df*."""
        start_count = len(df)
        computed = 0
        # B4 fix: save full input so cached sequences are included in the returned frame.
        # df is filtered to uncached-only for the API loop; full_df is returned with
        # predictions merged back from DuckDB at the end (matching PredictionStage pattern).
        full_df = df.copy()
        df = full_df.copy()  # working copy — may be filtered to uncached subset below

        # skip already-cached sequences to avoid redundant API calls.
        if "sequence_id" in full_df.columns:
            all_input_ids = full_df["sequence_id"].tolist()
            cached_ids = set(
                datastore.get_uncached_sequence_ids(all_input_ids, self.prediction_type, self.model_name)
            )
            # get_uncached_sequence_ids returns IDs that are NOT cached — invert to get cached
            cached_set = set(all_input_ids) - cached_ids
            if cached_set:
                df = full_df[~full_df["sequence_id"].isin(cached_set)].copy()
                if kwargs.get("verbose", True):
                    print(f"  CofoldingStage '{self.name}': {len(cached_set)} sequences already cached, skipping.")

        # create client with semaphore; shut it down in finally.
        api = None
        try:
            api = BioLMApiClient(
                self.model_name,
                semaphore=asyncio.Semaphore(self.batch_size),
                
            )

            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i : i + self.batch_size]
                items = [
                    self._build_item(row.get("sequence", ""), row=row.to_dict())
                    for _, row in batch.iterrows()
                ]

                results = await getattr(api, self.action)(
                    items=items, params=self.params
                )

                # B12 fix: guard against positional drift from length-mismatched responses
                if len(results) != len(batch):
                    raise ValueError(
                        f"CofoldingPredictionStage.process(): API returned {len(results)} results "
                        f"for {len(batch)} items — length mismatch; treating batch as failed"
                    )

                for _j, (result, (idx, row)) in enumerate(
                    zip(results, batch.iterrows())
                ):
                    if not isinstance(result, dict):
                        continue
                    cif = result.get("cif", "")
                    conf_data = result.get("confidence", {})
                    confidence = (
                        conf_data.get("confidence_score")
                        if isinstance(conf_data, dict)
                        else None
                    )
                    seq_id = row.get("sequence_id")
                    if seq_id is not None and cif:
                        datastore.add_structure(
                            seq_id,
                            model_name=self.model_name,
                            structure_str=cif,
                            format="cif",
                        )
                    if seq_id is not None and confidence is not None:
                        datastore.add_prediction(
                            seq_id,
                            self.prediction_type,
                            self.model_name,
                            float(confidence),
                            metadata={"cif_stored": bool(cif)},
                        )
                    df.at[idx, self.prediction_type] = confidence
                    if cif:
                        df.at[idx, "cif"] = cif
                    computed += 1

        finally:
            if api is not None:
                await api.shutdown()

        # B4 fix: merge predictions back into full_df for all sequence_ids (cached + new).
        # Mirrors the PredictionStage.process() merge-back pattern.
        if "sequence_id" in full_df.columns:
            all_seq_ids = full_df["sequence_id"].tolist()
            pred_df = datastore.get_predictions_bulk(
                all_seq_ids, self.prediction_type, self.model_name
            )
            if not pred_df.empty:
                val_map = dict(zip(pred_df["sequence_id"], pred_df["value"]))
                full_df[self.prediction_type] = full_df["sequence_id"].map(val_map)
            else:
                full_df[self.prediction_type] = None

        # Mirror PredictionStage.process() all_present mask: drop rows where
        # confidence is NaN so process() semantics match process_ws() output.
        if self.prediction_type in full_df.columns:
            full_df = full_df[full_df[self.prediction_type].notna()].copy()

        return full_df, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(full_df),
            computed_count=computed,
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Run co-folding prediction using WorkingSet — no DataFrame transport."""
        input_count = len(ws)
        computed = 0

        # skip already-cached sequences.
        input_ids = list(ws.sequence_ids)
        uncached_ids = set(
            datastore.get_uncached_sequence_ids(input_ids, self.prediction_type, self.model_name)
        )
        if kwargs.get("verbose", True) and len(uncached_ids) < len(input_ids):
            print(
                f"  CofoldingStage '{self.name}': "
                f"{len(input_ids) - len(uncached_ids)} sequences already cached, skipping."
            )
        ids_to_process = [sid for sid in input_ids if sid in uncached_ids]

        id_seq_pairs = datastore.get_sequences_for_ids(ids_to_process)

        # create client with semaphore and shut it down in finally.
        api = None
        try:
            api = BioLMApiClient(
                self.model_name,
                semaphore=asyncio.Semaphore(self.batch_size),
                
            )

            for i in range(0, len(id_seq_pairs), self.batch_size):
                chunk = id_seq_pairs[i : i + self.batch_size]
                # if item_columns is set, fetch row dicts with required columns.
                if self.item_columns:
                    col_names = list(self.item_columns.values())
                    chunk_ids = [sid for sid, _seq in chunk]
                    col_map = datastore.get_sequences_for_ids_with_columns(chunk_ids, col_names)
                    items = [
                        self._build_item(seq, row=col_map.get(sid, {}))
                        for sid, seq in chunk
                    ]
                else:
                    items = [self._build_item(seq) for _sid, seq in chunk]

                results = await getattr(api, self.action)(
                    items=items, params=self.params
                )

                # B13 fix: guard against positional drift from length-mismatched responses
                if len(results) != len(chunk):
                    raise ValueError(
                        f"CofoldingPredictionStage.process_ws(): API returned {len(results)} results "
                        f"for {len(chunk)} items — length mismatch; treating batch as failed"
                    )

                for (seq_id, _seq), result in zip(chunk, results):
                    if not isinstance(result, dict):
                        continue
                    cif = result.get("cif", "")
                    conf_data = result.get("confidence", {})
                    confidence = (
                        conf_data.get("confidence_score")
                        if isinstance(conf_data, dict)
                        else None
                    )
                    if cif:
                        datastore.add_structure(
                            seq_id,
                            model_name=self.model_name,
                            structure_str=cif,
                            format="cif",
                        )
                    if confidence is not None:
                        datastore.add_prediction(
                            seq_id,
                            self.prediction_type,
                            self.model_name,
                            float(confidence),
                            metadata={"cif_stored": bool(cif)},
                        )
                    computed += 1

        finally:
            if api is not None:
                await api.shutdown()

        # Return IDs that got predictions
        ids_with_pred = datastore.get_sequence_ids_with_prediction(
            list(ws.sequence_ids), self.prediction_type, self.model_name
        )
        ws_out = WorkingSet.from_ids(ids_with_pred)

        return ws_out, StageResult(
            stage_name=self.name,
            input_count=input_count,
            output_count=len(ws_out),
            computed_count=computed,
        )

    def to_spec(self) -> dict:
        """Serialize to a dict for pipeline definition persistence.

        Note: ``static_entities`` (FoldingEntity objects) are not serializable
        and are omitted.  A reconstructed pipeline will not have static_entities
        and must have them re-attached manually after ``from_db()``.
        """
        if self.static_entities:
            import warnings
            warnings.warn(
                f"CofoldingPredictionStage '{self.name}' has static_entities which "
                "cannot be serialized for from_db() recovery.  Re-attach them manually "
                "after reconstructing the pipeline.",
                UserWarning,
                stacklevel=2,
            )
        return {
            "type": "CofoldingPredictionStage",
            "name": self.name,
            "model_name": self.model_name,
            "action": self.action,
            "prediction_type": self.prediction_type,
            "sequence_chain_id": self.sequence_chain_id,
            "sequence_entity_type": self.sequence_entity_type,
            "static_entities": [],  # not serializable
            "params": self.params,
            "batch_size": self.batch_size,
            "depends_on": self.depends_on,
        }


class ClusteringStage(Stage):
    """
    Sequence clustering stage.

    Args:
        name: Stage name
        method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        n_clusters: Number of clusters (for kmeans/hierarchical)
        similarity_metric: How to measure similarity ('hamming', 'embedding')
        embedding_model: Model to use for embeddings (if similarity_metric='embedding')
    """

    def __init__(
        self,
        name: str,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        similarity_metric: str = "hamming",
        embedding_model: Optional[str] = None,
        max_sample: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.method = method
        self.n_clusters = n_clusters
        self.similarity_metric = similarity_metric
        self.embedding_model = embedding_model
        self.max_sample = max_sample
        # Strip Stage base-class kwargs so they aren't forwarded to SequenceClusterer
        _stage_keys = {"cache_key", "depends_on", "model_name", "max_concurrent"}
        self.cluster_kwargs = {k: v for k, v in kwargs.items() if k not in _stage_keys}

    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        return {
            "type": "ClusteringStage",
            "name": self.name,
            "method": self.method,
            "n_clusters": self.n_clusters,
            "similarity_metric": self.similarity_metric,
            "embedding_model": self.embedding_model,
            "max_sample": self.max_sample,
            "depends_on": self.depends_on,
            # PD-09: include cluster_kwargs so custom algorithm params (e.g. eps
            # for DBSCAN) are restored when reconstructing from a definition.
            "cluster_kwargs": self.cluster_kwargs,
        }

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Cluster sequences, add cluster columns, and return the enriched DataFrame."""
        import warnings

        from biolm.pipeline.clustering import (
            LARGE_DATASET_THRESHOLD,
            VERY_LARGE_DATASET_THRESHOLD,
            SequenceClusterer,
        )

        start_count = len(df)
        df = df.copy()

        n = start_count
        if self.similarity_metric == "hamming":
            if n > VERY_LARGE_DATASET_THRESHOLD:
                raise ValueError(
                    f"ClusteringStage: {n} sequences with hamming distance requires "
                    f"~{n**2 // 1_000_000}M comparisons. Use similarity_metric='embedding' "
                    f"or max_sample<={VERY_LARGE_DATASET_THRESHOLD}."
                )
            elif n > LARGE_DATASET_THRESHOLD:
                warnings.warn(
                    f"ClusteringStage: {n} sequences with hamming distance is O(n²) expensive. "
                    f"Consider similarity_metric='embedding' or max_sample={LARGE_DATASET_THRESHOLD}.",
                    ResourceWarning,
                    stacklevel=2,
                )

        _verbose = kwargs.get("verbose", True)
        if _verbose:
            print(f"  Clustering {start_count} sequences using {self.method}...")

        sequences = df["sequence"].tolist()

        # Load embeddings if needed — bulk fetch, not per-sequence
        embeddings = None
        if self.similarity_metric == "embedding":
            if self.embedding_model is None:
                raise ValueError(
                    "embedding_model required when similarity_metric='embedding'"
                )

            if _verbose:
                print(f"  Loading embeddings from {self.embedding_model}...")
            # Bug #6 fix: bulk fetch in a single JOIN instead of N per-sequence queries
            seq_ids = df["sequence_id"].tolist()
            emb_map = datastore.get_embeddings_bulk(
                seq_ids, model_name=self.embedding_model
            )
            embeddings_list = []
            for seq_id, seq in zip(seq_ids, sequences):
                emb_array = emb_map.get(int(seq_id))
                if emb_array is None:
                    raise ValueError(
                        f"No embedding found for sequence_id {seq_id} ({seq[:20]}...)"
                    )
                embeddings_list.append(emb_array)

            embeddings = np.stack(embeddings_list)

        # Perform clustering
        clusterer_kwargs = dict(self.cluster_kwargs.items())
        if self.max_sample is not None:
            clusterer_kwargs["max_sample"] = self.max_sample
        clusterer = SequenceClusterer(
            method=self.method,
            n_clusters=self.n_clusters,
            similarity_metric=self.similarity_metric,
            **clusterer_kwargs,
        )

        result = clusterer.cluster(sequences, embeddings)

        # Add cluster assignments to output DataFrame
        # reset_index so positional indexing via iloc is safe,
        # then use iloc to avoid label-based loc mismatches on filtered DataFrames.
        df = df.reset_index(drop=True).copy()
        df["cluster_id"] = result.cluster_ids
        df["is_centroid"] = False
        df.iloc[result.centroid_indices, df.columns.get_loc("is_centroid")] = True

        print(f"  Found {result.n_clusters} clusters")
        if result.silhouette_score is not None:
            print(f"  Silhouette score: {result.silhouette_score:.3f}")
        if result.davies_bouldin_score is not None:
            print(f"  Davies-Bouldin score: {result.davies_bouldin_score:.3f}")

        # Store summary metadata — convert numpy int keys to Python int for JSON compat
        cluster_sizes_safe = (
            {int(k): int(v) for k, v in result.cluster_sizes.items()}
            if result.cluster_sizes is not None
            else None
        )
        meta = {
            "method": self.method,
            "n_clusters": result.n_clusters,
            "silhouette_score": (
                float(result.silhouette_score) if result.silhouette_score is not None else None
            ),
            "davies_bouldin_score": (
                float(result.davies_bouldin_score) if result.davies_bouldin_score is not None else None
            ),
            "cluster_sizes": cluster_sizes_safe,
        }
        datastore.set_pipeline_metadata(f"clustering_{self.name}", meta)

        # Store per-sequence assignments for resume support
        # key by int(seq_id) not by sequence string (avoids duplicate key
        # collisions for sequences that appear more than once in the input).
        seq_ids = df["sequence_id"].tolist() if "sequence_id" in df.columns else list(range(len(sequences)))
        assignments = {
            int(sid): int(cid) for sid, cid in zip(seq_ids, result.cluster_ids.tolist())
        }
        datastore.set_pipeline_metadata(
            f"clustering_{self.name}_assignments", assignments
        )

        return df, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=start_count,
            computed_count=start_count,
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Cluster sequences — materializes internally for scikit-learn.

        Clustering doesn't filter rows (all sequences survive), so the
        returned WorkingSet is the same as the input.  Cluster assignments
        are stored as predictions (prediction_type ``'cluster_id'``) so that
        ``materialize_working_set()`` includes them in ``get_final_data()``.
        """
        verbose = kwargs.get("verbose", True)
        df = datastore.materialize_working_set(ws, include_predictions=False)
        df_out, result = await self.process(df, datastore, **kwargs)

        # Store cluster assignments as predictions so get_final_data() picks them up.
        # Use prediction_type=self.name (the stage name) rather than the fixed string
        # "cluster_id" — this makes each ClusteringStage produce a uniquely-named
        # column in the materialized DataFrame and prevents MAX() pivot ambiguity
        # when two clustering stages coexist in the same pipeline.
        if "cluster_id" in df_out.columns and "sequence_id" in df_out.columns:
            batch_df = df_out[["sequence_id", "cluster_id", "is_centroid"]].dropna(subset=["cluster_id"])
            batch = [
                {
                    "sequence_id": int(sid),
                    "prediction_type": self.name,  # unique per stage
                    "model_name": self.name,
                    "value": float(cid),
                }
                for sid, cid in zip(batch_df["sequence_id"], batch_df["cluster_id"])
            ]
            if batch:
                datastore.add_predictions_batch(batch)

            # also store is_centroid as a separate prediction column
            if "is_centroid" in batch_df.columns:
                centroid_batch = [
                    {
                        "sequence_id": int(sid),
                        "prediction_type": f"{self.name}_centroid",
                        "model_name": self.name,
                        "value": float(is_c),
                    }
                    for sid, is_c in zip(batch_df["sequence_id"], batch_df["is_centroid"])
                ]
                if centroid_batch:
                    datastore.add_predictions_batch(centroid_batch)

        return ws, result


class DataPipeline(BasePipeline):
    """
    Pipeline for processing existing sequences from files or lists.

    Load sequences from CSV/FASTA/lists and run predictions/filtering.

    Args:
        sequences: Input sequences (list of strings, DataFrame, or file path)
        datastore: DataStore instance or path
        run_id: Unique run ID
        output_dir: Output directory
        resume: Resume from previous run
        verbose: Enable verbose output
        diff_mode: If True, merge new sequences with existing cached results.
                  Only computes predictions for uncached sequences. Use get_merged_results()
                  or query_results() to efficiently access combined data without loading
                  millions of rows into memory (SQL-based queries).

    Example:
        >>> # Standard mode
        >>> pipeline = DataPipeline(sequences='sequences.csv')
        >>> pipeline.add_prediction('esmfold', extractions='mean_plddt', columns='plddt')
        >>> pipeline.add_filter(ThresholdFilter('plddt', min_value=70))
        >>> results = pipeline.run()

        >>> # Diff mode - add new sequences to existing pipeline (SQL-based, efficient)
        >>> pipeline = DataPipeline(sequences='new_sequences.csv', diff_mode=True)
        >>> pipeline.add_prediction('esmfold', extractions='mean_plddt', columns='plddt')
        >>> results = pipeline.run()
        >>> # Efficiently query specific data (doesn't load all millions of rows!)
        >>> high_quality = pipeline.query_results("s.length > 100 AND p.value > 70")
        >>> # Or get merged results with filters
        >>> merged = pipeline.get_merged_results(prediction_types=['plddt', 'tm'])
    """

    def __init__(
        self,
        sequences: Union[list[str], pd.DataFrame, str, Path] = None,
        diff_mode: bool = False,
        input_columns: Optional[list[str]] = None,
        **kwargs,
    ):
        resume = kwargs.get("resume", False)
        if sequences is None and not resume:
            raise ValueError(
                "sequences is required unless resume=True "
                "(use DataPipeline.from_db() to reconstruct from an existing database)."
            )
        # Build InputSchema if input_columns specified
        if input_columns is not None:
            kwargs["input_schema"] = InputSchema(columns=input_columns)
        super().__init__(**kwargs)
        self.input_sequences = sequences
        self.diff_mode = diff_mode
        self.input_columns = input_columns

    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """Load sequences into DataFrame."""

        if self.input_sequences is None:
            # resume=True + no sequences: pull existing IDs from DuckDB directly.
            # use SELECT * to capture multi-column inputs (heavy_chain,
            # light_chain, etc.) rather than hardcoding sequence_id + sequence only.
            # warn that shared databases will load ALL sequences without run_id scoping.
            import warnings
            warnings.warn(
                "DataPipeline(sequences=None, resume=True) loads all sequences from the database. "
                "For shared databases, specify run_id to scope correctly.",
                UserWarning,
                stacklevel=3,
            )
            all_rows_df = self.datastore.conn.execute(
                "SELECT * FROM sequences ORDER BY sequence_id"
            ).df()
            if all_rows_df.empty:
                raise ValueError(
                    "No sequences provided and no sequences found in the datastore. "
                    "Provide sequences or use a populated datastore."
                )
            return all_rows_df

        # raise early if the user passed an empty list.
        if isinstance(self.input_sequences, list) and len(self.input_sequences) == 0:
            raise ValueError(
                "sequences list is empty. Provide at least one sequence."
            )

        # Convert to DataFrame
        if isinstance(self.input_sequences, list):
            if self.input_sequences and isinstance(self.input_sequences[0], dict):
                df = pd.DataFrame(self.input_sequences)
            else:
                df = pd.DataFrame({"sequence": self.input_sequences})

        elif isinstance(self.input_sequences, pd.DataFrame):
            df = self.input_sequences.copy()

        elif isinstance(self.input_sequences, (str, Path)):
            path = Path(self.input_sequences)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if path.suffix == ".csv":
                df = pd.read_csv(path)

            elif path.suffix in [".fasta", ".fa", ".faa"]:
                # Simple FASTA parser
                sequences = []
                current_seq = []

                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(">"):
                            if current_seq:
                                sequences.append("".join(current_seq))
                                current_seq = []
                        else:
                            current_seq.append(line)

                    if current_seq:
                        sequences.append("".join(current_seq))

                if not sequences:
                    raise ValueError(f"No sequences found in FASTA file: {path}")
                df = pd.DataFrame({"sequence": sequences})

            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        else:
            raise TypeError(
                f"Unsupported type for sequences: {type(self.input_sequences)}"
            )

        # ----- Schema compatibility validation -----------------------------------
        ds_has_data = self.datastore.conn.execute(
            "SELECT COUNT(*) FROM sequences"
        ).fetchone()[0] > 0

        if ds_has_data:
            ds_extra = self.datastore.get_existing_input_columns()
            if self.input_columns is not None and not ds_extra:
                raise ValueError(
                    "Cannot use multi-column input (input_columns="
                    f"{self.input_columns}) on a datastore that already "
                    "contains single-column sequences. Use a fresh datastore "
                    "or match the existing schema."
                )
            if self.input_columns is None and ds_extra:
                raise ValueError(
                    "Cannot use single-column input on a datastore that "
                    f"already contains multi-column data (columns: "
                    f"{sorted(ds_extra)}). Use a fresh datastore or provide "
                    "input_columns matching the existing schema."
                )
            # when both pipelines use multi-column mode, also
            # verify the *same set* of columns is used.  A mismatch would silently
            # produce wrong hashes and leave sequences forever uncached.
            if self.input_columns is not None and ds_extra:
                new_cols = sorted(self.input_columns)
                existing_cols = sorted(ds_extra)
                if new_cols != existing_cols:
                    raise ValueError(
                        f"Column set mismatch: this pipeline uses input_columns="
                        f"{new_cols} but the datastore already contains multi-column "
                        f"data with columns {existing_cols}. Use a fresh datastore "
                        "or use the same input_columns as the existing data."
                    )

        # ----- Multi-column input path ------------------------------------------
        if self.input_columns is not None:
            # Validate that all input columns are present
            missing = [c for c in self.input_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"input_columns {missing} not found in DataFrame. "
                    f"Available: {list(df.columns)}"
                )

            # Ensure the sequences table has the extra columns
            self.datastore.ensure_input_columns(self.input_columns)

            # Deduplicate across all input columns
            initial_count = len(df)
            df = df.drop_duplicates(subset=self.input_columns).reset_index(drop=True)
            deduplicated_count = initial_count - len(df)
            if deduplicated_count > 0 and self.verbose:
                print(f"Deduplicated {deduplicated_count} rows ({len(df)} unique)")

            # Use the multi-column add_sequences_batch path
            df["sequence_id"] = self.datastore.add_sequences_batch(
                input_df=df, input_columns=self.input_columns
            )

            # Also persist non-input extra columns as sequence attributes
            extra_cols = [
                c
                for c in df.columns
                if c not in self.input_columns
                and c not in ("sequence", "sequence_id", "length", "hash")
            ]
            if extra_cols:
                seq_ids = df["sequence_id"].tolist()
                for col in extra_cols:
                    self.datastore.store_sequence_attributes(
                        seq_ids, col, df[col].tolist()
                    )

            return df

        # ----- Legacy single-column path ----------------------------------------
        if "sequence" not in df.columns:
            raise ValueError("DataFrame must have 'sequence' column")

        # Deduplicate before inserting into datastore
        initial_count = len(df)
        df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
        deduplicated_count = initial_count - len(df)
        if deduplicated_count > 0 and self.verbose:
            print(f"Deduplicated {deduplicated_count} sequences ({len(df)} unique)")

        # DS-05 fix: count existing sequences BEFORE add_sequences_batch() so
        # that the count reflects sequences already in the DB, not newly-added
        # ones (which would always be "existing" if counted after insertion).
        if self.diff_mode:
            existing_count = self.datastore.count_matching_sequences(
                df["sequence"].tolist()
            )
            new_count = len(df) - existing_count
            if self.verbose:
                print(f"Diff mode: {existing_count} sequences already in datastore")
                print(f"Diff mode: {new_count} new sequences to process")

        # Batch-add all sequences in one vectorized call instead of N individual calls
        df["sequence_id"] = self.datastore.add_sequences_batch(df["sequence"].tolist())

        # Persist extra columns (e.g. heavy_chain, light_chain) as sequence attributes
        # so they're available to stages that need them (item_columns in PredictionStage)
        extra_cols = [
            c for c in df.columns if c not in ("sequence", "sequence_id", "length", "hash")
        ]
        if extra_cols:
            seq_ids = df["sequence_id"].tolist()
            for col in extra_cols:
                self.datastore.store_sequence_attributes(
                    seq_ids, col, df[col].tolist()
                )

        return df

    def get_merged_results(
        self,
        prediction_types: Optional[list[str]] = None,
        sequence_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get results merged with existing cached data (for diff mode).

        This method is SQL-based and efficient - it doesn't load millions of rows.
        Instead, it queries only the data you need using DuckDB's columnar engine.

        NOTE (Bug #5): ``sequence_filter`` and ``sql_where`` in query_results() are
        interpolated directly into SQL.  These parameters are intended for internal/
        trusted caller use only — never pass untrusted user input to them.

        Args:
            prediction_types: List of prediction types to include (None = all)
            sequence_filter: SQL WHERE clause to filter sequences (e.g., "length > 50").
                             TRUSTED CALLERS ONLY — not safe for untrusted user input.

        Returns:
            DataFrame with requested sequences and predictions

        Example:
            >>> # Get all results (efficient - DuckDB only loads what's needed)
            >>> df = pipeline.get_merged_results()

            >>> # Get only specific predictions (columnar - even faster!)
            >>> df = pipeline.get_merged_results(prediction_types=['tm', 'plddt'])

            >>> # Get sequences matching criteria (predicate pushdown!)
            >>> df = pipeline.get_merged_results(sequence_filter="length > 100")
        """
        if not self.diff_mode:
            # Standard mode: just return final data
            return self.get_final_data()

        # Use DuckDB's efficient query engine
        query = """
            SELECT
                s.sequence_id,
                s.sequence,
                s.length
            FROM sequences s
        """

        if sequence_filter:
            _validate_sql_where(sequence_filter, "sequence_filter")
            query += f" WHERE {sequence_filter}"

        # Execute and load only requested data (columnar scan - fast!)
        df = self.datastore.query(query)

        if len(df) == 0:
            return df

        # Now load predictions for these sequences only
        if prediction_types:
            # Load specific prediction types
            for pred_type in prediction_types:
                self._add_predictions_to_df(df, pred_type)
        else:
            # Load all available prediction types for these sequences
            seq_ids_df = pd.DataFrame({"sequence_id": df["sequence_id"].tolist()})
            self.datastore.conn.register("_merged_seq_ids", seq_ids_df)
            try:
                pred_types_df = self.datastore.conn.execute(
                    """
                    SELECT DISTINCT p.prediction_type
                    FROM predictions p
                    INNER JOIN _merged_seq_ids m ON p.sequence_id = m.sequence_id
                """
                ).df()
            finally:
                self.datastore.conn.unregister("_merged_seq_ids")

            pred_types = pred_types_df["prediction_type"].tolist()
            for pred_type in pred_types:
                self._add_predictions_to_df(df, pred_type)

        if self.verbose:
            print(f"\nDiff mode: Loaded {len(df)} sequences with predictions")

        return df

    def _add_predictions_to_df(
        self,
        df: pd.DataFrame,
        prediction_type: str,
        model_name: Optional[str] = None,
    ):
        """
        Efficiently add a prediction column to DataFrame using DuckDB SQL.
        Modifies df in place by leveraging DuckDB's join capabilities.

        When *model_name* is provided the JOIN is scoped to that model, avoiding
        wrong values when multiple models write the same prediction_type (DS-04).
        Without *model_name* the most-recent value (MAX prediction_id) per
        sequence is used so reporting helpers still get a deterministic result.
        """
        if len(df) == 0:
            return

        # Use DuckDB to efficiently join predictions — register df explicitly first
        self.datastore.conn.register("_pred_df", df)
        try:
            if model_name is not None:
                result = self.datastore.conn.execute(
                    """
                    SELECT
                        _pred_df.sequence_id,
                        p.value
                    FROM _pred_df
                    LEFT JOIN predictions p
                        ON _pred_df.sequence_id = p.sequence_id
                        AND p.prediction_type = ?
                        AND p.model_name = ?
                """,
                    [prediction_type, model_name],
                ).df()
            else:
                # No model_name filter: pick the most recent value per sequence to
                # avoid returning an arbitrary value when multiple models share the
                # same prediction_type key (DS-04 determinism for reporting paths).
                result = self.datastore.conn.execute(
                    """
                    WITH ranked AS (
                        SELECT sequence_id, value,
                               ROW_NUMBER() OVER (
                                   PARTITION BY sequence_id
                                   ORDER BY prediction_id DESC
                               ) AS rn
                        FROM predictions
                        WHERE prediction_type = ?
                    )
                    SELECT
                        _pred_df.sequence_id,
                        ranked.value
                    FROM _pred_df
                    LEFT JOIN ranked
                        ON _pred_df.sequence_id = ranked.sequence_id
                        AND ranked.rn = 1
                """,
                    [prediction_type],
                ).df()
        finally:
            self.datastore.conn.unregister("_pred_df")

        # Create dict for fast lookup
        pred_dict = dict(zip(result["sequence_id"], result["value"]))

        # Add column to DataFrame
        df[prediction_type] = df["sequence_id"].map(pred_dict)

    def query_results(
        self, sql_where: str, columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Query results using SQL WHERE clause (for diff mode with large datasets).

        This leverages DuckDB's vectorized engine for maximum performance.

        NOTE (Bug #5): ``sql_where`` is interpolated directly into SQL.
        TRUSTED CALLERS ONLY — never pass untrusted user input to this parameter.

        Args:
            sql_where: SQL WHERE clause using table aliases:
                      - s.* for sequences table (e.g., "s.length > 100")
                      - Column names directly (no p. prefix needed)
                      TRUSTED CALLERS ONLY — not safe for untrusted user input.
            columns: Columns to include (None = all available)

        Returns:
            DataFrame with matching sequences (only loads what matches!)

        Example:
            >>> # Find long sequences (columnar scan - fast!)
            >>> df = pipeline.query_results("s.length > 200")

            >>> # Complex filter with predictions
            >>> df = pipeline.query_results(
            ...     "s.length > 100",
            ...     columns=['tm', 'plddt']
            ... )
        """
        _validate_sql_where(sql_where, "sql_where")

        # Build efficient DuckDB query
        query = f"""
            SELECT DISTINCT
                s.sequence_id,
                s.sequence,
                s.length
            FROM sequences s
            WHERE {sql_where}
        """

        # DuckDB executes with columnar scans and predicate pushdown
        df = self.datastore.query(query)

        # Add requested columns (predictions)
        if columns and len(df) > 0:
            for col in columns:
                self._add_predictions_to_df(df, col)

        return df

    # ------------------------------------------------------------------
    # Data exploration helpers
    # ------------------------------------------------------------------

    def explore(self) -> dict[str, Any]:
        """Return summary stats for the pipeline's datastore (all via SQL).

        Returns:
            Dict with keys: sequences, embeddings, generated, completed_stages,
            predictions (dict of prediction_type → count).
        """
        row = self.datastore.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM sequences)           AS seq_n,
                (SELECT COUNT(*) FROM embeddings)          AS emb_n,
                (SELECT COUNT(*) FROM generation_metadata) AS gen_n,
                (SELECT COUNT(*) FROM stage_completions
                 WHERE status = 'completed')               AS done_stages
        """
        ).fetchone()

        pred_rows = self.datastore.conn.execute(
            """
            SELECT prediction_type, COUNT(*) AS n
            FROM predictions
            GROUP BY prediction_type
        """
        ).fetchall()

        return {
            "sequences": int(row[0]),
            "embeddings": int(row[1]),
            "generated": int(row[2]),
            "completed_stages": int(row[3]),
            "predictions": {r[0]: int(r[1]) for r in pred_rows},
        }

    def stats(self, stage_name: Optional[str] = None) -> pd.DataFrame:
        """Return per-stage counts from stage_completions.

        Args:
            stage_name: If provided, filter to that stage only.

        Returns:
            DataFrame with columns: stage_name, status, input_count,
            output_count, completed_at.
        """
        if stage_name:
            return self.datastore.query(
                """
                SELECT stage_name, status, input_count, output_count, completed_at
                FROM stage_completions
                WHERE stage_name = ?
                ORDER BY completed_at
            """,
                [stage_name],
            )
        return self.datastore.query(
            """
            SELECT stage_name, status, input_count, output_count, completed_at
            FROM stage_completions
            ORDER BY completed_at
        """
        )

    def query(self, sql: str, params=None) -> pd.DataFrame:
        """Execute arbitrary SQL against the pipeline's DuckDB datastore.

        Args:
            sql: DuckDB SQL query string.
            params: Optional list of query parameters.

        Returns:
            DataFrame with results.

        Example:
            >>> pipeline.query("SELECT * FROM sequences WHERE length > 100")
        """
        return self.datastore.query(sql, params)

    def plot(self, kind: str = "funnel", **kwargs):
        """Convenience wrapper around PipelinePlotter.

        Args:
            kind: One of 'funnel', 'predictions', 'distributions', 'scatter',
                'correlation', 'diversity', 'temperature'.
            **kwargs: Forwarded to the underlying plotter method.
                scatter requires x_col and y_col.
                diversity accepts reference_sequence.
                temperature requires metric_col.
        """
        from biolm.pipeline.visualization import PipelinePlotter

        _kinds = {
            "funnel": "plot_funnel",
            "predictions": "plot_predictions",
            "distributions": "plot_distributions",
            "scatter": "plot_scatter",
            "correlation": "plot_correlation_matrix",
            "diversity": "plot_diversity",
            "temperature": "plot_temperature_scan",
        }
        if kind not in _kinds:
            raise ValueError(
                f"Unknown plot kind '{kind}'. Choose: {', '.join(repr(k) for k in _kinds)}"
            )
        plotter = PipelinePlotter(self)
        return getattr(plotter, _kinds[kind])(**kwargs)

    def add_prediction(
        self,
        model_name: str,
        action: str = "predict",
        extractions: Optional[Union[str, list[Union[str, ExtractionSpec]]]] = None,
        columns: Optional[Union[str, dict[str, str]]] = None,
        params: Optional[dict] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Add a prediction stage.

        Args:
            model_name: BioLM model name
            action: API action ('predict', 'encode', 'score')
            extractions: API response key(s) to extract. Required for
                predict/score actions. Can be a string for a single key or a
                list of strings / ExtractionSpec objects.
            columns: Output column name(s). A string renames a single
                extraction; a dict maps response keys to column names
                (unmapped keys keep their name).
            params: Optional API parameters
            stage_name: Custom stage name (defaults to ``predict_{first_column}``)
            depends_on: List of stage names this depends on. When ``None``
                (default), the stage auto-depends on the last added stage,
                creating a sequential chain.  Pass ``depends_on=[]`` to run
                at the earliest possible level (parallel with other level-0
                stages).

        Example::

            pipeline.add_prediction(
                "temberture-regression",
                extractions="prediction",
                columns="tm",
            )
        """
        # Auto-derive stage name from first column
        if stage_name is None:
            if columns and isinstance(columns, str):
                stage_name = f"predict_{columns}"
            elif extractions and isinstance(extractions, str):
                stage_name = f"predict_{extractions}"
            else:
                stage_name = f"{model_name}_{action}"

        # Auto-depend on the last stage when no explicit dependency is given.
        # This makes sequential calls to add_prediction() behave sequentially.
        # Callers that want explicit parallelism (e.g. add_predictions()) must
        # pass depends_on explicitly so this auto-logic does not trigger.
        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = PredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            extractions=extractions,
            columns=columns,
            params=params,
            depends_on=depends_on,
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_structure_prediction(
        self,
        model_name: str,
        structure_key: str = "pdb",
        extractions: Optional[Union[str, list[Union[str, ExtractionSpec]]]] = None,
        columns: Optional[Union[str, dict[str, str]]] = None,
        plddt_key: Optional[str] = None,
        structure_format: Optional[str] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """Add a prediction stage that stores the structure from the response.

        Convenience wrapper around ``add_prediction()`` with a ``StructureSpec``.

        Args:
            model_name: BioLM model name (e.g. 'esmfold', 'alphafold2').
            structure_key: Response key containing the structure string.
            extractions: Optional scalar extractions (e.g. 'mean_plddt').
            columns: Output column name(s) for scalar extractions.
            plddt_key: Optional response key for confidence score.
            structure_format: 'pdb' or 'cif' (auto-detected from key if None).
            stage_name: Custom stage name.
            depends_on: Stage dependencies.

        Example::

            pipeline.add_structure_prediction(
                "esmfold",
                extractions="mean_plddt", columns="plddt",
                plddt_key="mean_plddt",
            )
        """
        spec = StructureSpec(
            key=structure_key,
            format=structure_format,
            plddt_key=plddt_key,
        )
        return self.add_prediction(
            model_name=model_name,
            extractions=extractions,
            columns=columns,
            stage_name=stage_name,
            depends_on=depends_on,
            structure_output=spec,
            **kwargs,
        )

    def add_predictions(
        self,
        models: list[Union[str, dict]],
        action: str = "predict",
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Add multiple prediction stages at the same level (run in parallel).

        Args:
            models: List of model names or dicts with model configs
            action: Default action if not specified in dict
            depends_on: List of stage names all these depend on
            **kwargs: Default kwargs for all stages

        Returns:
            self for chaining

        Example:
            >>> pipeline.add_predictions([
            ...     {'model_name': 'temberture-regression', 'extractions': 'prediction', 'columns': 'tm'},
            ...     {'model_name': 'biolmsol', 'extractions': 'solubility_score', 'columns': 'solubility'},
            ... ])
        """
        # All parallel models should share the same upstream dependency —
        # capture it BEFORE adding any of them so that later models in the loop
        # don't accidentally chain onto the first model.
        parallel_deps: list[str] = (
            depends_on if depends_on is not None
            else ([self.stages[-1].name] if self.stages else [])
        )

        for model in models:
            if isinstance(model, str):
                self.add_prediction(
                    model_name=model, action=action, depends_on=parallel_deps, **kwargs
                )
            elif isinstance(model, dict):
                model_config = {**kwargs, **model}
                # Per-model depends_on takes priority over the parallel_deps default
                model_depends = model_config.pop("depends_on", parallel_deps)
                model_name = model_config.pop("model_name")
                self.add_prediction(model_name=model_name, depends_on=model_depends, **model_config)
            else:
                raise TypeError(f"Model must be str or dict, got {type(model)}")

        return self

    def add_filter(
        self,
        filter_func: Union["BaseFilter", Callable[..., Any]],
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Add a filter stage.

        Args:
            filter_func: Filter function or BaseFilter instance
            stage_name: Custom stage name
            depends_on: List of stage names this depends on. When ``None``
                (default), auto-depends on the last added stage. Pass
                ``depends_on=[]`` for earliest-level execution.
        """
        stage_name = stage_name or f"filter_{len(self.stages)}"

        # Auto-depend on the last stage when no explicit dependency is given.
        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = FilterStage(
            name=stage_name,
            filter_func=filter_func,
            depends_on=depends_on,
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_clustering(
        self,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        similarity_metric: str = "hamming",
        embedding_model: Optional[str] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Add a sequence clustering stage.

        Clusters sequences by similarity and adds cluster_id column to DataFrame.

        Args:
            method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (required for kmeans/hierarchical)
            similarity_metric: 'hamming' or 'embedding'
            embedding_model: Model name if using embedding similarity
            stage_name: Optional custom stage name
            depends_on: Optional list of stage names this stage depends on
            **kwargs: Additional arguments for clustering (eps, min_samples, etc.)

        Example:
            >>> # Cluster by sequence similarity
            >>> pipeline.add_clustering(method='kmeans', n_clusters=10)
            >>>
            >>> # Cluster by embeddings
            >>> pipeline.add_prediction('esm2-650m', action='encode', stage_name='embed')
            >>> pipeline.add_clustering(
            ...     method='kmeans',
            ...     n_clusters=5,
            ...     similarity_metric='embedding',
            ...     embedding_model='esm2-650m',
            ...     depends_on=['embed']
            ... )
        """
        if stage_name is None:
            stage_name = f"cluster_{method}"

        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = ClusteringStage(
            name=stage_name,
            method=method,
            n_clusters=n_clusters,
            similarity_metric=similarity_metric,
            embedding_model=embedding_model,
            depends_on=depends_on,
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_cofolding_prediction(
        self,
        model_name: str,
        action: str = "predict",
        stage_name: Optional[str] = None,
        prediction_type: str = "structure",
        sequence_chain_id: str = "A",
        sequence_entity_type: str = "protein",
        static_entities=None,  # List[FoldingEntity]
        depends_on: Optional[list[str]] = None,
        params: Optional[dict] = None,
        batch_size: int = 1,
    ):
        """
        Add a co-folding prediction stage (Boltz2, Chai-1).

        Each pipeline sequence becomes the primary entity in a multi-molecule
        folding request.  ``static_entities`` injects ligands, cofactors,
        DNA/RNA strands, or additional protein chains that are constant across
        all designs.

        The caller is responsible for providing the correct molecule field names
        via ``static_entities`` and the right ``params`` for the model.

        Args:
            model_name: BioLM model slug (``'boltz2'``, ``'chai1'``).
            action: API action (default ``'predict'``).
            stage_name: Optional stage name (defaults to ``model_name``).
            prediction_type: Column name for the confidence score.
            sequence_chain_id: Chain ID assigned to the primary sequence
                (e.g. ``'A'`` for Boltz, molecule ``name`` for Chai-1).
            sequence_entity_type: Entity type for the primary sequence
                (``'protein'``, ``'dna'``, ``'rna'``).
            static_entities: List of :class:`FoldingEntity` objects to include
                in every request alongside the primary sequence.
            depends_on: Upstream stage names.
            params: Model-specific params (e.g. ``{'recycling_steps': 3}``).
            batch_size: Sequences per API call (default 1).

        Example::

            from biolm.pipeline import FoldingEntity

            pipeline.add_cofolding_prediction(
                model_name='boltz2',
                static_entities=[
                    FoldingEntity(id='L', entity_type='ligand', smiles='c1ccccc1'),
                ],
                params={'recycling_steps': 3, 'sampling_steps': 20},
                depends_on=['filter_top50'],
            )
        """
        stage_name = stage_name or model_name

        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = CofoldingPredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            prediction_type=prediction_type,
            sequence_chain_id=sequence_chain_id,
            sequence_entity_type=sequence_entity_type,
            static_entities=static_entities or [],
            params=params or {},
            batch_size=batch_size,
            depends_on=depends_on,
        )
        self.add_stage(stage)
        return self


class SingleStepPipeline(DataPipeline):
    """
    Simplified pipeline for single-step predictions.

    Convenience class for running a single prediction model on sequences.

    Args:
        model_name: BioLM model name
        action: API action ('predict', 'encode', 'score')
        sequences: Input sequences
        params: Optional API parameters
        **kwargs: Additional arguments passed to DataPipeline

    Example:
        >>> pipeline = SingleStepPipeline(
        ...     model_name='esmfold',
        ...     sequences=['MKTAYIAKQRQ', 'MKLAVID']
        ... )
        >>> results = pipeline.run()
        >>> df = pipeline.get_final_data()
    """

    def __init__(
        self,
        model_name: str,
        action: str = "predict",
        sequences: Union[list[str], pd.DataFrame, str, Path] = None,
        params: Optional[dict] = None,
        extractions=None,
        columns=None,
        embedding_extractor=None,
        **kwargs,
    ):
        # validate extractions before calling add_prediction() so the
        # error message is actionable at the SingleStepPipeline level.
        if action in ("predict", "score") and extractions is None:
            raise ValueError(
                f"extractions= is required for action='{action}'. "
                f"Example: extractions='prediction'. "
                f"Check your model's API response to find the correct key."
            )

        super().__init__(sequences=sequences, **kwargs)

        # Automatically add single prediction stage
        self.add_prediction(
            model_name=model_name,
            action=action,
            params=params,
            extractions=extractions,
            columns=columns,
            embedding_extractor=embedding_extractor,
        )


# Convenience aliases
def Predict(
    model_name: str,
    sequences: Union[list[str], pd.DataFrame, str, Path],
    extractions: Optional[Union[str, list]] = None,
    params: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function for single-step prediction.

    Args:
        model_name: BioLM model name
        sequences: Input sequences
        extractions: API response key(s) to extract. Required — pass the response
            key for your model (e.g. ``extractions='prediction'`` for temberture,
            ``extractions='mean_plddt'`` for esmfold). Use a list or
            ExtractionSpec for multiple extractions.
        params: Optional API parameters
        **kwargs: Additional arguments

    Returns:
        DataFrame with predictions

    Example:
        >>> df = Predict('temberture-regression', sequences=['MKTAYIAKQRQ'], extractions='prediction')
    """
    # pass extractions through to SingleStepPipeline/PredictionStage.
    pipeline = SingleStepPipeline(
        model_name=model_name,
        action="predict",
        sequences=sequences,
        params=params,
        extractions=extractions,
        **kwargs,
    )
    pipeline.run()
    return pipeline.get_final_data()


def Embed(
    model_name: str,
    sequences: Union[list[str], pd.DataFrame, str, Path],
    layer: Optional[int] = None,
    key: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function for generating embeddings.

    Args:
        model_name: BioLM model name (e.g., ``'esm2-8m'``, ``'esm2-650m'``,
            ``'ablang2'``).  See ``biolm.list_models()`` for the full slugged
            list — the family name alone (``'esm2'``) is not a valid endpoint.
        sequences: Input sequences
        layer: Optional layer number
        key: Response dict key containing embeddings. Auto-detected if None:
            ESM2 models use "embeddings", AbLang2 uses "seqcoding",
            others default to "embedding".
        **kwargs: Additional arguments

    Returns:
        DataFrame with 'sequence', 'sequence_id', and 'embedding' columns

    Example:
        >>> df = Embed('esm2-8m', sequences=['MKTAYIAKQRQ', 'MKLAVID'])
    """
    params = {"layer": layer} if layer is not None else None

    # Auto-detect the embedding response key from a per-family prefix list.
    # Substring matching is too loose ("esm" matches esmfold, which doesn't
    # return embeddings); a prefix allowlist keeps misclassifications visible.
    if key is None:
        _mn = model_name.lower()
        _esm_prefixes = ("esm2", "esm1v", "esm1b", "esmc", "esm-2", "esm-1v")
        if _mn.startswith("ablang"):
            key = "seqcoding"
        elif any(_mn.startswith(p) for p in _esm_prefixes):
            key = "embeddings"
        else:
            key = "embedding"

    # create an explicit datastore so the connection stays open after
    # run() completes.  When no datastore is provided, run_async() closes the
    # auto-created one in its finally block — before we can call
    # get_embeddings_bulk().  By passing our own datastore we set
    # _auto_created_datastore=False, keeping the connection alive.
    user_ds = kwargs.pop("datastore", None)
    _tmpdir: Optional[str] = None
    if user_ds is None:
        import tempfile as _tempfile

        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            _tmpdir = _tempfile.mkdtemp(prefix="biolm_embed_")
            output_dir = Path(_tmpdir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        from biolm.pipeline.datastore_duckdb import DuckDBDataStore as _DS

        _owned_ds: Optional["_DS"] = _DS(
            db_path=output_dir / "embed.duckdb",
            data_dir=output_dir / "data",
        )
    else:
        _owned_ds = None

    ds = _owned_ds if _owned_ds is not None else user_ds

    # pass layer= to EmbeddingSpec so cache check and storage use the correct layer.
    pipeline = SingleStepPipeline(
        model_name=model_name,
        action="encode",
        sequences=sequences,
        params=params,
        embedding_extractor=EmbeddingSpec(key=key, layer=layer),
        datastore=ds,
        **kwargs,
    )
    try:
        pipeline.run()
        df = pipeline.get_final_data()

        # Load embeddings via bulk fetch (single DuckDB JOIN + batch Parquet reads)
        # instead of N per-sequence queries
        seq_ids = df["sequence_id"].tolist()
        emb_map = ds.get_embeddings_bulk(seq_ids, model_name=model_name)
        df["embedding"] = [emb_map.get(int(sid)) for sid in seq_ids]
    finally:
        if _owned_ds is not None:
            _owned_ds.close()
        # Clean up the auto-created temp directory now that embeddings are
        # materialized into the returned DataFrame.  Without this, every
        # Embed() call leaks a `biolm_embed_*` dir under the system temp.
        if _tmpdir is not None:
            import shutil as _shutil
            _shutil.rmtree(_tmpdir, ignore_errors=True)

    return df
