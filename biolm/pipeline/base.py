"""
Base Pipeline classes for stage management and execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from biolm.pipeline.datastore_duckdb import DuckDBDataStore as DataStore

_logger = logging.getLogger(__name__)

# prevent calling nest_asyncio.apply() more than once per process.
# nest_asyncio is designed to be idempotent but re-patching loops adds overhead.
_nest_asyncio_applied: bool = False


@dataclass(frozen=True)
class InputSchema:
    """Describes the input columns for a pipeline.

    When set, these columns are the primary data — ``sequence`` is not required.
    Columns are stored directly on the ``sequences`` table via
    ``ALTER TABLE ADD COLUMN`` so that ``materialize_working_set()``, SQL
    filters, and ``item_columns`` all work via direct JOINs.

    Hashing uses all columns (sorted alphabetically) joined with ``\\x00``
    separators so that identical rows produce the same hash regardless of
    column order.

    Args:
        columns: List of column names that comprise the primary input.
    """

    columns: list[str]

    def hash_row(self, row: dict[str, str]) -> str:
        """SHA-256 hash of the row values across all input columns (sorted)."""
        import hashlib

        parts = [str(row.get(c, "")) for c in sorted(self.columns)]
        payload = "\x00".join(parts)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


class PipelineContext:
    """Shared key-value store backed by DuckDB for inter-stage communication.

    Stages can read/write arbitrary data through the context. Common use
    case: stage 1 predicts structures (stored in the ``structures`` table),
    stage 2 reads them for structure-conditioned generation.

    Args:
        datastore: The pipeline's DuckDB datastore.
        run_id: Current pipeline run ID.
    """

    def __init__(self, datastore: DataStore, run_id: str):
        self._datastore = datastore
        self._run_id = run_id

    def set(self, key: str, value: Any):
        """Store a value in the pipeline context table."""
        self._datastore.set_context(self._run_id, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the pipeline context table."""
        val = self._datastore.get_context(self._run_id, key)
        return val if val is not None else default

    def get_structure(self, sequence_id: int, model_name: Optional[str] = None) -> Optional[dict]:
        """Convenience: fetch a structure from the datastore's structures table."""
        return self._datastore.get_structure(sequence_id, model_name)

    def get_structures_for_ws(
        self, ws: "WorkingSet", model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch structures for all sequences in a WorkingSet."""
        ids = list(ws.sequence_ids)
        df = self._datastore.get_structures_bulk(ids)
        if model_name and not df.empty:
            df = df[df["model_name"] == model_name]
        return df


@dataclass(frozen=True)
class WorkingSet:
    """Lightweight set of sequence IDs — replaces DataFrame as inter-stage transport.

    Stages operate on DuckDB directly and pass only the set of surviving
    sequence IDs to the next stage.  Materialization to DataFrame happens
    once at ``get_final_data()`` time.

    Memory: 1M IDs ≈ 28 MB (frozenset[int]) vs 500 MB+ DataFrame.
    """

    sequence_ids: frozenset[int]

    # --- convenience helpers ---------------------------------------------------

    def intersect(self, other: "WorkingSet") -> "WorkingSet":
        """Return a new WorkingSet containing only IDs present in both sets."""
        return WorkingSet(self.sequence_ids & other.sequence_ids)

    def union(self, other: "WorkingSet") -> "WorkingSet":
        """Return a new WorkingSet containing IDs from either set."""
        return WorkingSet(self.sequence_ids | other.sequence_ids)

    def difference(self, other: "WorkingSet") -> "WorkingSet":
        """Return IDs in *self* but not in *other*."""
        return WorkingSet(self.sequence_ids - other.sequence_ids)

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __bool__(self) -> bool:
        return bool(self.sequence_ids)

    def __iter__(self):
        return iter(self.sequence_ids)

    def __contains__(self, item: int) -> bool:
        return item in self.sequence_ids

    @classmethod
    def from_ids(cls, ids) -> "WorkingSet":
        """Create from any iterable of ints."""
        return cls(frozenset(int(i) for i in ids))

    def to_list(self) -> list[int]:
        """Return sorted list (useful for DuckDB queries)."""
        return sorted(self.sequence_ids)


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage_name: str
    input_count: int
    output_count: int
    filtered_count: int = 0
    cached_count: int = 0
    computed_count: int = 0
    elapsed_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"StageResult({self.stage_name}: "
            f"in={self.input_count}, out={self.output_count}, "
            f"cached={self.cached_count}, computed={self.computed_count}, "
            f"filtered={self.filtered_count}, time={self.elapsed_time:.1f}s)"
        )


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    A stage represents a single processing step in the pipeline.
    It can filter data, compute predictions, or transform sequences.

    Args:
        name: Stage name (must be unique within pipeline)
        cache_key: Unused collision-dedup key (auto-derived by PredictionStage)
        depends_on: List of stage names this stage depends on
        model_name: Model name for predictions/structures
        max_concurrent: Maximum concurrent API calls (for rate limiting)

    Class Attributes:
        merge_mode: How this stage's output is merged when it runs in parallel
            with other stages.  ``"intersect"`` (default) = only keep sequences
            that pass *all* parallel stages (correct for filters).
            ``"union"`` = keep sequences that appear in *any* parallel stage
            output (correct for independent prediction stages that may skip
            sequences on error but should not drop others).
    """

    merge_mode: str = "intersect"

    def __init__(
        self,
        name: str,
        cache_key: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        model_name: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        self.name = name
        self.cache_key = cache_key or name
        self.depends_on = depends_on or []
        self.model_name = model_name
        self.max_concurrent = max_concurrent

    @abstractmethod
    async def process_ws(
        self, ws: "WorkingSet", datastore: DataStore, **kwargs
    ) -> tuple["WorkingSet", StageResult]:
        """Process data using WorkingSet (DuckDB-native).

        All stages must implement this method.  Stages that need actual
        sequence data (e.g. ClusteringStage) should call
        ``datastore.materialize_working_set(ws)`` internally — that is
        *the stage's* responsibility, not the pipeline's.

        Args:
            ws: Input WorkingSet (set of sequence IDs).
            datastore: DataStore for reading/writing data.
            **kwargs: Additional arguments (e.g. ``run_id``).

        Returns:
            Tuple of (output WorkingSet, StageResult).
        """
        pass

    def to_spec(self) -> dict:
        """Return a serializable dict describing this stage.

        Used by BasePipeline.run_async() to persist pipeline definitions to DuckDB
        (enabling DataPipeline.from_db() recovery after kernel death).

        Subclasses must override this. Raises NotImplementedError by default.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement to_spec(). "
            "Implement to_spec() to enable pipeline definition saving and from_db() recovery."
        )

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Legacy DataFrame interface — used by streaming mode and GenerationStage.

        Subclasses may override this for backward compatibility or for cases
        where a DataFrame is the natural input (e.g. generation with an empty df).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement process()"
        )

    def __repr__(self):
        deps = f", depends_on={self.depends_on}" if self.depends_on else ""
        return f"{self.__class__.__name__}('{self.name}'{deps})"


@dataclass
class PipelineMetadata:
    """Metadata for a pipeline run — lets users retrieve and reuse cached results.

    Attributes:
        pipeline_id: Unique identifier for the pipeline's cache directory.
        cache_dir: Path to the ``.biolm/pipelines/<id>`` cache directory.
        db_path: Path to the DuckDB database file inside the cache directory.
        run_id: The run ID for this execution (there can be multiple runs
            sharing the same cache).

    Example::

        pipeline = DataPipeline(sequences=[...])
        pipeline.run()
        meta = pipeline.metadata
        print(meta.pipeline_id)   # "20260302_143022_a1b2c3d4"
        print(meta.cache_dir)     # ".biolm/pipelines/20260302_143022_a1b2c3d4"

        # Later — reuse the same cache:
        pipeline2 = DataPipeline(
            sequences=new_seqs,
            datastore=meta.db_path,   # or str(meta.cache_dir)
            resume=True,
        )
    """

    pipeline_id: str
    cache_dir: Path
    db_path: Path
    run_id: str


# Default cache root — lives alongside other dotfiles in the working directory
_BIOLM_CACHE_ROOT = Path(".biolm") / "pipelines"


class BasePipeline(ABC):
    """
    Base class for all pipeline types.

    Provides:
    - Stage management and dependency resolution
    - Async execution with progress tracking
    - Caching and resumability
    - Export and visualization

    When no ``datastore`` is provided, the pipeline automatically creates a
    DuckDB cache under ``.biolm/pipelines/<pipeline_id>/``.  The
    ``pipeline_id`` (and full cache path) is exposed via :attr:`metadata` so
    users can reconnect to the same cache in later sessions.

    Args:
        datastore: DataStore instance or path to a DuckDB file.  Required.
        run_id: Unique run identifier (auto-generated if not provided).
        output_dir: Directory for CSV/Parquet exports (default ``pipeline_outputs``).
        resume: Whether to resume from a previous run.
        verbose: Enable verbose output.
    """

    def __init__(
        self,
        datastore: Union[DataStore, str, Path, None] = None,
        run_id: Optional[str] = None,
        output_dir: Union[str, Path] = "pipeline_outputs",
        resume: bool = False,
        verbose: bool = True,
        input_schema: Optional[InputSchema] = None,
    ):
        self.run_id = run_id or self._generate_run_id()

        # Setup datastore
        if datastore is None:
            # Auto-create: .biolm/pipelines/<run_id>/pipeline.duckdb
            cache_dir = _BIOLM_CACHE_ROOT / self.run_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.datastore = DataStore(str(cache_dir / "pipeline.duckdb"))
            self._pipeline_id = self.run_id
            self._cache_dir = cache_dir
            self._auto_created_datastore = True
        elif isinstance(datastore, DataStore):
            self.datastore = datastore
            self._pipeline_id = self.run_id
            self._cache_dir = Path(datastore.db_path).parent
            self._auto_created_datastore = False
        elif isinstance(datastore, (str, Path)):
            self.datastore = DataStore(str(datastore))
            self._pipeline_id = self.run_id
            self._cache_dir = Path(datastore).parent
            self._auto_created_datastore = False
        else:
            raise TypeError(
                f"datastore must be a DuckDBDataStore instance or a path to a "
                f"DuckDB file, got {type(datastore).__name__}. Example: "
                f"DataPipeline(sequences=..., datastore='my_pipeline.duckdb')"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.verbose = verbose
        self.input_schema = input_schema

        # Pipeline context for inter-stage communication
        self.context = PipelineContext(self.datastore, self.run_id)

        # Stage management
        self.stages: list[Stage] = []
        self.stage_results: dict[str, StageResult] = {}
        # WorkingSet is now primary inter-stage transport
        self._working_sets: dict[str, WorkingSet] = {}
        # Final merged working set at end of run() — authoritative for
        # get_final_data() so branched DAGs return the correct sink instead
        # of self.stages[-1] (which is just the last-added, not last-executed).
        self._final_ws: Optional[WorkingSet] = None
        self._final_df: Optional[pd.DataFrame] = None
        # Legacy: kept for backward compat (populated lazily by get_final_data)
        self._stage_data: dict[str, pd.DataFrame] = {}

        # Pipeline state
        self.pipeline_type = self.__class__.__name__
        self.status = "initialized"
        self.start_time = None
        self.end_time = None

    @property
    def metadata(self) -> PipelineMetadata:
        """Return metadata for reconnecting to this pipeline's cache later."""
        return PipelineMetadata(
            pipeline_id=self._pipeline_id,
            cache_dir=self._cache_dir,
            db_path=Path(self.datastore.db_path),
            run_id=self.run_id,
        )

    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"

    def add_stage(self, stage: Stage):
        """Add a stage to the pipeline."""
        # Check for duplicate stage name
        existing_names = {s.name for s in self.stages}
        if stage.name in existing_names:
            raise ValueError(
                f"Duplicate stage name '{stage.name}'. "
                "Use stage_name= to provide a unique name."
            )

        # Check for column name collision.
        # Two stages may share output columns only if they have the exact same
        # (model_name, action) pair — meaning they are the same prediction and
        # one will just hit the cache.  Different model OR different action with
        # the same column name would silently overwrite DuckDB storage.
        stage_columns = {r.column for r in getattr(stage, "_resolved", [])}
        stage_key = (getattr(stage, "model_name", None), getattr(stage, "action", None))
        if stage_columns:
            for s in self.stages:
                s_columns = {r.column for r in getattr(s, "_resolved", [])}
                s_key = (getattr(s, "model_name", None), getattr(s, "action", None))
                overlap = stage_columns & s_columns
                if overlap and s_key != stage_key:
                    raise ValueError(
                        f"Column(s) {overlap} already used by stage '{s.name}' "
                        f"(model='{s_key[0]}', action='{s_key[1]}'). "
                        f"New stage '{stage.name}' uses model='{stage_key[0]}', "
                        f"action='{stage_key[1]}'. Use different column names via "
                        "the 'columns' parameter to avoid collision."
                    )

        # Cross-pipeline column registry check: ensure no column conflicts with
        # data stored in this DuckDB from a different (model, action) pair.
        # PD-05: skip this check when resume=True — a reconstructed pipeline is
        # by definition consistent with the registry it wrote, so re-validating
        # against the registry would raise false-positive errors.
        if not getattr(self, "resume", False) and self.datastore and stage_columns:
            for r in getattr(stage, "_resolved", []):
                existing = self.datastore.get_column_registry_entry(r.column)
                if existing and (existing["model_name"], existing["action"]) != (
                    getattr(stage, "model_name", None),
                    getattr(stage, "action", None),
                ):
                    raise ValueError(
                        f"Column '{r.column}' was previously used by "
                        f"model='{existing['model_name']}', action='{existing['action']}' "
                        f"(definition '{existing['definition_id']}'). "
                        "Using a different (model, action) with the same column name would "
                        "overwrite existing predictions in DuckDB. "
                        "Use a different column name via the 'columns' parameter."
                    )

        # Validate dependencies
        for dep in stage.depends_on:
            if dep not in existing_names:
                raise ValueError(
                    f"Stage '{stage.name}' depends on '{dep}' which hasn't been added yet"
                )

        self.stages.append(stage)
        if self.verbose:
            print(f"Added stage: {stage}")

    def _save_definition_and_register_columns(self):
        """Serialize stages, save definition to DuckDB, register output columns."""
        if self.datastore is None:
            return
        from biolm.pipeline.pipeline_def import _pipeline_def_hash

        stages_specs = []
        for s in self.stages:
            try:
                stages_specs.append(s.to_spec())
            except NotImplementedError as exc:
                # mark the spec as partially serializable so that
                # from_db() / pipeline_from_definition() can surface a clear error
                # rather than silently reconstructing an incomplete pipeline.
                stages_specs.append(
                    {
                        "type": s.__class__.__name__,
                        "name": s.name,
                        "_serialization_error": True,
                        "_serialization_error_msg": str(exc),
                    }
                )
            except Exception as exc:
                # Unexpected serialization failure — propagate so the caller knows
                # the definition was NOT saved cleanly.
                raise RuntimeError(
                    f"Stage '{s.name}' ({s.__class__.__name__}) raised an unexpected "
                    f"error during serialization: {exc}. "
                    "Fix to_spec() or exclude this stage before saving the definition."
                ) from exc

        input_cols = self.input_schema.columns if self.input_schema else None
        # Hash computed from the FULL spec (before blob extraction) so the
        # definition ID is stable regardless of blob storage threshold changes.
        def_id = _pipeline_def_hash(self.pipeline_type, input_cols, stages_specs)
        # Externalize large strings (PDB files etc.) into pipeline_blobs and
        # replace them with {"_blob_ref": blob_id} before writing stages_json.
        from biolm.pipeline.pipeline_def import _extract_blobs
        stages_specs_stored = _extract_blobs(stages_specs, self.datastore)
        self.datastore.save_pipeline_definition(
            def_id,
            self.pipeline_type,
            json.dumps(input_cols) if input_cols is not None else None,
            json.dumps(stages_specs_stored),
        )
        for s in self.stages:
            for r in getattr(s, "_resolved", []):
                self.datastore.register_column(
                    r.column,
                    getattr(s, "model_name", ""),
                    getattr(s, "action", ""),
                    def_id,
                    s.name,
                )
        self.datastore.conn.execute(
            "UPDATE pipeline_runs SET definition_id = ? WHERE run_id = ?",
            [def_id, self.run_id],
        )

    @classmethod
    def from_db(
        cls,
        db_path: Union[str, "Path"],
        definition_id: Optional[str] = None,
        run_id: Optional[str] = None,
        verbose: bool = True,
    ) -> "BasePipeline":
        """Reconstruct a pipeline from an existing DuckDB database.

        Useful for recovering after a kernel death without re-running
        already-completed stages.

        Args:
            db_path: Path to the DuckDB database file.
            definition_id: Specific definition to load (``None`` = latest).
            run_id: Run ID for the reconstructed pipeline (``None`` = generate new).
            verbose: Enable verbose output.

        Returns:
            Reconstructed :class:`BasePipeline` subclass instance.

        Example::

            pipeline = DataPipeline.from_db("my_pipeline.duckdb")
            pipeline.run(resume=True)
        """
        from biolm.pipeline.pipeline_def import pipeline_from_definition

        ds = DataStore(str(db_path))
        # If a run_id is given, resolve its definition_id directly so we load
        # the definition that was actually used for that run, not the latest one.
        if definition_id is None and run_id is not None:
            row = ds.conn.execute(
                "SELECT definition_id FROM pipeline_runs WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if row and row[0]:
                definition_id = row[0]
        defn = ds.load_pipeline_definition(definition_id)
        if defn is None:
            raise ValueError(
                f"No pipeline definition found in '{db_path}'. "
                "Run a pipeline with this datastore first to persist a definition."
            )
        return pipeline_from_definition(defn, datastore=ds, run_id=run_id, verbose=verbose)

    def _resolve_dependencies(self) -> list[list[Stage]]:
        """
        Resolve stage dependencies and return execution order.

        Returns:
            List of stage groups, where stages in each group can run in parallel
        """
        # Build dependency graph
        stage_map = {s.name: s for s in self.stages}

        # validate all depends_on references exist before sorting.
        # Without this check, a typo in depends_on silently drops the dependency
        # and the stage runs in an incorrect (too-early) level.
        for s in self.stages:
            for dep in s.depends_on:
                if dep not in stage_map:
                    raise ValueError(
                        f"Stage '{s.name}' depends on '{dep}' which does not exist. "
                        f"Available stages: {sorted(stage_map.keys())}"
                    )

        # Topological sort with level detection
        in_degree = {s.name: len(s.depends_on) for s in self.stages}
        levels = []

        while in_degree:
            # Find all stages with no dependencies
            current_level = [
                stage_map[name] for name, degree in in_degree.items() if degree == 0
            ]

            if not current_level:
                remaining = list(in_degree.keys())
                raise ValueError(
                    f"Circular dependency detected among stages: {remaining}"
                )

            levels.append(current_level)

            # Remove current level from graph
            for stage in current_level:
                del in_degree[stage.name]

            # Decrease in-degree for dependent stages
            for stage in current_level:
                for other_stage in self.stages:
                    if (
                        stage.name in other_stage.depends_on
                        and other_stage.name in in_degree
                    ):
                        in_degree[other_stage.name] -= 1

        # validate that no stage in a level depends on another stage
        # in the same level — that would create an intra-level circular dependency
        # that is silently undetectable by the standard topological sort above.
        for level in levels:
            level_names = {s.name for s in level}
            for s in level:
                intra_deps = [dep for dep in s.depends_on if dep in level_names]
                if intra_deps:
                    raise ValueError(
                        f"Stage '{s.name}' depends on {intra_deps} which are in the same "
                        f"parallel execution level. This creates a circular dependency. "
                        f"Ensure stages that depend on each other have explicit ordering "
                        f"via depends_on."
                    )

        return levels

    def _reload_stage_working_set(
        self,
        stage: Stage,
        ws_input: WorkingSet,
    ) -> Optional[WorkingSet]:
        """Reconstruct a stage's WorkingSet from the datastore (for resume).

        All reload paths scope to the current run by intersecting
        with ``ws_input``.  ``ws_input`` is already run-scoped (built from
        ``_get_initial_data()`` which only inserts/returns the current run's
        sequences).  The FilterStage path additionally scopes by ``run_id``
        via ``get_filter_results()``.  This prevents sequences from earlier
        runs sharing the same DuckDB from bleeding into a resumed run.

        Returns the reloaded WorkingSet, or None if it cannot be reconstructed
        (caller should then re-run the stage normally).
        """
        if self.datastore is None:
            return None
        try:
            from biolm.pipeline.data import (
                ClusteringStage,
                CofoldingPredictionStage,
                FilterStage,
                PredictionStage,
            )
            from biolm.pipeline.generative import GenerationStage

            if isinstance(stage, PredictionStage):
                if getattr(stage, "action", None) == "encode":
                    # Encode stages write to the embeddings table, not predictions.
                    # scope by layer when EmbeddingSpec(layer=N) is set so
                    # that reloading does not confuse embeddings from different layers
                    # stored for the same model.
                    try:
                        from biolm.pipeline.data import EmbeddingSpec

                        emb_extractor = getattr(stage, "_embedding_extractor", None)
                        layer = None
                        if isinstance(emb_extractor, EmbeddingSpec):
                            layer = emb_extractor.layer

                        if layer is not None:
                            emb_ids = set(
                                self.datastore.conn.execute(
                                    "SELECT DISTINCT sequence_id FROM embeddings "
                                    "WHERE model_name = ? AND layer = ?",
                                    [stage.model_name, layer],
                                ).df()["sequence_id"].tolist()
                            )
                        else:
                            emb_ids = set(
                                self.datastore.conn.execute(
                                    "SELECT DISTINCT sequence_id FROM embeddings "
                                    "WHERE model_name = ? AND layer IS NULL",
                                    [stage.model_name],
                                ).df()["sequence_id"].tolist()
                            )
                        candidate_ids = ws_input.sequence_ids & emb_ids
                        if not candidate_ids:
                            return None
                        return WorkingSet.from_ids(candidate_ids)
                    except Exception:
                        return None

                # For predict/score: IDs that have ALL resolved columns in DuckDB.
                resolved = getattr(stage, "_resolved", [])
                if not resolved:
                    return None
                candidate_ids = set(ws_input.sequence_ids)
                for r in resolved:
                    ids_with_col = self.datastore.get_sequence_ids_with_prediction(
                        list(candidate_ids),
                        r.column,
                        stage.model_name,
                    )
                    # Exclude NULL-valued predictions (failed batches stored with
                    # skip_on_error=True — these must be retried, not treated as cached)
                    if ids_with_col:
                        ids_with_valid = set(
                            self.datastore.conn.execute(
                                "SELECT DISTINCT sequence_id FROM predictions "
                                "WHERE sequence_id IN ({}) AND prediction_type = ? "
                                "AND model_name = ? AND value IS NOT NULL".format(
                                    ",".join(str(i) for i in ids_with_col)
                                ),
                                [r.column, stage.model_name],
                            ).df()["sequence_id"].tolist()
                        )
                        candidate_ids &= ids_with_valid
                    else:
                        candidate_ids = frozenset()
                    if not candidate_ids:
                        return None
                return WorkingSet.from_ids(candidate_ids)

            elif isinstance(stage, FilterStage):
                # get_filter_results is already scoped by run_id.
                # Intersect with ws_input to guard against stale IDs if the
                # input set has shrunk since the filter originally ran.
                passed_ids = self.datastore.get_filter_results(self.run_id, stage.name)
                if not passed_ids:
                    return None
                candidate_ids = ws_input.sequence_ids & set(passed_ids)
                if not candidate_ids:
                    return None
                return WorkingSet.from_ids(candidate_ids)

            elif isinstance(stage, ClusteringStage):
                # ClusteringStage doesn't filter — it annotates
                # sequences with cluster assignments stored in pipeline_metadata.
                # On resume, verify the assignments key exists (proving this
                # stage completed), then return ws_input unchanged so the
                # pipeline continues without re-clustering.
                # ws_input is already run-scoped (from _get_initial_data), so
                # no extra intersection needed here.
                assignments = self.datastore.get_pipeline_metadata(
                    f"clustering_{stage.name}_assignments"
                )
                if assignments is None:
                    return None
                # ClusteringStage survives all input sequences; return ws_input
                # as the output WorkingSet (cluster columns added at materialize).
                return ws_input

            elif isinstance(stage, CofoldingPredictionStage):
                # Co-folding results are stored in the structures table.
                try:
                    struct_ids = set(
                        self.datastore.conn.execute(
                            "SELECT DISTINCT sequence_id FROM structures WHERE model_name = ?",
                            [stage.model_name],
                        ).df()["sequence_id"].tolist()
                    )
                    candidate_ids = ws_input.sequence_ids & struct_ids
                    if not candidate_ids:
                        return None
                    return WorkingSet.from_ids(candidate_ids)
                except Exception:
                    return None

            elif isinstance(stage, GenerationStage):
                # Generation is stochastic — always re-run to produce fresh
                # sequences. Downstream prediction/filter stages will still use
                # their caches for any sequences they've already processed.
                return None

        except Exception as exc:
            _logger.warning(
                "Could not reload stage '%s' from datastore: %s", stage.name, exc
            )

        return None

    def _get_stage_input_ws(self, stage: Stage, ws_initial: WorkingSet) -> WorkingSet:
        """Resolve the correct input WorkingSet for a stage based on depends_on.

        - No dependencies → use the initial (root) working set
        - One dependency → use that dependency's stored output
        - Multiple dependencies → intersect all dependency outputs

        Raises if a declared dependency name is missing from ``_working_sets``
        — silently falling back to the root would silently substitute the
        wrong dataset for the stage on a typo.
        """
        deps = stage.depends_on if hasattr(stage, "depends_on") else []
        if not deps:
            return ws_initial

        missing = [d for d in deps if d not in self._working_sets]
        if missing:
            known = sorted(self._working_sets.keys())
            raise KeyError(
                f"Stage {stage.name!r} declared depends_on={list(deps)!r} but "
                f"{missing!r} have no recorded output. Known stages: {known!r}."
            )

        dep_sets = [self._working_sets[d] for d in deps]
        result = dep_sets[0]
        for ws in dep_sets[1:]:
            result = result.intersect(ws)
        return result

    async def _execute_stage_ws(
        self, stage: Stage, ws_input: WorkingSet
    ) -> tuple[WorkingSet, StageResult]:
        """Execute a single stage using WorkingSet transport."""
        start_time = time.time()

        # DS-06 fix: skip the stage immediately when the incoming WorkingSet is
        # empty.  Stages that receive zero sequences may raise errors (e.g. when
        # materializing an empty set), so short-circuit here with a warning.
        if not ws_input:
            _logger.warning(
                "Stage '%s' received empty WorkingSet — skipping", stage.name
            )
            # mark the stage complete even when skipped so that resume
            # logic correctly identifies it as already processed on future runs.
            _empty_stage_id = f"{self.run_id}_{stage.name}"
            if self.datastore:
                self.datastore.mark_stage_complete(
                    stage_id=_empty_stage_id,
                    run_id=self.run_id,
                    stage_name=stage.name,
                    input_count=0,
                    output_count=0,
                    status="completed",
                )
            return WorkingSet(frozenset()), StageResult(
                stage_name=stage.name,
                input_count=0,
                output_count=0,
                elapsed_time=0.0,
                metadata={"skipped": "empty_input"},
            )

        # Check if stage is already complete (resumability)
        stage_id = f"{self.run_id}_{stage.name}"
        if (
            self.resume
            and self.datastore
            and self.datastore.is_stage_complete(stage_id)
        ):
            if self.verbose:
                print(f"\n✓ Stage '{stage.name}' already complete — reloading from DB")

            ws_resumed = self._reload_stage_working_set(stage, ws_input)
            if ws_resumed is not None:
                # Integrity check: warn if reloaded count differs from recorded output_count.
                # This catches partial writes (crash mid-batch) where stage_completions
                # says "completed" but fewer rows were actually committed.
                sc_row = self.datastore.conn.execute(
                    "SELECT output_count FROM stage_completions WHERE stage_id = ?",
                    [stage_id],
                ).fetchone()
                if sc_row is not None and sc_row[0] is not None:
                    expected = sc_row[0]
                    actual = len(ws_resumed)
                    if actual != expected:
                        import warnings
                        warnings.warn(
                            f"Stage '{stage.name}' resume integrity mismatch: "
                            f"stage_completions recorded {expected} output sequences but "
                            f"only {actual} have verifiable data in the datastore. "
                            f"Some sequences may be missing downstream predictions. "
                            f"Re-run without resume=True to reprocess missing sequences.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                if self.verbose:
                    print(f"  Reloaded {len(ws_resumed):,} sequences")
                return ws_resumed, StageResult(
                    stage_name=stage.name,
                    input_count=len(ws_input),
                    output_count=len(ws_resumed),
                    elapsed_time=0.0,
                    metadata={"resumed": True},
                )
            else:
                if self.verbose:
                    print(f"  Cannot reload from DB — re-running stage '{stage.name}'")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Stage: {stage.name}")
            print(f"Input: {len(ws_input):,} sequences")
            if stage.depends_on:
                print(f"Depends on: {', '.join(stage.depends_on)}")

        # Execute stage — all stages implement process_ws().
        # note: run_id, context, and verbose are passed as explicit keyword
        # arguments so stages can declare them in their process_ws() signature
        # rather than relying on **kwargs swallowing them silently.  All stage
        # implementations should declare: process_ws(self, ws, datastore, *,
        # run_id="", context=None, verbose=False, **kwargs).
        ws_out, result = await stage.process_ws(
            ws_input, self.datastore, run_id=self.run_id, context=self.context,
            verbose=self.verbose,
        )

        result.elapsed_time = time.time() - start_time

        # Mark stage complete
        if self.datastore:
            self.datastore.mark_stage_complete(
                stage_id=stage_id,
                run_id=self.run_id,
                stage_name=stage.name,
                input_count=result.input_count,
                output_count=result.output_count,
                status="completed",
            )

        if self.verbose:
            print(f"\n{result}")
            print(f"{'='*60}")

        return ws_out, result

    async def run_async(
        self, enable_streaming: bool = True, **kwargs
    ) -> dict[str, StageResult]:
        """
        Run the pipeline asynchronously.

        Args:
            enable_streaming: Stream prediction results through per-sequence
                filters for better parallelism and lower latency (default True).

        Returns:
            Dict mapping stage names to StageResults
        """
        # Reset per-run state so a second run() call starts clean
        self.stage_results = {}
        self._working_sets = {}
        self._stage_data = {}
        self.status = "running"
        self.start_time = time.time()
        # reset end_time so a re-run after failure doesn't retain
        # the previous run's end timestamp.
        self.end_time = None
        # clear the filter-WS cache on each FilterStage so a second
        # run() call doesn't reuse stale cached results from the previous run.
        for stage in self.stages:
            if hasattr(stage, "_cached_ws_ids"):
                stage._cached_ws_ids = None

        # Guard: prevent two pipelines from writing to the same DuckDB simultaneously.
        # Sequential use of the same DB (accumulation, resume) is fine — the second
        # pipeline re-reads MAX(id) at init so counters don't collide.
        # The only unsafe case is two run_async() calls active at the same time.
        from biolm.pipeline.datastore_duckdb import _RUNNING_DB_PATHS
        _db_path_key = str(self.datastore.db_path) if self.datastore else None
        if _db_path_key and _db_path_key in _RUNNING_DB_PATHS:
            raise RuntimeError(
                f"Another pipeline is already running on '{_db_path_key}'.\n"
                "Two pipelines cannot write to the same DuckDB simultaneously — "
                "primary-key counters would collide. Wait for the first run to "
                "finish, or use a different db_path."
            )
        if _db_path_key:
            _RUNNING_DB_PATHS.add(_db_path_key)

        # Create pipeline run record
        config = self._get_config()
        self.datastore.create_pipeline_run(
            run_id=self.run_id,
            pipeline_type=self.pipeline_type,
            config=config,
            status="running",
        )

        try:
            # Save pipeline definition (content-hash dedup) + register output columns.
            # Skip on resume: if this run_id already has a definition_id the definition
            # was already persisted on the first run. Re-saving would create a new
            # definition entry based on the *reconstructed* stage list, which may be
            # incomplete (e.g. from_db() recovery), corrupting pipeline_definitions.
            existing_def = self.datastore.conn.execute(
                "SELECT definition_id FROM pipeline_runs WHERE run_id = ? "
                "AND definition_id IS NOT NULL",
                [self.run_id],
            ).fetchone()
            if existing_def is None:
                self._save_definition_and_register_columns()

            # Get initial data — returns WorkingSet
            ws_current = await self._get_initial_data_ws(**kwargs)

            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"# Pipeline: {self.pipeline_type}")
                print(f"# Run ID: {self.run_id}")
                print(f"# Initial sequences: {len(ws_current):,}")
                if enable_streaming:
                    print("# Streaming: ENABLED")
                print(f"{'#'*60}")

            # Resolve dependencies and get execution order
            stage_levels = self._resolve_dependencies()

            if self.verbose:
                print(f"\nExecution plan: {len(stage_levels)} level(s)")
                for i, level in enumerate(stage_levels):
                    stage_names = [s.name for s in level]
                    parallel_str = " (parallel)" if len(level) > 1 else ""
                    print(f"  Level {i+1}: {', '.join(stage_names)}{parallel_str}")

            # Execute stages level by level
            processed_stages = set()  # Track which stages we've already processed

            for level_idx, level_stages in enumerate(stage_levels):
                if len(level_stages) == 1:
                    # Single stage in level
                    stage = level_stages[0]

                    # Skip if already processed via streaming
                    if stage.name in processed_stages:
                        continue

                    # Check if we can stream through this stage
                    next_stage = self._get_next_stage(stage, stage_levels, level_idx)
                    can_stream = (
                        enable_streaming
                        and next_stage is not None
                        and next_stage.name not in processed_stages
                        and hasattr(stage, "process_streaming")
                        and self._can_stream_to_next(stage, next_stage)
                    )

                    if can_stream:
                        # STREAMING: uses legacy DataFrame path
                        ws_stage_input = self._get_stage_input_ws(stage, ws_current)
                        df_current = self.datastore.materialize_working_set(ws_stage_input)
                        df_out = await self._execute_stage_streaming(
                            stage, next_stage, df_current
                        )
                        # prediction stage WS = all inputs (pre-filter);
                        # filter stage WS = survivors (post-filter).  Using the same
                        # post-filter WS for both caused incorrect resume cache keys.
                        ws_pred = (
                            WorkingSet.from_ids(df_current["sequence_id"].tolist())
                            if "sequence_id" in df_current.columns
                            else ws_current
                        )
                        ws_filter = (
                            WorkingSet.from_ids(df_out["sequence_id"].tolist())
                            if "sequence_id" in df_out.columns
                            else WorkingSet(frozenset())
                        )
                        self._stage_data[stage.name] = df_out
                        self._stage_data[next_stage.name] = df_out
                        self._working_sets[stage.name] = ws_pred
                        self._working_sets[next_stage.name] = ws_filter
                        ws_current = ws_filter
                        processed_stages.add(stage.name)
                        processed_stages.add(next_stage.name)
                    else:
                        # BATCHING: WorkingSet path
                        # resolve input from declared dependencies, not global ws_current
                        ws_stage_input = self._get_stage_input_ws(stage, ws_current)
                        ws_out, result = await self._execute_stage_ws(
                            stage, ws_stage_input
                        )
                        self.stage_results[stage.name] = result
                        self._working_sets[stage.name] = ws_out
                        ws_current = ws_out
                        processed_stages.add(stage.name)
                else:
                    # Multiple stages in level - execute in parallel

                    # mixed merge modes within a single level have undefined
                    # semantics — a prediction stage outputs a (possibly larger) union
                    # while a filter stage outputs a (possibly smaller) intersection.
                    # Applying both simultaneously makes the final merged set ambiguous.
                    # Raise early so the user can fix their pipeline topology.
                    level_merge_modes = {
                        getattr(s, "merge_mode", "intersect") for s in level_stages
                    }
                    if "union" in level_merge_modes and "intersect" in level_merge_modes:
                        raise ValueError(
                            f"Level {level_idx} mixes 'union' (prediction) and "
                            f"'intersect' (filter) stages: "
                            f"{[s.name for s in level_stages]}. "
                            "This has undefined semantics — separate them with "
                            "explicit depends_on."
                        )

                    if self.verbose:
                        print(f"\nExecuting {len(level_stages)} stages in parallel...")

                    tasks = [
                        self._execute_stage_ws(
                            stage, self._get_stage_input_ws(stage, ws_current)
                        )
                        for stage in level_stages
                    ]
                    # use return_exceptions=True so that a failure
                    # in one parallel stage does not cancel all other in-flight stages
                    # via asyncio.gather()'s default cancellation behaviour.
                    # After gathering, we check for exceptions and re-raise the first.
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Check for exceptions before processing results
                    exceptions = [r for r in results if isinstance(r, Exception)]
                    if exceptions:
                        raise exceptions[0]

                    # Parallel merge = set intersection (was: DataFrame merge)
                    all_working_sets = []
                    for stage, (ws_out, result) in zip(level_stages, results):
                        self.stage_results[stage.name] = result
                        self._working_sets[stage.name] = ws_out
                        all_working_sets.append(ws_out)

                    # Parallel merge: union prediction outputs, intersect filter outputs.
                    # PredictionStage/CofoldingPredictionStage use merge_mode="union"
                    # so sequences processed by any prediction stage survive (independent
                    # predictions should not cancel each other out).
                    # FilterStage and ClusteringStage use merge_mode="intersect" (default)
                    # so a sequence must pass *all* parallel filters.
                    if all_working_sets:
                        union_pairs = [
                            ws for s, ws in zip(level_stages, all_working_sets)
                            if getattr(s, "merge_mode", "intersect") == "union"
                        ]
                        intersect_pairs = [
                            ws for s, ws in zip(level_stages, all_working_sets)
                            if getattr(s, "merge_mode", "intersect") == "intersect"
                        ]

                        # Start from the union of all prediction-stage outputs
                        if union_pairs:
                            ws_merged = union_pairs[0]
                            for ws in union_pairs[1:]:
                                ws_merged = WorkingSet(ws_merged.sequence_ids | ws.sequence_ids)
                            # Then restrict to sequences that passed every filter
                            for ws in intersect_pairs:
                                ws_merged = ws_merged.intersect(ws)
                        elif intersect_pairs:
                            ws_merged = intersect_pairs[0]
                            for ws in intersect_pairs[1:]:
                                ws_merged = ws_merged.intersect(ws)
                        else:
                            ws_merged = ws_current  # level had no stages (shouldn't happen)

                        ws_current = ws_merged
                    else:
                        ws_current = WorkingSet(frozenset())

                    # ws_current now holds the merged level result; per-stage
                    # outputs were already saved to _working_sets as each task
                    # completed, so a child stage's _get_stage_input_ws() can
                    # still see the un-merged WorkingSet of its specific dependency.

            self.status = "completed"
            self.end_time = time.time()
            self.datastore.update_pipeline_run_status(self.run_id, "completed")

            # Materialize the final DataFrame now while the connection is still open,
            # so get_final_data() works even after the auto-created datastore is closed.
            if self.stages and ws_current is not None:
                last_name = self.stages[-1].name
                self._final_ws = ws_current
                try:
                    self._final_df = self.datastore.materialize_working_set(ws_current)
                    self._stage_data[last_name] = self._final_df
                except Exception:
                    pass

            if self.verbose:
                total_time = self.end_time - self.start_time
                print(f"\n{'#'*60}")
                print(f"# Pipeline completed in {total_time:.1f}s")
                print(f"# Final sequences: {len(ws_current):,}")
                print(f"{'#'*60}\n")

            return self.stage_results

        except Exception:
            self.status = "failed"
            self.end_time = time.time()
            self.datastore.update_pipeline_run_status(self.run_id, "failed")
            raise

        finally:
            # Always release the running-path lock so the next sequential run can proceed.
            if _db_path_key:
                _RUNNING_DB_PATHS.discard(_db_path_key)
            # NOTE: We intentionally do NOT auto-close the datastore here.
            # Post-run methods (query_results, explore, stats, get_final_data,
            # run(resume=True)) all need the datastore alive.  Closing happens
            # in close(), __exit__, __aexit__, or __del__.

    async def _get_initial_data_ws(self, **kwargs) -> WorkingSet:
        """Get initial WorkingSet for the pipeline.

        Default implementation calls the legacy _get_initial_data() and converts.
        Subclasses can override for a more efficient direct path.
        """
        df = await self._get_initial_data(**kwargs)
        if "sequence_id" in df.columns:
            return WorkingSet.from_ids(df["sequence_id"].tolist())
        return WorkingSet(frozenset())

    def _get_next_stage(
        self, current_stage: Stage, stage_levels: list[list[Stage]], current_level: int
    ) -> Optional[Stage]:
        """Get the next stage after current_stage, if any."""
        if current_level + 1 >= len(stage_levels):
            return None

        next_level = stage_levels[current_level + 1]
        if len(next_level) == 1:
            return next_level[0]
        return None  # Can't stream to multiple parallel stages

    def _can_stream_to_next(self, current_stage: Stage, next_stage: Stage) -> bool:
        """Check if current stage can stream to next stage.

        Only FilterStage is supported as a streaming target.  Streaming to a
        PredictionStage is intentionally disabled: _execute_stage_streaming()
        only handles FilterStage, and allowing PredictionStage here would cause
        chunks to be silently dropped.
        """
        from biolm.pipeline.data import FilterStage

        # Can stream only to a filter that doesn't require complete data
        if isinstance(next_stage, FilterStage):
            return not next_stage.requires_complete_data

        return False

    async def _execute_stage_streaming(
        self, stage: Stage, next_stage: Stage, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Execute stage in streaming mode, passing results to next_stage incrementally.

        Supports within-stage checkpointing for crash recovery: after every
        chunk, filter results are written to DuckDB and a progress checkpoint
        is updated.  On resume with ``self.resume=True``, already-processed
        sequences are skipped so only the remaining tail is re-streamed.

        Returns:
            Final output DataFrame after both stages (only ``sequence_id``
            column is guaranteed; prediction columns depend on chunk content).
        """
        import json as _json

        from biolm.pipeline.data import FilterStage

        if self.verbose:
            print(f"\n[Stage: {stage.name}] (streaming to {next_stage.name})")

        # ------------------------------------------------------------------
        # STREAMING RESUME: reload checkpoint and skip already-done sequences.
        # Checkpoint key stores the list of sequence_ids that have completed
        # both the prediction step and the filter step in a previous partial run.
        # ------------------------------------------------------------------
        _checkpoint_key = f"_stream_done_{self.run_id}_{stage.name}"
        _already_done_ids: set[int] = set()
        _pre_passed_ids: set[int] = set()
        _full_input_count = len(df)

        if self.resume and self.datastore and "sequence_id" in df.columns:
            _raw_checkpoint = self.datastore.get_pipeline_metadata(_checkpoint_key)
            if _raw_checkpoint:
                _already_done_ids = set(_raw_checkpoint)
                # Sequences from the previous partial run that passed the filter
                _pre_passed_ids = set(
                    self.datastore.get_filter_results(self.run_id, next_stage.name)
                ) & _already_done_ids
                # Only stream the sequences not yet processed
                df = df[~df["sequence_id"].isin(_already_done_ids)].reset_index(drop=True)
                if self.verbose:
                    print(
                        f"  Resuming stream: {len(_already_done_ids)} sequences already "
                        f"checkpointed, {len(df)} remaining"
                    )

        _all_processed_ids: list[int] = list(_already_done_ids)
        output_chunks: list[pd.DataFrame] = []
        processed_count = len(_pre_passed_ids)
        filtered_count = len(_already_done_ids) - len(_pre_passed_ids)

        # ------------------------------------------------------------------
        # Stream through both stages, persisting results after every chunk.
        # ------------------------------------------------------------------
        async for chunk_df in stage.process_streaming(df, self.datastore):
            if isinstance(next_stage, FilterStage):
                # Guard: filters that need all data cannot run in streaming mode
                if getattr(
                    getattr(next_stage, "filter_func", None),
                    "requires_complete_data",
                    False,
                ):
                    raise RuntimeError(
                        f"Filter '{next_stage.filter_func}' requires complete data and "
                        "cannot be used in streaming mode. Remove enable_streaming=True "
                        "or use a per-sequence filter."
                    )
                start_chunk_count = len(chunk_df)
                filtered_chunk = next_stage.filter_func(chunk_df)
                filtered_count += start_chunk_count - len(filtered_chunk)

                # --- INCREMENTAL PERSISTENCE (checkpoint) ---
                _chunk_ids = (
                    chunk_df["sequence_id"].tolist()
                    if "sequence_id" in chunk_df.columns
                    else []
                )
                _chunk_passed = (
                    filtered_chunk["sequence_id"].tolist()
                    if "sequence_id" in filtered_chunk.columns
                    else []
                )
                if _chunk_passed and self.datastore:
                    # Write filter results for this chunk immediately so a crash
                    # after this point won't lose them (ON CONFLICT DO NOTHING is
                    # idempotent, so safe to call again on resume).
                    self.datastore.save_filter_results(
                        self.run_id, next_stage.name, _chunk_passed
                    )
                if _chunk_ids and self.datastore:
                    # Update checkpoint: record all sequence_ids fully processed
                    # so far (both prediction and filter).
                    _all_processed_ids.extend(_chunk_ids)
                    self.datastore.set_pipeline_metadata(
                        _checkpoint_key, _all_processed_ids
                    )
                # --- END INCREMENTAL PERSISTENCE ---

                if filtered_chunk is not None and len(filtered_chunk) > 0:
                    output_chunks.append(filtered_chunk)
                    processed_count += len(filtered_chunk)

                    if self.verbose and processed_count % 100 == 0:
                        print(f"  Processed: {processed_count} sequences (streaming)")

        # ------------------------------------------------------------------
        # Assemble final DataFrame: include pre-checkpointed survivors.
        # (Only sequence_id column is needed downstream — run_async() converts
        # to WorkingSet immediately after this returns.)
        # ------------------------------------------------------------------
        if _pre_passed_ids:
            output_chunks.insert(
                0, pd.DataFrame({"sequence_id": sorted(_pre_passed_ids)})
            )

        if output_chunks:
            df_out = pd.concat(output_chunks, ignore_index=True)
        else:
            df_out = pd.DataFrame(columns=["sequence_id"])

        # Record results for both stages
        self.stage_results[stage.name] = StageResult(
            stage_name=stage.name,
            input_count=_full_input_count,
            output_count=_full_input_count,
            filtered_count=0,
        )
        self.stage_results[next_stage.name] = StageResult(
            stage_name=next_stage.name,
            input_count=_full_input_count,
            output_count=len(df_out),
            filtered_count=filtered_count,
        )

        # Safety net: save any filter results not yet persisted incrementally.
        # ON CONFLICT DO NOTHING makes this idempotent.
        if (
            self.datastore
            and isinstance(next_stage, FilterStage)
            and "sequence_id" in df_out.columns
        ):
            self.datastore.save_filter_results(
                self.run_id, next_stage.name, df_out["sequence_id"].tolist()
            )

        # Mark both stages complete
        if self.datastore:
            stage_id = f"{self.run_id}_{stage.name}"
            self.datastore.mark_stage_complete(
                run_id=self.run_id,
                stage_name=stage.name,
                stage_id=stage_id,
                input_count=_full_input_count,
                output_count=_full_input_count,
                status="completed",
            )
            next_stage_id = f"{self.run_id}_{next_stage.name}"
            self.datastore.mark_stage_complete(
                run_id=self.run_id,
                stage_name=next_stage.name,
                stage_id=next_stage_id,
                input_count=_full_input_count,
                output_count=len(df_out),
                status="completed",
            )
            # Clear checkpoint now that both stages completed successfully.
            self.datastore.conn.execute(
                "DELETE FROM pipeline_metadata WHERE key = ?", [_checkpoint_key]
            )

        if self.verbose:
            print(f"  {stage.name}: processed {_full_input_count} sequences")
            print(
                f"  {next_stage.name}: {len(df_out)} passed filter "
                f"(filtered {filtered_count})"
            )

        return df_out

    def run(self, enable_streaming: bool = True, **kwargs) -> dict[str, StageResult]:
        """
        Run the pipeline synchronously.

        This is a convenience wrapper around run_async(). Works in both
        script/notebook environments (detects running event loops).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside a running event loop (e.g. Jupyter).
            try:
                import nest_asyncio
                global _nest_asyncio_applied
                if not _nest_asyncio_applied:
                    nest_asyncio.apply()
                    _nest_asyncio_applied = True
            except ImportError:
                import warnings
                warnings.warn(
                    "pipeline.run() called from inside a running event loop (e.g. Jupyter). "
                    "Use 'await pipeline.run_async()' instead for proper async execution, "
                    "or install nest_asyncio: pip install nest_asyncio",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # if nest_asyncio is not installed and we are inside a
                # running event loop, loop.run_until_complete() will deadlock.
                # Raise a clear, actionable error rather than hanging silently.
                if loop.is_running():
                    raise RuntimeError(
                        "nest_asyncio is required to run pipelines inside Jupyter. "
                        "Install it with: pip install nest_asyncio"
                    )
            return loop.run_until_complete(
                self.run_async(enable_streaming=enable_streaming, **kwargs)
            )
        return asyncio.run(self.run_async(enable_streaming=enable_streaming, **kwargs))

    @abstractmethod
    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """
        Get initial DataFrame for the pipeline.

        Must return DataFrame with at least 'sequence' column.
        Should add 'sequence_id' column by inserting into datastore.
        """
        pass

    def _get_config(self) -> dict:
        """Get pipeline configuration for serialization."""
        return {
            "pipeline_type": self.pipeline_type,
            "run_id": self.run_id,
            "stages": [
                {
                    "name": s.name,
                    "type": s.__class__.__name__,
                    "depends_on": s.depends_on,
                }
                for s in self.stages
            ],
        }

    def results(self) -> pd.DataFrame:
        """Return the final output DataFrame.  Alias for :meth:`get_final_data`.

        The returned DataFrame always contains at minimum:

        - ``sequence_id``, ``sequence``, ``length``, ``hash``
        - ``source_label`` — the label set on the generation config
          (e.g. ``DirectGenerationConfig.label``).  ``None`` when no label
          was set.  **Always present** — do not guard on
          ``"source_label" in df.columns``.
        - One column per prediction type computed by any prediction stages.
        """
        return self.get_final_data()

    def get_final_data(self) -> pd.DataFrame:
        """Get the final output DataFrame.

        Materializes from the merged final WorkingSet via DuckDB.  In a
        branched DAG the last-added stage is not necessarily the last
        executed sink, so we prefer ``_final_ws`` (set at end of run()).

        The returned DataFrame always contains at minimum:

        - ``sequence_id``, ``sequence``, ``length``, ``hash``
        - ``source_label`` — label from the generation config; ``None`` when
          unset.  **Always present** regardless of whether any labels were set.
        - One column per prediction type from upstream prediction stages.
        """
        if not self._working_sets and not self._stage_data and self._final_df is None:
            raise RuntimeError("Pipeline has not been run yet")

        if not self.stages:
            raise RuntimeError("No stages defined in this pipeline.")

        datastore_alive = (
            self.datastore is not None
            and getattr(self.datastore, "conn", None) is not None
        )

        if self._final_ws is not None and datastore_alive:
            return self.datastore.materialize_working_set(self._final_ws)

        if self._final_df is not None:
            return self._final_df

        last_stage_name = self.stages[-1].name
        if last_stage_name in self._working_sets and datastore_alive:
            ws = self._working_sets[last_stage_name]
            return self.datastore.materialize_working_set(ws)

        return self._stage_data.get(last_stage_name, pd.DataFrame())

    # ------------------------------------------------------------------
    # Context manager support — ensures DuckDB connection is closed
    # whether the user calls .run() explicitly or not (e.g. in notebooks).
    # ------------------------------------------------------------------

    def close(self):
        """Close the pipeline's datastore connection.

        Safe to call multiple times.  Called automatically by ``__exit__``
        and ``__aexit__``.  Also called by ``__del__`` for auto-created
        datastores only (user-provided datastores are not closed on GC so
        the caller can continue using them after the pipeline is discarded).
        """
        if self.datastore:
            try:
                self.datastore.close()
            except Exception:
                pass

    def __del__(self):
        """Close the datastore on garbage collection, but only if this pipeline
        created it.  User-provided datastores are left open so the caller can
        continue using them after the pipeline object is garbage-collected.
        """
        if getattr(self, "_auto_created_datastore", False):
            self.close()

    def __enter__(self) -> "BasePipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Close the datastore on exit from a `with` block."""
        self.close()
        return False

    async def __aenter__(self) -> "BasePipeline":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Close the datastore on exit from an `async with` block."""
        self.close()
        return False

    def query(self, sql: str, params=None) -> pd.DataFrame:
        """Execute arbitrary SQL against the pipeline's DuckDB datastore."""
        return self.datastore.query(sql, params)

    def export_to_csv(self, output_path: Optional[Union[str, Path]] = None):
        """Export final results to CSV."""
        if output_path is None:
            output_path = self.output_dir / f"{self.run_id}_final.csv"

        df = self.get_final_data()
        df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"Exported {len(df)} sequences to {output_path}")

    def summary(self) -> pd.DataFrame:
        """Get pipeline summary statistics."""
        if not self.stage_results:
            print("Pipeline has not been run yet")
            return pd.DataFrame()

        summary_data = []
        for _stage_name, result in self.stage_results.items():
            summary_data.append(
                {
                    "Stage": result.stage_name,
                    "Input": result.input_count,
                    "Output": result.output_count,
                    "Filtered": result.filtered_count,
                    "Cached": result.cached_count,
                    "Computed": result.computed_count,
                    "Time (s)": f"{result.elapsed_time:.1f}",
                }
            )

        return pd.DataFrame(summary_data)

    def __repr__(self):
        return (
            f"{self.pipeline_type}("
            f"run_id='{self.run_id}', "
            f"stages={len(self.stages)}, "
            f"status='{self.status}')"
        )
