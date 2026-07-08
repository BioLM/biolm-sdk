"""
DuckDB-based DataStore with Parquet backend for efficient large-scale data management.

Key features:
- Columnar storage (Parquet)
- Out-of-core queries (bigger than RAM)
- Vectorized anti-join deduplication
- 5-50× faster than pandas for aggregations
- Diff-mode friendly batch inserts
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re as _re
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, Optional, Union

import duckdb
import numpy as np
import pandas as pd

_SAFE_COL_RE = _re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Module-level counter for unique temp table names can be reused by GC)
import itertools as _itertools
_TEMP_TABLE_COUNTER = _itertools.count()

# Process-level set of DuckDB paths currently being WRITTEN TO by a running pipeline.
# Two datastores on the same path can exist sequentially (accumulation / resume) —
# the second one re-reads MAX(id) at init so its counter starts correctly.
# The only dangerous case is two pipelines writing SIMULTANEOUSLY: both init their
# counters from the same MAX(id) before either writes, causing primary-key collisions.
# This set is checked at run_async() entry, not at DuckDBDataStore.__init__ time.
_RUNNING_DB_PATHS: set[str] = set()


def _validate_column_name(col: str) -> None:
    """Raise ValueError if *col* is not a safe SQL identifier."""
    if not _SAFE_COL_RE.match(col):
        raise ValueError(
            f"Invalid column name '{col}': must match ^[a-zA-Z_][a-zA-Z0-9_]*$"
        )

if TYPE_CHECKING:
    from biolm.pipeline.base import WorkingSet


class DuckDBDataStore:
    """
    DuckDB + Parquet based datastore for efficient sequence management.

    Optimized for:
    - Millions of sequences
    - Complex queries (joins, filters, aggregations)
    - Out-of-core operations (bigger than RAM)
    - Diff mode with efficient deduplication

    Args:
        db_path: Path to DuckDB database file
        data_dir: Directory for Parquet files and large data

    Example:
        >>> ds = DuckDBDataStore("pipeline.db", "data/")
        >>> seq_id = ds.add_sequence("MKLLIV")
        >>> ds.add_prediction(seq_id, "tm", "temberture-regression", 65.5)
        >>>
        >>> # Efficient query - no memory explosion
        >>> high_tm = ds.query("SELECT * FROM predictions WHERE value > 60")
    """

    def __init__(
        self,
        db_path: Union[str, Path, None] = None,
        data_dir: Union[str, Path, None] = None,
    ):
        # Default DB lives in ~/.biolm/pipelines/, NOT the current working
        # directory.  The previous default (`./pipeline.duckdb`) caused
        # accidental git commits and dropped sensitive sequence data into
        # whatever folder the user happened to launch Python from.  Callers
        # who want a project-local DB should pass `db_path="./my.duckdb"`
        # explicitly.
        if db_path is None:
            from biolm.core.paths import resolve_user_path

            default_root = resolve_user_path("pipelines")
            default_root.mkdir(parents=True, exist_ok=True)
            db_path = default_root / "default.duckdb"
            warnings.warn(
                f"DuckDBDataStore using default db_path={db_path!s}.  "
                f"Sequences and predictions will persist here across runs; "
                f"pass an explicit db_path= to control the location.",
                stacklevel=2,
            )
        if data_dir is None:
            data_dir = Path(db_path).resolve().parent / (Path(db_path).stem + "_data")
        self.db_path = Path(db_path).resolve()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create Parquet directories
        self.sequences_dir = self.data_dir / "sequences"
        self.predictions_dir = self.data_dir / "predictions"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.structures_dir = self.data_dir / "structures"

        for d in [
            self.sequences_dir,
            self.predictions_dir,
            self.embeddings_dir,
            self.structures_dir,
        ]:
            d.mkdir(exist_ok=True)

        # Connect — DuckDB holds an exclusive file lock when open in read-write mode,
        # so a second *process* opening the same file gets a clear OS error.
        try:
            self.conn = duckdb.connect(str(self.db_path))
        except Exception as exc:
            exc_str = str(exc)
            if "Could not set lock" in exc_str or "lock" in exc_str.lower():
                raise RuntimeError(
                    f"DuckDB at '{self.db_path}' is locked by another process.\n"
                    "Only one pipeline can write to a given database at a time. "
                    "Wait for the other process to finish, or use a different db_path."
                ) from exc
            # WAL corruption (e.g. from kernel restart mid-write) — delete the
            # WAL file and retry. The main .duckdb file is still valid; only
            # uncommitted transactions from the WAL are lost.
            wal_path = Path(str(self.db_path) + ".wal")
            if wal_path.exists() and ("WAL" in exc_str or "replay" in exc_str.lower()):
                logger.warning(
                    "Corrupted WAL file detected at %s — removing and retrying. "
                    "Uncommitted transactions may be lost.",
                    wal_path,
                )
                wal_path.unlink()
                self.conn = duckdb.connect(str(self.db_path))
            else:
                raise

        # Initialize schema
        self._init_schema()
        self._maybe_migrate_hashes_v2()

    def _init_schema(self):
        """Initialize DuckDB tables backed by Parquet files."""

        # Sequences table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_id INTEGER PRIMARY KEY,
                sequence VARCHAR NOT NULL,
                length INTEGER,
                hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create unique index for deduplication
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sequence_hash
            ON sequences(hash)
        """
        )

        # Predictions table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                prediction_type VARCHAR,
                value DOUBLE,
                metadata VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (sequence_id, prediction_type, model_name)
            )
        """
        )

        # Indexes for fast queries
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pred_seq
            ON predictions(sequence_id)
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pred_type
            ON predictions(prediction_type, model_name)
        """
        )

        # Embeddings table — arrays stored inline as FLOAT[] (no per-file Parquet overhead)
        # UNIQUE(sequence_id, model_name, layer) to enable ON CONFLICT upsert
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                layer INTEGER,
                values FLOAT[],
                dimension INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (sequence_id, model_name, layer)
            )
        """
        )
        # Structures table — stores inline structure content (PDB/CIF strings)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structures (
                structure_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                format VARCHAR,
                structure_path VARCHAR,
                structure_str TEXT,
                plddt DOUBLE,
                metadata VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_struct_seq_model
            ON structures(sequence_id, model_name)
        """
        )
        # Migration: add structure_str column to existing databases
        try:
            self.conn.execute("ALTER TABLE structures ADD COLUMN structure_str TEXT")
        except duckdb.CatalogException:
            pass  # Column already exists
        # Migration: add compressed structure_data BLOB column (replaces uncompressed TEXT)
        try:
            self.conn.execute("ALTER TABLE structures ADD COLUMN structure_data BLOB")
        except duckdb.CatalogException:
            pass  # Column already exists

        # Pipeline runs table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id VARCHAR PRIMARY KEY,
                pipeline_type VARCHAR,
                config VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """
        )

        # Stage completions table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_completions (
                stage_id VARCHAR PRIMARY KEY,
                run_id VARCHAR,
                stage_name VARCHAR,
                status VARCHAR,
                input_count INTEGER,
                output_count INTEGER,
                completed_at TIMESTAMP
            )
        """
        )

        # Pipeline metadata (clustering results, stage diagnostics, etc.)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Generation metadata (parameters used to produce each generated sequence)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_metadata (
                metadata_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                run_id VARCHAR NOT NULL DEFAULT '',
                model_name VARCHAR,
                temperature DOUBLE,
                top_k INTEGER,
                top_p DOUBLE,
                num_return_sequences INTEGER,
                do_sample BOOLEAN,
                repetition_penalty DOUBLE,
                max_length INTEGER,
                sampling_params VARCHAR,
                label VARCHAR,
                metadata VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (sequence_id, run_id)
            )
        """
        )
        # GEN-04: add sampling_params column to existing DBs that pre-date this schema change
        try:
            self.conn.execute(
                "ALTER TABLE generation_metadata ADD COLUMN IF NOT EXISTS sampling_params VARCHAR"
            )
        except Exception:
            pass
        # Add label column to existing DBs
        try:
            self.conn.execute(
                "ALTER TABLE generation_metadata ADD COLUMN IF NOT EXISTS label VARCHAR"
            )
        except Exception as _migration_err:
            logger.warning(
                "generation_metadata schema migration failed: could not add 'label' column "
                "(%s). Subsequent generation_metadata inserts that include a label will "
                "fail with a schema error. Check that the DuckDB file is writable.",
                _migration_err,
            )
        # Add metadata column to existing DBs (DMS provenance JSON blob)
        try:
            self.conn.execute(
                "ALTER TABLE generation_metadata ADD COLUMN IF NOT EXISTS metadata VARCHAR"
            )
        except Exception as _migration_err:
            logger.warning(
                "generation_metadata schema migration failed: could not add 'metadata' column "
                "(%s). DMS provenance will not be persisted for existing databases.",
                _migration_err,
            )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_gen_meta_seq
            ON generation_metadata(sequence_id)
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_gen_meta_run
            ON generation_metadata(run_id)
        """
        )

        # Initialize in-memory counters for primary key generation.
        # Bug Q note: these Python int counters are safe under asyncio cooperative
        # multitasking because asyncio is single-threaded — counter increments are
        # never interleaved with another coroutine.  They are NOT safe under
        # multi-process or multi-threaded access (use DuckDB SEQUENCE objects instead).
        result = self.conn.execute("SELECT MAX(sequence_id) FROM sequences").fetchone()
        self._sequence_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(prediction_id) FROM predictions"
        ).fetchone()
        self._prediction_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(embedding_id) FROM embeddings"
        ).fetchone()
        self._embedding_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(structure_id) FROM structures"
        ).fetchone()
        self._structure_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(metadata_id) FROM generation_metadata"
        ).fetchone()
        self._generation_metadata_counter = (result[0] or 0) + 1

        # Filter results table — records which sequence_ids passed each filter stage
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS filter_results (
                filter_id    INTEGER PRIMARY KEY,
                run_id       TEXT NOT NULL,
                stage_name   TEXT NOT NULL,
                sequence_id  INTEGER NOT NULL,
                passed       BOOLEAN NOT NULL DEFAULT TRUE,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_filter_results_lookup
            ON filter_results(run_id, stage_name)
        """
        )
        # Unique constraint to prevent duplicate (run_id, stage_name, sequence_id) rows
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_filter_results_unique
            ON filter_results(run_id, stage_name, sequence_id)
        """
        )
        result = self.conn.execute(
            "SELECT MAX(filter_id) FROM filter_results"
        ).fetchone()
        self._filter_id_counter = (result[0] or 0) + 1

        # Sequence attributes table — stores extra per-sequence columns from
        # the input DataFrame (e.g. heavy_chain, light_chain for antibody models).
        # UNIQUE(sequence_id, attr_name) is enforced via PRIMARY KEY (same semantics).
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sequence_attributes (
                sequence_id  INTEGER NOT NULL,
                attr_name    TEXT NOT NULL,
                attr_value   TEXT,
                PRIMARY KEY (sequence_id, attr_name),
                UNIQUE (sequence_id, attr_name)
            )
        """
        )

        # Pipeline context table — inter-stage shared key-value store
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_context (
                run_id  TEXT NOT NULL,
                key     TEXT NOT NULL,
                value   TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, key)
            )
        """
        )

        # Pipeline definitions table — content-hashed, one row per unique stage config
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_definitions (
                definition_id VARCHAR PRIMARY KEY,
                pipeline_type VARCHAR NOT NULL,
                input_schema_json TEXT,
                stages_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Large-value store for pipeline definition blobs (e.g. PDB strings).
        # stages_json stores {"_blob_ref": blob_id} instead of the raw content.
        # Content-addressed: same PDB in two pipelines → one row.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_blobs (
                blob_id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Prediction column registry — enforces (column, model, action) uniqueness per DB
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_column_registry (
                column_name VARCHAR NOT NULL,
                model_name VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                definition_id VARCHAR NOT NULL,
                stage_name VARCHAR NOT NULL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (column_name, model_name, action)
            )
        """
        )

        # Add definition_id to pipeline_runs (migration — safe to run on existing DBs)
        try:
            self.conn.execute(
                "ALTER TABLE pipeline_runs ADD COLUMN definition_id VARCHAR"
            )
        except duckdb.CatalogException:
            pass  # Column already exists

        # Track which extra columns have been added to the sequences table
        self._extra_columns: set[str] = set()
        # Discover columns already present (for resume / re-open)
        cols_info = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sequences'"
        ).fetchall()
        base_cols = {"sequence_id", "sequence", "length", "hash", "created_at"}
        for (col_name,) in cols_info:
            if col_name not in base_cols:
                self._extra_columns.add(col_name)

    def _refresh_extra_columns(self) -> None:
        """Re-read extra sequence columns from the DB."""
        cols_info = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sequences'"
        ).fetchall()
        base_cols = {"sequence_id", "sequence", "length", "hash", "created_at"}
        self._extra_columns = {col_name for (col_name,) in cols_info if col_name not in base_cols}

    @staticmethod
    def _hash_sequence(sequence: str) -> str:
        """Create hash of sequence for deduplication."""
        return hashlib.sha256(sequence.encode()).hexdigest()[:32]

    @contextmanager
    def _transaction(self):
        """Wrap a block in begin/commit/rollback so a partial INSERT can't
        leak partially-written rows on error."""
        self.conn.begin()
        try:
            yield
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def _maybe_migrate_hashes_v2(self) -> None:
        """Widen legacy 16-char hashes to 32 chars (C9 backward-compat).

        Without this, a DB written by a pre-C9 version stores 16-char hashes
        and post-C9 lookups produce 32-char keys that miss every row, leaving
        sequences silently re-added as duplicates.  Idempotent — checks a
        version key in pipeline_metadata so subsequent opens are a no-op.
        """
        try:
            row = self.conn.execute(
                "SELECT value FROM pipeline_metadata WHERE key = ?",
                ["hash_schema_version"],
            ).fetchone()
            if row and str(row[0]) >= "2":
                return
        except Exception:
            return  # Table not ready yet; safe to retry on next open

        try:
            short = self.conn.execute(
                "SELECT COUNT(*) FROM sequences WHERE hash IS NOT NULL AND LENGTH(hash) < 32"
            ).fetchone()
        except Exception:
            return

        if short and short[0] > 0:
            # Single-column path — most common, easy to recompute.
            sequence_rows = self.conn.execute(
                "SELECT sequence_id, sequence FROM sequences "
                "WHERE LENGTH(hash) < 32 AND COALESCE(sequence, '') != ''"
            ).fetchall()
            for sid, seq in sequence_rows:
                new_hash = self._hash_sequence(seq)
                self.conn.execute(
                    "UPDATE sequences SET hash = ? WHERE sequence_id = ?",
                    [new_hash, sid],
                )
            # Multi-column rows (sequence='') are left at 16 chars — recomputing
            # would require iterating the dynamic input columns and is rare;
            # callers see at most one extra duplicate row per legacy multi-col
            # input.  Log for visibility.
            try:
                multi_col_left = self.conn.execute(
                    "SELECT COUNT(*) FROM sequences "
                    "WHERE LENGTH(hash) < 32 AND COALESCE(sequence, '') = ''"
                ).fetchone()
                if multi_col_left and multi_col_left[0] > 0:
                    logger.warning(
                        "Hash schema migration: %d multi-column rows still hold "
                        "16-char hashes.  Re-adding identical multi-column inputs "
                        "may insert duplicates until you re-run with sequence text.",
                        multi_col_left[0],
                    )
            except Exception:
                pass

        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO pipeline_metadata (key, value) VALUES (?, ?)",
                ["hash_schema_version", "2"],
            )
        except Exception:
            pass

    @staticmethod
    def _hash_row(values: dict[str, str]) -> str:
        """Hash multiple column values for multi-column deduplication.

        Columns are sorted alphabetically; values joined with ``\\x00``.

        Raises:
            ValueError: If any value is None or NaN — these cannot be hashed
                safely for deduplication (would silently hash to "None"/"nan").
        """
        parts = []
        for c in sorted(values.keys()):
            val = values.get(c, "")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                raise ValueError(
                    f"Input column '{c}' contains None/NaN values which cannot "
                    f"be hashed for deduplication. Fill or drop these rows "
                    f"before adding to the pipeline."
                )
            parts.append(str(val))
        payload = "\x00".join(parts)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def ensure_input_columns(self, columns: list[str]):
        """Ensure the ``sequences`` table has all the given columns.

        Uses ``ALTER TABLE ADD COLUMN`` for any that don't already exist.
        This is idempotent — safe to call on every pipeline run.

        Raises:
            duckdb.CatalogException: If column creation fails for an unexpected
                reason (e.g. type conflict). The "column already exists" case is
                silently ignored as it is the normal idempotent path.
        """
        for col in columns:
            _validate_column_name(col)
            if col in self._extra_columns:
                continue
            # Don't add columns that are part of the base schema
            if col in ("sequence_id", "sequence", "length", "hash", "created_at"):
                continue
            try:
                self.conn.execute(
                    f'ALTER TABLE sequences ADD COLUMN "{col}" TEXT'
                )
            except duckdb.CatalogException as exc:
                # Only suppress the "column already exists" case.
                # Any other CatalogException (e.g. type conflict) propagates.
                if "already exists" not in str(exc).lower():
                    raise
            self._extra_columns.add(col)

    def add_sequences_batch(
        self,
        sequences: Optional[list[str]] = None,
        deduplicate: bool = True,
        input_df: Optional[pd.DataFrame] = None,
        input_columns: Optional[list[str]] = None,
    ) -> list[int]:
        """
        Add multiple sequences efficiently using anti-join deduplication.

        This is the RECOMMENDED way to add sequences - vectorized and fast!

        Two calling conventions:

        1. **Legacy** (sequence-only):
           ``add_sequences_batch(["MKLLIV", ...])``

        2. **Multi-column** (arbitrary input columns):
           ``add_sequences_batch(input_df=df, input_columns=["heavy_chain", "light_chain"])``
           In this mode, the hash is computed across all input columns, and
           the column values are stored directly on the ``sequences`` table.
           A ``sequence`` column is still written (concatenation of all
           input columns joined with ``:``) so downstream code has a fallback.

        Args:
            sequences: List of sequence strings (legacy path).
            deduplicate: Use anti-join to skip existing sequences.
            input_df: DataFrame with input columns (multi-column path).
            input_columns: Column names in *input_df* to use as primary data.

        Returns:
            List of sequence_ids (new and existing), preserving input order.
        """
        # ----- Multi-column path ------------------------------------------------
        if input_df is not None and input_columns is not None:
            if input_df.empty:
                return []

            # Reject reserved column names that would collide with base schema columns
            _RESERVED = {"sequence_id", "hash", "length", "created_at"}
            reserved_clash = [c for c in input_columns if c in _RESERVED]
            if reserved_clash:
                raise ValueError(
                    f"input_columns contains reserved column name(s): {reserved_clash}. "
                    f"The names 'sequence_id', 'hash', 'length', and 'created_at' are "
                    f"reserved by the datastore schema."
                )

            # Validate input columns exist in DataFrame
            missing = [c for c in input_columns if c not in input_df.columns]
            if missing:
                raise ValueError(
                    f"input_df missing columns: {missing}. "
                    f"Available: {list(input_df.columns)}"
                )

            # Ensure sequences table has the needed columns
            self.ensure_input_columns(input_columns)

            # Build df_new with hash across all input columns
            df_new = input_df[input_columns].copy()
            # validate None/NaN before hashing (vectorized check)
            for _hcol in input_columns:
                if df_new[_hcol].isna().any():
                    raise ValueError(
                        f"Input column '{_hcol}' contains None/NaN values which cannot "
                        f"be hashed for deduplication. Fill or drop these rows "
                        f"before adding to the pipeline."
                    )
            _sorted_hcols = sorted(input_columns)
            _combined = (
                df_new[_sorted_hcols]
                .astype(str)
                .agg("\x00".join, axis=1)
            )
            df_new["hash"] = [
                hashlib.sha256(v.encode()).hexdigest()[:32] for v in _combined
            ]
            # Set 'sequence' column: use real value if provided, else empty string
            # (column is NOT NULL in schema). Real data lives in input columns.
            if "sequence" in input_columns:
                df_new["sequence"] = input_df["sequence"]
                df_new["length"] = df_new["sequence"].str.len()
            else:
                df_new["sequence"] = ""
                df_new["length"] = sum(
                    df_new[c].astype(str).str.len() for c in input_columns
                )

            # Dedup within batch
            original_hashes = df_new["hash"].tolist()
            df_new = df_new.drop_duplicates(subset=["hash"], keep="first")

            self.conn.register("_seq_batch", df_new)
            try:
                if deduplicate:
                    df_to_insert = self.conn.execute(
                        """
                        SELECT s.*
                        FROM _seq_batch s
                        LEFT JOIN sequences t ON s.hash = t.hash
                        WHERE t.hash IS NULL
                    """
                    ).df()
                else:
                    df_to_insert = df_new.copy()

                if len(df_to_insert) > 0:
                    start_id = self._sequence_counter
                    df_to_insert["sequence_id"] = range(
                        start_id, start_id + len(df_to_insert)
                    )
                    df_to_insert["created_at"] = datetime.now()

                    # Build column list for INSERT
                    # second defense-in-depth _validate_column_name check
                    base_cols = ["sequence_id", "sequence", "length", "hash", "created_at"]
                    extra = [c for c in input_columns if c not in base_cols and c != "sequence"]
                    for _c in extra:
                        _validate_column_name(_c)
                    all_cols = base_cols + extra
                    col_list = ", ".join(f'"{c}"' for c in all_cols)
                    sel_list = ", ".join(f'"{c}"' for c in all_cols)

                    _insert_tmp = f"_seq_insert_{next(_TEMP_TABLE_COUNTER)}"
                    self.conn.register(_insert_tmp, df_to_insert)
                    # wrap INSERT + SELECT-back in an explicit transaction
                    try:
                        self.conn.begin()
                        self.conn.execute(
                            f"INSERT INTO sequences ({col_list}) SELECT {sel_list} FROM {_insert_tmp}"
                        )
                        self.conn.commit()
                        # resync counter from DB after commit to handle any gaps.
                        # Counter represents "next available ID", so set to MAX + 1.
                        max_id_row = self.conn.execute(
                            "SELECT MAX(sequence_id) FROM sequences"
                        ).fetchone()
                        if max_id_row and max_id_row[0] is not None:
                            self._sequence_counter = max_id_row[0] + 1
                    except Exception:
                        self.conn.rollback()
                        raise
                    finally:
                        self.conn.unregister(_insert_tmp)

                df_result = self.conn.execute(
                    """
                    SELECT hash, sequence_id
                    FROM sequences
                    WHERE hash IN (SELECT hash FROM _seq_batch)
                """
                ).df()
            finally:
                self.conn.unregister("_seq_batch")

            hash_to_id = dict(zip(df_result["hash"], df_result["sequence_id"]))
            return [hash_to_id[h] for h in original_hashes]

        # ----- Legacy single-column path ----------------------------------------
        if not sequences:
            return []

        # Validate inputs before hashing — None/NaN/non-string inputs would
        # raise an opaque TypeError from `len()` or `.encode()`.  Check up
        # front and report the offending row indices.
        bad: list[int] = []
        for i, s in enumerate(sequences):
            if s is None or not isinstance(s, str):
                bad.append(i)
            elif s == "":
                bad.append(i)
        if bad:
            preview = bad[:5]
            tail = "..." if len(bad) > 5 else ""
            raise ValueError(
                f"add_sequences_batch received {len(bad)} invalid 'sequence' "
                f"values (None / empty / non-string) at row index(es) "
                f"{preview}{tail}.  Drop or fill these rows before calling."
            )

        df_new = pd.DataFrame(
            {
                "sequence": sequences,
                "length": [len(s) for s in sequences],
                "hash": [self._hash_sequence(s) for s in sequences],
            }
        )

        # Remove duplicates within the batch
        df_new = df_new.drop_duplicates(subset=["hash"], keep="first")

        # Register df_new explicitly — DuckDB scope introspection is fragile in async
        self.conn.register("_seq_batch", df_new)
        try:
            if deduplicate:
                # Anti-join: find sequences NOT already in database
                df_to_insert = self.conn.execute(
                    """
                    SELECT s.*
                    FROM _seq_batch s
                    LEFT JOIN sequences t ON s.hash = t.hash
                    WHERE t.hash IS NULL
                """
                ).df()
            else:
                df_to_insert = df_new.copy()

            # Assign IDs
            if len(df_to_insert) > 0:
                start_id = self._sequence_counter
                df_to_insert["sequence_id"] = range(
                    start_id, start_id + len(df_to_insert)
                )
                df_to_insert["created_at"] = datetime.now()

                # Batch insert (vectorized!) - specify column order explicitly
                _legacy_insert_tmp = f"_seq_insert_{next(_TEMP_TABLE_COUNTER)}"
                self.conn.register(_legacy_insert_tmp, df_to_insert)
                # wrap INSERT + SELECT-back in an explicit transaction
                try:
                    self.conn.begin()
                    self.conn.execute(
                        f"""
                        INSERT INTO sequences (sequence_id, sequence, length, hash, created_at)
                        SELECT sequence_id, sequence, length, hash, created_at FROM {_legacy_insert_tmp}
                    """
                    )
                    self.conn.commit()
                    # resync counter from DB after commit to handle any gaps.
                    # Counter represents "next available ID", so set to MAX + 1.
                    max_id_row = self.conn.execute(
                        "SELECT MAX(sequence_id) FROM sequences"
                    ).fetchone()
                    if max_id_row and max_id_row[0] is not None:
                        self._sequence_counter = max_id_row[0] + 1
                except Exception:
                    self.conn.rollback()
                    raise
                finally:
                    self.conn.unregister(_legacy_insert_tmp)

            # Get all IDs (including existing)
            df_result = self.conn.execute(
                """
                SELECT hash, sequence_id
                FROM sequences
                WHERE hash IN (SELECT hash FROM _seq_batch)
            """
            ).df()
        finally:
            self.conn.unregister("_seq_batch")

        # Map back to original order
        hash_to_id = dict(zip(df_result["hash"], df_result["sequence_id"]))
        return [hash_to_id[self._hash_sequence(seq)] for seq in sequences]

    def add_sequence(self, sequence: str) -> int:
        """Add single sequence (convenience wrapper)."""
        return self.add_sequences_batch([sequence])[0]

    def add_predictions_batch(self, data: list[dict[str, Any]]):
        """
        Batch add predictions efficiently.

        Args:
            data: List of dicts with keys: sequence_id, prediction_type,
                  model_name, value, metadata (optional)

        Example:
            >>> ds.add_predictions_batch([
            ...     {'sequence_id': 1, 'prediction_type': 'tm',
            ...      'model_name': 'temberture-regression', 'value': 65.5},
            ...     {'sequence_id': 2, 'prediction_type': 'tm',
            ...      'model_name': 'temberture-regression', 'value': 70.2},
            ... ])
        """
        if not data:
            return

        df = pd.DataFrame(data)

        # Assign IDs
        start_id = self._prediction_counter
        df["prediction_id"] = range(start_id, start_id + len(df))
        df["created_at"] = datetime.now()

        # Ensure metadata is JSON string — B9 fix: apply json.dumps to any non-str
        # value (lists, numpy arrays, tuples) not just dicts.
        if "metadata" in df.columns:
            def _serialize_meta(x):
                if x is None or isinstance(x, str):
                    return x
                try:
                    return json.dumps(x)
                except (TypeError, ValueError) as _e:
                    raise ValueError(
                        f"metadata value is not JSON-serializable: {type(x)} — {_e}"
                    ) from _e
            df["metadata"] = df["metadata"].apply(_serialize_meta)
        else:
            df["metadata"] = None

        # Use a monotonically-increasing counter for the temp table name so that
        # if Python GC reuses a DataFrame's memory address (id(df)) between coroutine
        # suspensions, we still get a unique name.
        _tmp = f"_pred_batch_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_tmp, df)
        try:
            with self._transaction():
                self.conn.execute(
                    f"""
                    INSERT INTO predictions (prediction_id, sequence_id, model_name, prediction_type, value, metadata, created_at)
                    SELECT prediction_id, sequence_id, model_name, prediction_type, value, metadata, created_at FROM {_tmp}
                    ON CONFLICT (sequence_id, prediction_type, model_name) DO UPDATE SET
                        value = CASE WHEN excluded.value IS NOT NULL THEN excluded.value ELSE predictions.value END,
                        metadata = CASE WHEN excluded.value IS NOT NULL THEN excluded.metadata ELSE predictions.metadata END,
                        prediction_id = predictions.prediction_id
                """
                )
                # We pre-assigned IDs up to start_id + len(df) - 1; setting
                # counter to start_id + len(df) avoids ID collisions even
                # when ON CONFLICT skipped some rows.
                self._prediction_counter = start_id + len(df)
        finally:
            self.conn.unregister(_tmp)

    def add_prediction(
        self,
        sequence_id: int,
        prediction_type: str,
        model_name: str,
        value: Optional[float],
        metadata: Optional[dict] = None,
    ):
        """Add single prediction (convenience wrapper)."""
        self.add_predictions_batch(
            [
                {
                    "sequence_id": sequence_id,
                    "prediction_type": prediction_type,
                    "model_name": model_name,
                    "value": value,
                    "metadata": metadata,
                }
            ]
        )

    def has_prediction(
        self, sequence: str, prediction_type: str, model_name: str
    ) -> bool:
        """Check if prediction exists for sequence."""
        seq_hash = self._hash_sequence(sequence)
        result = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM predictions p
            JOIN sequences s ON p.sequence_id = s.sequence_id
            WHERE s.hash = ?
            AND p.prediction_type = ?
            AND p.model_name = ?
        """,
            [seq_hash, prediction_type, model_name],
        ).fetchone()

        return result is not None and result[0] > 0

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """
        Execute arbitrary SQL query and return DataFrame.

        This is the POWER feature - query directly without loading everything!

        Args:
            sql: DuckDB SQL query
            params: Optional query parameters

        Returns:
            DataFrame with results (only loads what matches query!)

        Example:
            >>> # Find high-quality long sequences
            >>> df = ds.query('''
            ...     SELECT s.sequence, p.value as plddt
            ...     FROM sequences s
            ...     JOIN predictions p ON s.sequence_id = p.sequence_id
            ...     WHERE s.length > 200
            ...     AND p.prediction_type = 'plddt'
            ...     AND p.value > 80
            ... ''')
        """
        if params:
            return self.conn.execute(sql, params).df()
        return self.conn.execute(sql).df()

    def get_predictions_by_sequence(
        self,
        sequence: str,
        prediction_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get predictions for a sequence."""
        seq_hash = self._hash_sequence(sequence)

        sql = """
            SELECT p.*
            FROM predictions p
            JOIN sequences s ON p.sequence_id = s.sequence_id
            WHERE s.hash = ?
        """
        params = [seq_hash]

        if prediction_type:
            sql += " AND p.prediction_type = ?"
            params.append(prediction_type)

        if model_name:
            sql += " AND p.model_name = ?"
            params.append(model_name)

        return self.conn.execute(sql, params).df()

    def get_sequence_id(self, sequence: str) -> Optional[int]:
        """Get sequence_id for a sequence."""
        seq_hash = self._hash_sequence(sequence)
        result = self.conn.execute(
            "SELECT sequence_id FROM sequences WHERE hash = ?", [seq_hash]
        ).fetchone()
        return result[0] if result else None

    def add_embedding(
        self,
        sequence_id: int,
        model_name: str,
        embedding: np.ndarray,
        layer: Optional[int] = None,
    ):
        """
        Add embedding stored inline in DuckDB as FLOAT[] (no per-file Parquet overhead).

        Args:
            sequence_id: Sequence ID
            model_name: Model name
            embedding: Numpy array
            layer: Optional layer number
        """
        embedding_id = self._embedding_counter
        self._embedding_counter += 1

        # ON CONFLICT DO UPDATE to handle re-insertion of the same embedding
        self.conn.execute(
            """
            INSERT INTO embeddings
            (embedding_id, sequence_id, model_name, layer, values, dimension, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (sequence_id, model_name, layer) DO UPDATE SET
                values = excluded.values,
                dimension = excluded.dimension
        """,
            [
                embedding_id,
                sequence_id,
                model_name,
                layer,
                embedding.tolist(),
                len(embedding),
                datetime.now(),
            ],
        )
        return embedding_id

    def add_embeddings_batch(self, data: list[dict]):
        """Batch-insert embeddings in a single DuckDB statement.

        Each dict must have: sequence_id (int), model_name (str),
        embedding (np.ndarray). Optional: layer (int or None).

        Significantly faster than N individual add_embedding() calls —
        one INSERT…SELECT vs. N individual row inserts.
        """
        if not data:
            return
        start_id = self._embedding_counter
        rows = []
        for i, item in enumerate(data):
            emb = item["embedding"]
            if not isinstance(emb, np.ndarray):
                emb = np.asarray(emb, dtype=float)
            rows.append(
                {
                    "embedding_id": start_id + i,
                    "sequence_id": int(item["sequence_id"]),
                    "model_name": item["model_name"],
                    "layer": item.get("layer"),
                    "values": emb.tolist(),
                    "dimension": len(emb),
                    "created_at": datetime.now(),
                }
            )
        df = pd.DataFrame(rows)
        _tmp = f"_emb_batch_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_tmp, df)
        try:
            with self._transaction():
                self.conn.execute(
                    f"""
                    INSERT INTO embeddings
                        (embedding_id, sequence_id, model_name, layer, values, dimension, created_at)
                    SELECT embedding_id, sequence_id, model_name, layer, values, dimension, created_at
                    FROM {_tmp}
                    ON CONFLICT (sequence_id, model_name, layer) DO UPDATE SET
                        values = excluded.values,
                        dimension = excluded.dimension
                """
                )
                self._embedding_counter = start_id + len(data)
        finally:
            self.conn.unregister(_tmp)

    def get_embeddings_by_sequence(
        self, sequence: str, model_name: Optional[str] = None, load_data: bool = False
    ) -> list[dict]:
        """Get embeddings for a sequence."""
        seq_hash = self._hash_sequence(sequence)

        sql = """
            SELECT e.*
            FROM embeddings e
            JOIN sequences s ON e.sequence_id = s.sequence_id
            WHERE s.hash = ?
        """
        params = [seq_hash]

        if model_name:
            sql += " AND e.model_name = ?"
            params.append(model_name)

        df = self.conn.execute(sql, params).df()
        results = df.to_dict("records")

        if load_data:
            for r in results:
                vals = r.get("values")
                if vals is not None:
                    # Inline storage: values is a Python list from DuckDB FLOAT[]
                    r["embedding"] = np.array(vals, dtype=np.float32)
                elif r.get("embedding_path"):
                    # Backward compat: old-style per-file Parquet
                    df_emb = pd.read_parquet(r["embedding_path"])
                    r["embedding"] = np.array(df_emb["values"].iloc[0])

        return results

    def create_pipeline_run(
        self, run_id: str, pipeline_type: str, config: dict, status: str = "running"
    ):
        """Create or update a pipeline run record (safe for resume runs)."""
        now = datetime.now()
        self.conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, pipeline_type, config, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at
        """,
            [
                run_id,
                pipeline_type,
                json.dumps(config),
                status,
                now,
                now,
            ],
        )

    def update_pipeline_run_status(self, run_id: str, status: str):
        """Update pipeline run status."""
        self.conn.execute(
            """
            UPDATE pipeline_runs
            SET status = ?, updated_at = ?
            WHERE run_id = ?
        """,
            [status, datetime.now(), run_id],
        )

    def mark_stage_complete(
        self,
        run_id: str,
        stage_name: str,
        stage_id: str,
        input_count: int,
        output_count: int,
        status: str = "completed",
    ):
        """Mark stage as complete (or failed/skipped).

        uses INSERT ... ON CONFLICT DO NOTHING so that a stage that
        was already marked 'completed' does not get its completed_at timestamp
        overwritten on resume.  Only truly new rows are inserted.
        """
        self.conn.execute(
            """
            INSERT INTO stage_completions
            (stage_id, run_id, stage_name, status, input_count, output_count, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (stage_id) DO NOTHING
        """,
            [
                stage_id,
                run_id,
                stage_name,
                status,
                input_count,
                output_count,
                datetime.now(),
            ],
        )

    def is_stage_complete(self, stage_id: str) -> bool:
        """Check if stage is complete."""
        result = self.conn.execute(
            """
            SELECT COUNT(*) FROM stage_completions
            WHERE stage_id = ? AND status = 'completed'
        """,
            [stage_id],
        ).fetchone()
        return result is not None and result[0] > 0

    _ALLOWED_TABLES = frozenset(
        {
            "sequences",
            "predictions",
            "embeddings",
            "structures",
            "pipeline_runs",
            "stage_completions",
        }
    )

    def export_to_parquet(self, table_name: str, output_path: Union[str, Path]):
        """
        Export table to Parquet file (for sharing/archiving).

        Args:
            table_name: Table to export (sequences, predictions, etc.)
            output_path: Path to output Parquet file

        Example:
            >>> ds.export_to_parquet('sequences', 'sequences_backup.parquet')
        """
        if table_name not in self._ALLOWED_TABLES:
            raise ValueError(
                f"table_name must be one of {set(self._ALLOWED_TABLES)}, got '{table_name}'"
            )
        self.conn.execute(
            f"COPY (SELECT * FROM {table_name}) TO ? (FORMAT PARQUET, COMPRESSION SNAPPY)",
            [str(output_path)],
        )

    def add_generation_metadata(
        self,
        sequence_id: int,
        model_name: str,
        run_id: str = "",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        max_length: Optional[int] = None,
        sampling_params: Optional[
            dict
        ] = None,  # extra params stored but not in dedicated columns
    ) -> int:
        """Store generation parameters for a sequence.

        Returns:
            metadata_id of the inserted row.
        """
        # Merge sampling_params into known fields where they overlap
        if sampling_params:
            top_k = top_k if top_k is not None else sampling_params.get("top_k")
            top_p = top_p if top_p is not None else sampling_params.get("top_p")

        # GEN-01-DS: serialize sampling_params as JSON for storage
        sampling_params_json = json.dumps(sampling_params) if sampling_params else None

        metadata_id = self._generation_metadata_counter
        # ON CONFLICT (sequence_id, run_id) DO NOTHING to prevent duplicate rows
        self.conn.execute(
            """
            INSERT INTO generation_metadata
            (metadata_id, sequence_id, run_id, model_name, temperature, top_k, top_p,
             num_return_sequences, do_sample, repetition_penalty, max_length,
             sampling_params, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (sequence_id, run_id) DO NOTHING
        """,
            [
                metadata_id,
                sequence_id,
                run_id,
                model_name,
                temperature,
                top_k,
                top_p,
                num_return_sequences,
                do_sample,
                repetition_penalty,
                max_length,
                sampling_params_json,
                datetime.now(),
            ],
        )
        self._generation_metadata_counter += 1
        # B8 fix: query back actual stored row ID — ON CONFLICT DO NOTHING means the
        # pre-assigned counter ID may not match what's in the DB (conflict keeps old row).
        actual = self.conn.execute(
            "SELECT metadata_id FROM generation_metadata WHERE sequence_id = ? AND run_id = ?",
            [sequence_id, run_id],
        ).fetchone()
        if actual is None:
            raise RuntimeError(
                f"Failed to retrieve metadata_id after insert for "
                f"sequence_id={sequence_id}, run_id={run_id}"
            )
        return actual[0]

    def add_generation_metadata_batch(self, rows: list[dict]) -> None:
        """Batch-insert generation metadata — one DuckDB round-trip.

        Each dict in *rows* must have at minimum ``sequence_id`` and ``model_name``.
        Optional fields: ``temperature``, ``top_k``, ``top_p``,
        ``num_return_sequences``, ``do_sample``, ``repetition_penalty``,
        ``max_length``.
        """
        if not rows:
            return
        now = datetime.now()
        records = []
        for row in rows:
            metadata_id = self._generation_metadata_counter
            self._generation_metadata_counter += 1
            sp = row.get("sampling_params") or {}
            # GEN-04: serialize sampling_params as JSON string for storage
            sampling_params_json = json.dumps(sp) if sp else None
            # Serialize DMS provenance (or any extra metadata) as a JSON string
            extra_meta = row.get("metadata")
            metadata_json = json.dumps(extra_meta) if extra_meta else None
            records.append({
                "metadata_id": metadata_id,
                "sequence_id": int(row["sequence_id"]),
                "run_id": row.get("run_id", ""),
                "model_name": row["model_name"],
                "temperature": row.get("temperature"),
                "top_k": row.get("top_k") if row.get("top_k") is not None else sp.get("top_k"),
                "top_p": row.get("top_p") if row.get("top_p") is not None else sp.get("top_p"),
                "num_return_sequences": row.get("num_return_sequences", sp.get("num_return_sequences")),
                "do_sample": row.get("do_sample", sp.get("do_sample")),
                "repetition_penalty": row.get("repetition_penalty", sp.get("repetition_penalty")),
                "max_length": row.get("max_length", sp.get("max_length")),
                "sampling_params": sampling_params_json,
                "label": row.get("label"),
                "metadata": metadata_json,
                "created_at": now,
            })
        df = pd.DataFrame(records)
        _tmp = f"_gen_meta_batch_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_tmp, df)
        try:
            self.conn.execute(
                f"""
                INSERT INTO generation_metadata
                (metadata_id, sequence_id, run_id, model_name, temperature, top_k, top_p,
                 num_return_sequences, do_sample, repetition_penalty, max_length, sampling_params,
                 label, metadata, created_at)
                SELECT metadata_id, sequence_id, run_id, model_name, temperature, top_k, top_p,
                       num_return_sequences, do_sample, repetition_penalty, max_length, sampling_params,
                       label, metadata, created_at
                FROM {_tmp}
                ON CONFLICT DO NOTHING
                """
            )
        finally:
            self.conn.unregister(_tmp)

    def export_to_dataframe(
        self,
        include_sequences: bool = True,
        include_predictions: bool = True,
        include_generation_metadata: bool = False,
        prediction_types: Optional[list[str]] = None,
        run_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Export data to a flat DataFrame using a single DuckDB SQL query.

        Uses conditional aggregation (CASE WHEN pivot) — no per-type queries,
        no pandas merges, no full table loads.

        Args:
            include_sequences: Always True; includes sequence_id, sequence, length.
            include_predictions: Pivot prediction_type values into columns.
            include_generation_metadata: Join generation_metadata columns.
            prediction_types: Limit to specific prediction types (None = all).

        Returns:
            Wide-format DataFrame: one row per sequence, one column per prediction type.
        """
        # Refresh _extra_columns from DB (guards against cross-process staleness)
        self._refresh_extra_columns()

        # Determine which prediction types to pivot
        if include_predictions:
            if prediction_types is None:
                rows = self.conn.execute(
                    "SELECT DISTINCT prediction_type FROM predictions WHERE prediction_type IS NOT NULL"
                ).fetchall()
                prediction_types = [r[0] for r in rows]
            else:
                prediction_types = [pt for pt in prediction_types if pt]

        # Build CASE WHEN pivot for predictions — use ranked CTE to resolve
        # multi-model conflicts: when two models write the same prediction_type,
        # use the most recent row (highest prediction_id) (DS-03).
        pred_cols_sql = ""
        pred_cte_sql = ""
        if include_predictions and prediction_types:
            # validate prediction_type values; skip any that fail
            import logging as _logging
            _valid_pts = []
            for pt in prediction_types:
                try:
                    _validate_column_name(pt)
                    _valid_pts.append(pt)
                except ValueError:
                    _logging.getLogger(__name__).warning(
                        "export_to_dataframe: skipping invalid prediction_type %r", pt
                    )
            prediction_types = _valid_pts
            cases = [
                "MAX(CASE WHEN p.prediction_type = '{}' THEN p.value END) AS \"{}\"".format(
                    pt.replace("'", "''"), pt.replace('"', '""')
                )
                for pt in prediction_types
            ]
            pred_cols_sql = ",\n            " + ",\n            ".join(cases)
            pred_cte_sql = """
            _ranked_preds AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY sequence_id, prediction_type
                    ORDER BY prediction_id DESC
                ) AS _rn
                FROM predictions
                WHERE prediction_type IS NOT NULL
            )"""

        # Build generation metadata join.
        # Use MAX() aggregation so multi-run sequences produce one row each.
        # GEN-10: scope to run_id when provided to avoid pulling metadata from prior runs.
        gen_join_sql = ""
        gen_cols_sql = ""
        gen_join_params: list = []
        if include_generation_metadata:
            if run_id:
                gen_join_sql = (
                    "LEFT JOIN generation_metadata gm "
                    "ON s.sequence_id = gm.sequence_id AND gm.run_id = ?"
                )
                gen_join_params = [run_id]
            else:
                gen_join_sql = (
                    "LEFT JOIN generation_metadata gm ON s.sequence_id = gm.sequence_id"
                )
            gen_cols_sql = """,
            MAX(gm.model_name) AS gen_model_name,
            MAX(gm.temperature) AS gen_temperature,
            MAX(gm.top_k) AS gen_top_k,
            MAX(gm.top_p) AS gen_top_p,
            MAX(gm.sampling_params) AS gen_sampling_params"""

        pred_join_sql = (
            "LEFT JOIN _ranked_preds p ON s.sequence_id = p.sequence_id AND p._rn = 1"
            if include_predictions and prediction_types
            else ""
        )
        # group_extra intentionally empty: generation metadata columns now use MAX()
        # aggregation and do not require GROUP BY, preserving one-row-per-sequence
        # guarantee even when n_runs > 1.
        group_extra = ""

        # Discover all columns on sequences table for SELECT — use cached set
        # PIVOT-02: validate extra column names; skip invalid ones with a warning
        import logging as _logging
        base_seq_cols = ["sequence_id", "sequence", "length", "hash"]
        _valid_extra_etdf = []
        for _ec in sorted(self._extra_columns):
            try:
                _validate_column_name(_ec)
                _valid_extra_etdf.append(_ec)
            except ValueError:
                _logging.getLogger(__name__).warning(
                    "export_to_dataframe: skipping invalid extra column %r", _ec
                )
        extra_seq_cols = _valid_extra_etdf
        seq_col_names = base_seq_cols + extra_seq_cols
        seq_select = ", ".join(f's."{c}"' for c in seq_col_names)
        seq_group = ", ".join(f's."{c}"' for c in seq_col_names)

        cte_clause = f"WITH{pred_cte_sql}" if pred_cte_sql else ""
        query = f"""
            {cte_clause}
            SELECT
                {seq_select}{pred_cols_sql}{gen_cols_sql}
            FROM sequences s
            {pred_join_sql}
            {gen_join_sql}
            GROUP BY {seq_group}{group_extra}
        """
        return self.conn.execute(query, gen_join_params).df()

    def get_embeddings_bulk(
        self,
        sequence_ids: list[int],
        model_name: Optional[str] = None,
    ) -> dict[int, np.ndarray]:
        """
        Fetch embeddings for multiple sequences in a single JOIN query.

        Replaces N individual get_embeddings_by_sequence() calls (O(n) queries → O(1)).

        Args:
            sequence_ids: List of sequence IDs to fetch.
            model_name: Optional model filter.

        Returns:
            Dict mapping sequence_id → numpy embedding array for sequences that have an embedding.
        """
        if not sequence_ids:
            return {}
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        # unique temp name to avoid concurrent coroutine collision
        _emb_tmp = f"_emb_bulk_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_emb_tmp, ids_df)
        try:
            sql = f"""
                SELECT e.sequence_id, e.values
                FROM embeddings e
                INNER JOIN {_emb_tmp} b ON e.sequence_id = b.sequence_id
            """
            params: list[Any] = []
            if model_name:
                sql += " AND e.model_name = ?"
                params.append(model_name)
            df = self.conn.execute(sql, params).df()
        finally:
            self.conn.unregister(_emb_tmp)

        # use itertuples() for faster iteration than iterrows()
        result: dict[int, np.ndarray] = {}
        for r in df.itertuples(index=False):
            vals = r.values
            if vals is not None and not (isinstance(vals, float) and np.isnan(vals)):
                result[int(r.sequence_id)] = np.array(vals, dtype=np.float32)
        return result

    def get_embeddings_concat(
        self,
        sequence_ids: list[int],
        model_names: list[str],
    ) -> dict[int, np.ndarray]:
        """Fetch and concatenate embeddings from multiple models per sequence.

        For each sequence_id, retrieves the embedding from each model in
        ``model_names`` order and horizontally concatenates them into a single
        vector.  Sequences missing an embedding from *any* requested model are
        omitted from the result.

        Args:
            sequence_ids: Sequence IDs to fetch.
            model_names: Ordered list of model names whose embeddings will be
                concatenated (e.g. ``["esm2-8m", "esmc-300m"]``).

        Returns:
            Dict mapping ``sequence_id`` → concatenated numpy array.
        """
        if not sequence_ids or not model_names:
            return {}

        per_model: list[dict[int, np.ndarray]] = []
        for mn in model_names:
            per_model.append(self.get_embeddings_bulk(sequence_ids, model_name=mn))

        # Only keep IDs present in ALL models
        common_ids = set(sequence_ids)
        for emb_map in per_model:
            common_ids &= set(emb_map.keys())

        result: dict[int, np.ndarray] = {}
        for sid in common_ids:
            result[sid] = np.concatenate([pm[sid] for pm in per_model])
        return result

    def get_uncached_sequence_ids(
        self,
        sequence_ids: list[int],
        prediction_type: str,
        model_name: str,
    ) -> list[int]:
        """
        Return sequence_ids that do NOT yet have a given prediction (vectorized anti-join).

        Replaces N individual has_prediction() calls with a single SQL query.

        Args:
            sequence_ids: Candidate sequence IDs to check.
            prediction_type: Prediction type key.
            model_name: Model name.

        Returns:
            List of sequence_ids with no cached prediction.
        """
        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        # unique temp name to avoid concurrent coroutine collision
        _check_tmp = f"_check_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_check_tmp, ids_df)
        try:
            # A row is uncached if either (a) no prediction row exists at all,
            # or (b) one exists with NULL value AND metadata.status != 'failed'
            # (i.e. an interrupted/partial write — retry it).  Rows explicitly
            # marked failed by skip_on_error are kept as cached so we don't
            # endlessly retry sequences the API can't handle.
            result = self.conn.execute(
                f"""
                SELECT DISTINCT c.sequence_id
                FROM {_check_tmp} c
                LEFT JOIN predictions p
                    ON c.sequence_id = p.sequence_id
                    AND p.prediction_type = ?
                    AND p.model_name = ?
                WHERE p.prediction_id IS NULL
                   OR (p.value IS NULL
                       AND COALESCE(json_extract_string(p.metadata, '$.status'), '') != 'failed')
            """,
                [prediction_type, model_name],
            ).fetchall()
        finally:
            self.conn.unregister(_check_tmp)
        return [r[0] for r in result]

    def get_predictions_bulk(
        self,
        sequence_ids: list[int],
        prediction_type: str,
        model_name: str,
    ) -> pd.DataFrame:
        """
        Fetch predictions for multiple sequences in a single JOIN query.

        Replaces N individual get_predictions_by_sequence() calls.

        Returns:
            DataFrame with columns: sequence_id, value, metadata
        """
        if not sequence_ids:
            return pd.DataFrame(columns=["sequence_id", "value", "metadata"])
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        # unique temp name to avoid concurrent coroutine collision
        _bulk_tmp = f"_bulk_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_bulk_tmp, ids_df)
        try:
            result = self.conn.execute(
                f"""
                SELECT p.sequence_id, p.value, p.metadata
                FROM predictions p
                INNER JOIN {_bulk_tmp} b ON p.sequence_id = b.sequence_id
                WHERE p.prediction_type = ? AND p.model_name = ?
            """,
                [prediction_type, model_name],
            ).df()
        finally:
            self.conn.unregister(_bulk_tmp)
        return result

    def count_matching_sequences(self, sequences: list[str]) -> int:
        """
        Count how many of the given sequences already exist in the datastore.

        Uses a single vectorized hash join instead of N individual lookups.
        Safe to call from any context — registers DataFrame explicitly.
        """
        if not sequences:
            return 0
        hashes_df = pd.DataFrame({"hash": [self._hash_sequence(s) for s in sequences]})
        # unique temp name to avoid concurrent coroutine collision
        _count_tmp = f"_count_hashes_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_count_tmp, hashes_df)
        try:
            result = self.conn.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM sequences s
                INNER JOIN {_count_tmp} h ON s.hash = h.hash
            """
            ).fetchone()
        finally:
            self.conn.unregister(_count_tmp)
        return result[0] if result else 0

    def set_pipeline_metadata(self, key: str, value: Any):
        """Upsert a key/value pair in the pipeline_metadata table."""
        self.conn.execute(
            """
            INSERT INTO pipeline_metadata (key, value, created_at) VALUES (?, ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value
        """,
            [
                key,
                json.dumps(value) if not isinstance(value, str) else value,
                datetime.now(),
            ],
        )

    def get_pipeline_metadata(self, key: str) -> Optional[Any]:
        """Retrieve a value from pipeline_metadata by key."""
        result = self.conn.execute(
            "SELECT value FROM pipeline_metadata WHERE key = ?", [key]
        ).fetchone()
        if result is None:
            return None
        try:
            return json.loads(result[0])
        except (json.JSONDecodeError, TypeError):
            return result[0]

    # ------------------------------------------------------------------
    # Structure methods
    # ------------------------------------------------------------------

    def add_structure(
        self,
        sequence_id: int,
        model_name: str,
        structure_str: Optional[str] = None,
        format: str = "pdb",
        plddt_mean: Optional[float] = None,
        plddt: Optional[float] = None,  # alias for plddt_mean
    ) -> int:
        """Store a structure gzip-compressed as BLOB (~8-12x smaller than plain TEXT).

        Args:
            sequence_id: Sequence ID.
            model_name: Model that produced the structure (e.g. 'esmfold').
            structure_str: Full structure file content as a string.
            format: 'pdb' or 'cif' (default 'pdb').
            plddt_mean: Mean pLDDT score (optional).
            plddt: Alias for plddt_mean.

        Returns:
            structure_id of the inserted row.
        """
        actual_plddt = plddt_mean if plddt_mean is not None else plddt
        structure_id = self._structure_counter
        if structure_str is not None:
            compressed = gzip.compress(structure_str.encode("utf-8"), compresslevel=6)
        else:
            compressed = None
        self.conn.execute(
            """
            INSERT INTO structures
            (structure_id, sequence_id, model_name, format, structure_data, plddt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                structure_id,
                sequence_id,
                model_name,
                format,
                compressed,
                actual_plddt,
                datetime.now(),
            ],
        )
        self._structure_counter += 1
        return structure_id

    def get_structure(
        self,
        sequence_id: int,
        model_name: Optional[str] = None,
    ) -> Optional[dict]:
        """Fetch the most recent structure for a sequence, decompressing on read.

        Returns a dict with 'structure_str' key (always decompressed string) regardless
        of whether data was stored compressed (structure_data BLOB) or as legacy plain TEXT.

        Args:
            sequence_id: Sequence ID.
            model_name: Optional model filter.

        Returns:
            Dict with structure record, or None if not found.
        """
        sql = """
            SELECT structure_id, sequence_id, model_name, format,
                   structure_data, structure_str, structure_path, plddt, created_at
            FROM structures
            WHERE sequence_id = ?
        """
        params = [sequence_id]
        if model_name:
            sql += " AND model_name = ?"
            params.append(model_name)
        sql += " ORDER BY created_at DESC LIMIT 1"
        df = self.conn.execute(sql, params).df()
        if df.empty:
            return None
        record = df.iloc[0].to_dict()
        # Decompress structure_data (new path) or fall back to structure_str (old data)
        data_blob = record.pop("structure_data", None)
        if data_blob is not None:
            try:
                blob_bytes = (
                    bytes(data_blob) if not isinstance(data_blob, bytes) else data_blob
                )
                record["structure_str"] = gzip.decompress(blob_bytes).decode("utf-8")
            except (OSError, gzip.BadGzipFile):
                pass  # Corrupt or non-gzip blob — fall back to structure_str
        return record

    def get_structures_for_ids(
        self,
        sequence_ids: list[int],
        model_name: str,
    ) -> dict[int, dict]:
        """Fetch the most recent structure for each sequence_id in one query.

        Returns {sequence_id: record} where record has a 'structure_str' key.
        Replaces N individual get_structure() calls for batch structure injection.
        """
        if not sequence_ids:
            return {}
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        _tmp = f"_struct_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_tmp, ids_df)
        try:
            df = self.conn.execute(
                f"""
                WITH ranked AS (
                    SELECT
                        sequence_id, structure_data, structure_str, format, plddt,
                        ROW_NUMBER() OVER (
                            PARTITION BY sequence_id ORDER BY created_at DESC
                        ) AS _rn
                    FROM structures
                    WHERE model_name = ?
                      AND sequence_id IN (SELECT sequence_id FROM {_tmp})
                )
                SELECT sequence_id, structure_data, structure_str, format, plddt
                FROM ranked
                WHERE _rn = 1
            """,
                [model_name],
            ).df()
        finally:
            self.conn.unregister(_tmp)

        result: dict[int, dict] = {}
        for record in df.to_dict("records"):
            seq_id = int(record["sequence_id"])
            data_blob = record.pop("structure_data", None)
            if data_blob is not None:
                try:
                    blob_bytes = (
                        bytes(data_blob)
                        if not isinstance(data_blob, bytes)
                        else data_blob
                    )
                    record["structure_str"] = gzip.decompress(blob_bytes).decode("utf-8")
                except (OSError, gzip.BadGzipFile):
                    pass
            result[seq_id] = record
        return result

    def get_structures_bulk(self, sequence_ids: list[int]) -> pd.DataFrame:
        """Fetch structures for multiple sequences, decompressing structure content.

        Returns a DataFrame with a 'structure_str' column (always plain text) regardless
        of whether data was stored compressed (structure_data BLOB) or as legacy plain TEXT.

        Args:
            sequence_ids: List of sequence IDs to look up.

        Returns:
            DataFrame with one row per structure record.
        """
        if not sequence_ids:
            return pd.DataFrame(
                columns=[
                    "structure_id",
                    "sequence_id",
                    "model_name",
                    "format",
                    "structure_str",
                    "plddt",
                ]
            )
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        # unique temp name to avoid concurrent coroutine collision
        _struct_tmp = f"_struct_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_struct_tmp, ids_df)
        try:
            result = self.conn.execute(
                f"""
                SELECT s.structure_id, s.sequence_id, s.model_name,
                       s.format, s.structure_data, s.structure_str, s.plddt
                FROM structures s
                INNER JOIN {_struct_tmp} b ON s.sequence_id = b.sequence_id
            """
            ).df()
        finally:
            self.conn.unregister(_struct_tmp)

        def _decompress_row(row):
            blob = row.get("structure_data")
            if blob is not None:
                try:
                    blob_bytes = bytes(blob) if not isinstance(blob, bytes) else blob
                    return gzip.decompress(blob_bytes).decode("utf-8")
                except (OSError, gzip.BadGzipFile):
                    pass  # Corrupt or non-gzip blob — fall back to structure_str
            return row.get("structure_str")

        result["structure_str"] = result.apply(_decompress_row, axis=1)
        result = result.drop(columns=["structure_data"], errors="ignore")
        return result

    # ------------------------------------------------------------------
    # Filter results methods (for resume support)
    # ------------------------------------------------------------------

    def save_filter_results(
        self,
        run_id: str,
        stage_name: str,
        passed_sequence_ids: list[int],
    ):
        """Record which sequence_ids passed a filter stage (for resume support).

        Args:
            run_id: Pipeline run ID.
            stage_name: Filter stage name.
            passed_sequence_ids: IDs of sequences that passed the filter.
        """
        if not passed_sequence_ids:
            return
        rows = []
        for seq_id in passed_sequence_ids:
            rows.append(
                {
                    "filter_id": self._filter_id_counter,
                    "run_id": run_id,
                    "stage_name": stage_name,
                    "sequence_id": seq_id,
                    "passed": True,
                    "created_at": datetime.now(),
                }
            )
            self._filter_id_counter += 1

        df = pd.DataFrame(rows)
        _filter_tmp = f"_filter_batch_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_filter_tmp, df)
        try:
            self.conn.execute(
                f"""
                INSERT INTO filter_results
                (filter_id, run_id, stage_name, sequence_id, passed, created_at)
                SELECT filter_id, run_id, stage_name, sequence_id, passed, created_at
                FROM {_filter_tmp}
                ON CONFLICT DO NOTHING
            """
            )
        finally:
            self.conn.unregister(_filter_tmp)

    def get_filter_results(
        self,
        run_id: str,
        stage_name: str,
    ) -> list[int]:
        """Return sequence_ids that passed a given filter stage in a run.

        Args:
            run_id: Pipeline run ID.
            stage_name: Filter stage name.

        Returns:
            List of sequence_ids that passed the filter, or empty list if no data.
        """
        rows = self.conn.execute(
            """
            SELECT sequence_id
            FROM filter_results
            WHERE run_id = ? AND stage_name = ? AND passed = TRUE
        """,
            [run_id, stage_name],
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Convenience / compatibility methods
    # ------------------------------------------------------------------

    def get_sequence(self, sequence_id: int) -> Optional[str]:
        """Return the sequence string for a given sequence_id, or None if not found."""
        result = self.conn.execute(
            "SELECT sequence FROM sequences WHERE sequence_id = ?", [sequence_id]
        ).fetchone()
        return result[0] if result else None

    def get_all_sequences(self) -> pd.DataFrame:
        """Return all sequences as a DataFrame with sequence_id, sequence, length columns."""
        return self.conn.execute(
            "SELECT sequence_id, sequence, length FROM sequences ORDER BY sequence_id"
        ).df()

    def add_prediction_by_sequence(
        self,
        sequence: str,
        prediction_type: str,
        model_name: str,
        value: Optional[float],
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a prediction, creating the sequence if it doesn't exist.

        Returns:
            prediction_id of the inserted row.
        """
        seq_id = self.add_sequence(sequence)
        self.add_predictions_batch(
            [
                {
                    "sequence_id": seq_id,
                    "prediction_type": prediction_type,
                    "model_name": model_name,
                    "value": value,
                    "metadata": metadata,
                }
            ]
        )
        # COUNTER-01: query back actual stored prediction_id (ON CONFLICT keeps old ID)
        actual_row = self.conn.execute(
            "SELECT prediction_id FROM predictions "
            "WHERE sequence_id = ? AND prediction_type = ? AND model_name = ?",
            [seq_id, prediction_type, model_name],
        ).fetchone()
        return actual_row[0] if actual_row else -1

    def get_predictions(
        self,
        sequence_id: int,
        prediction_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return predictions for a sequence_id as a DataFrame."""
        sql = "SELECT * FROM predictions WHERE sequence_id = ?"
        params: list[Any] = [sequence_id]
        if prediction_type:
            sql += " AND prediction_type = ?"
            params.append(prediction_type)
        if model_name:
            sql += " AND model_name = ?"
            params.append(model_name)
        return self.conn.execute(sql, params).df()

    def get_generation_metadata(self, sequence_id: int) -> list[dict]:
        """Return generation metadata records for a sequence_id."""
        df = self.conn.execute(
            "SELECT * FROM generation_metadata WHERE sequence_id = ?", [sequence_id]
        ).df()
        return df.to_dict("records")

    def get_structures_by_sequence(
        self,
        sequence: str,
        model_name: Optional[str] = None,
    ) -> list[dict]:
        """Return structure records for a sequence string (decompressed)."""
        seq_hash = self._hash_sequence(sequence)
        sql = """
            SELECT st.*
            FROM structures st
            JOIN sequences s ON st.sequence_id = s.sequence_id
            WHERE s.hash = ?
        """
        params: list[Any] = [seq_hash]
        if model_name:
            sql += " AND st.model_name = ?"
            params.append(model_name)
        df = self.conn.execute(sql, params).df()
        records = df.to_dict("records")
        for rec in records:
            blob = rec.pop("structure_data", None)
            if blob is not None:
                try:
                    blob_bytes = bytes(blob) if not isinstance(blob, bytes) else blob
                    rec["structure_str"] = gzip.decompress(blob_bytes).decode("utf-8")
                except (OSError, gzip.BadGzipFile):
                    pass  # Corrupt or non-gzip blob — fall back to structure_str
        return records

    def get_structure_by_id(self, structure_id: int) -> Optional[dict]:
        """Return a structure record by its structure_id (primary key)."""
        df = self.conn.execute(
            "SELECT * FROM structures WHERE structure_id = ?", [structure_id]
        ).df()
        if df.empty:
            return None
        record = df.iloc[0].to_dict()
        blob = record.pop("structure_data", None)
        if blob is not None:
            try:
                blob_bytes = bytes(blob) if not isinstance(blob, bytes) else blob
                record["structure_str"] = gzip.decompress(blob_bytes).decode("utf-8")
            except (OSError, gzip.BadGzipFile):
                pass  # Corrupt or non-gzip blob — fall back to structure_str
        return record

    def get_embedding(self, embedding_id: int) -> Optional[tuple]:
        """Return (metadata_dict, embedding_array) for a given embedding_id, or None."""
        df = self.conn.execute(
            "SELECT * FROM embeddings WHERE embedding_id = ?", [embedding_id]
        ).df()
        if df.empty:
            return None
        record = df.iloc[0].to_dict()
        vals = record.get("values")
        if vals is not None and not (isinstance(vals, float) and np.isnan(vals)):
            emb_array = np.array(vals, dtype=np.float32)
        else:
            emb_path = record.get("embedding_path")
            if emb_path:
                df_emb = pd.read_parquet(emb_path)
                emb_array = np.array(df_emb["values"].iloc[0])
            else:
                emb_array = None
        record.pop("values", None)
        return record, emb_array

    def get_pipeline_run(self, run_id: str) -> Optional[dict]:
        """Return pipeline run record as a dict, or None if not found."""
        df = self.conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", [run_id]
        ).df()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_stats(self) -> dict[str, int]:
        """Return row counts for the main tables."""
        return {
            "sequences": self.conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[
                0
            ],
            "predictions": self.conn.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0],
            "embeddings": self.conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0],
            "structures": self.conn.execute(
                "SELECT COUNT(*) FROM structures"
            ).fetchone()[0],
            "pipeline_runs": self.conn.execute(
                "SELECT COUNT(*) FROM pipeline_runs"
            ).fetchone()[0],
        }

    def export_to_csv(self, path: Union[str, Path], **kwargs) -> None:
        """Export all data to CSV (convenience wrapper around export_to_dataframe)."""
        df = self.export_to_dataframe(**kwargs)
        df.to_csv(str(path), index=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # WorkingSet support methods
    # ------------------------------------------------------------------

    def materialize_working_set(
        self,
        ws: WorkingSet,
        include_predictions: bool = True,
        prediction_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Materialize a WorkingSet into a DataFrame via a single DuckDB pivot query.

        Args:
            ws: WorkingSet containing the sequence IDs to materialize.
            include_predictions: If True, pivot prediction values into columns.
            prediction_types: Limit to specific types (None = all available).

        Returns:
            Wide-format DataFrame: one row per sequence, one column per prediction type.
            Always includes the following columns regardless of pipeline configuration:

            - ``sequence_id``, ``sequence``, ``length``, ``hash`` — core sequence data.
            - ``source_label`` — label set on the generation config (e.g.
              ``DirectGenerationConfig.label``, ``SaturationMutagenesisConfig.label``).
              ``NULL`` / ``None`` when no label was supplied.  This column is **always
              present** even when no labels have been set, so downstream code should
              not branch on ``"source_label" in df.columns`` — it is always there.
        """
        if not ws:
            return pd.DataFrame(columns=["sequence_id", "sequence", "length"])

        # Refresh _extra_columns from DB (guards against cross-process staleness)
        self._refresh_extra_columns()

        ids_df = pd.DataFrame({"sequence_id": list(ws.sequence_ids)})
        # unique temp name to avoid concurrent coroutine collision
        _ws_tmp = f"_ws_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_ws_tmp, ids_df)
        try:
            # Determine prediction types to pivot
            if include_predictions:
                if prediction_types is None:
                    rows = self.conn.execute(
                        f"""
                        SELECT DISTINCT p.prediction_type
                        FROM predictions p
                        INNER JOIN {_ws_tmp} w ON p.sequence_id = w.sequence_id
                        WHERE p.prediction_type IS NOT NULL
                    """
                    ).fetchall()
                    prediction_types = [r[0] for r in rows]

            # Build CASE WHEN pivot — use ranked CTE to resolve multi-model conflicts:
            # when two models write the same prediction_type, use the most recent row
            # (highest prediction_id) instead of picking an arbitrary value (DS-03).
            pred_cols_sql = ""
            pred_join_sql = ""
            pred_cte_sql = ""
            if include_predictions and prediction_types:
                # validate prediction_type values; skip any that fail
                import logging as _logging
                _valid_pts = []
                for pt in prediction_types:
                    try:
                        _validate_column_name(pt)
                        _valid_pts.append(pt)
                    except ValueError:
                        _logging.getLogger(__name__).warning(
                            "materialize_working_set: skipping invalid prediction_type %r", pt
                        )
                prediction_types = _valid_pts
                cases = [
                    "MAX(CASE WHEN p.prediction_type = '{}' THEN p.value END) AS \"{}\"".format(
                        pt.replace("'", "''"), pt.replace('"', '""')
                    )
                    for pt in prediction_types
                ]
                pred_cols_sql = ",\n                " + ",\n                ".join(cases)
                pred_cte_sql = f"""
                _ranked_preds AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY sequence_id, prediction_type
                        ORDER BY prediction_id DESC
                    ) AS _rn
                    FROM predictions
                    WHERE prediction_type IS NOT NULL
                      AND sequence_id IN (SELECT sequence_id FROM {_ws_tmp})
                )"""
                pred_join_sql = (
                    "LEFT JOIN _ranked_preds p ON s.sequence_id = p.sequence_id AND p._rn = 1"
                )

            # Discover all columns on sequences table for SELECT — use cached set
            # PIVOT-02: validate extra column names; skip invalid ones with a warning
            import logging as _logging
            base_seq_cols = ["sequence_id", "sequence", "length", "hash"]
            _valid_extra_ws = []
            for _ec in sorted(self._extra_columns):
                try:
                    _validate_column_name(_ec)
                    _valid_extra_ws.append(_ec)
                except ValueError:
                    _logging.getLogger(__name__).warning(
                        "materialize_working_set: skipping invalid extra column %r", _ec
                    )
            extra_seq_cols = _valid_extra_ws
            seq_col_names = base_seq_cols + extra_seq_cols
            seq_select = ", ".join(f's."{c}"' for c in seq_col_names)
            seq_group = ", ".join(f's."{c}"' for c in seq_col_names)

            # Always include source_label from generation_metadata.
            # Use a ranked subquery (most recent metadata_id per sequence) to
            # avoid fan-out when a sequence has multiple generation_metadata rows
            # (e.g. generated under different run_ids). NULL when no label was set.
            # MAX() is an aggregate because the outer query has GROUP BY.
            # The subquery guarantees at most one row per sequence, so MAX() is
            # equivalent to ANY_VALUE() and returns the single label or NULL.
            label_col_sql = ",\n                MAX(gm_label.label) AS source_label"
            label_join_sql = (
                f"LEFT JOIN ("
                f"    SELECT sequence_id, label"
                f"    FROM ("
                f"        SELECT sequence_id, label,"
                f"               ROW_NUMBER() OVER ("
                f"                   PARTITION BY sequence_id"
                f"                   ORDER BY metadata_id DESC"
                f"               ) AS _rn"
                f"        FROM generation_metadata"
                f"        WHERE sequence_id IN (SELECT sequence_id FROM {_ws_tmp})"
                f"    ) _ranked"
                f"    WHERE _rn = 1"
                f") gm_label ON s.sequence_id = gm_label.sequence_id"
            )

            cte_clause = f"WITH{pred_cte_sql}" if pred_cte_sql else ""
            query = f"""
                {cte_clause}
                SELECT
                    {seq_select}{pred_cols_sql}{label_col_sql}
                FROM sequences s
                INNER JOIN {_ws_tmp} w ON s.sequence_id = w.sequence_id
                {pred_join_sql}
                {label_join_sql}
                GROUP BY {seq_group}
            """
            return self.conn.execute(query).df()
        finally:
            self.conn.unregister(_ws_tmp)

    def get_sequence_ids_with_prediction(
        self,
        sequence_ids: list[int],
        prediction_type: str,
        model_name: str,
    ) -> list[int]:
        """Return sequence_ids that DO have a given prediction (inverse of uncached check).

        Args:
            sequence_ids: Candidate sequence IDs.
            prediction_type: Prediction type key.
            model_name: Model name.

        Returns:
            List of sequence_ids that have a cached prediction.
        """
        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        # unique temp name to avoid concurrent coroutine collision
        _has_pred_tmp = f"_has_pred_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_has_pred_tmp, ids_df)
        try:
            result = self.conn.execute(
                f"""
                SELECT DISTINCT c.sequence_id
                FROM {_has_pred_tmp} c
                INNER JOIN predictions p
                    ON c.sequence_id = p.sequence_id
                    AND p.prediction_type = ?
                    AND p.model_name = ?
                WHERE p.value IS NOT NULL
            """,
                [prediction_type, model_name],
            ).fetchall()
        finally:
            self.conn.unregister(_has_pred_tmp)
        return [r[0] for r in result]

    def get_sequences_for_ids(
        self,
        sequence_ids: list[int],
    ) -> list[tuple[int, str]]:
        """Fetch (sequence_id, sequence) pairs for the given IDs.

        Lightweight fetch for building API request items without materializing
        a full DataFrame.

        Args:
            sequence_ids: List of sequence IDs to look up.

        Returns:
            List of (sequence_id, sequence_string) tuples.
        """
        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        # unique temp name to avoid concurrent coroutine collision
        _seq_for_ids_tmp = f"_seq_for_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_seq_for_ids_tmp, ids_df)
        try:
            rows = self.conn.execute(
                f"""
                SELECT s.sequence_id, s.sequence
                FROM sequences s
                INNER JOIN {_seq_for_ids_tmp} w ON s.sequence_id = w.sequence_id
                ORDER BY s.sequence_id
            """
            ).fetchall()
        finally:
            self.conn.unregister(_seq_for_ids_tmp)
        return [(r[0], r[1]) for r in rows]

    def make_filter_ws_name(self) -> str:
        """Mint a unique ``_filter_ws_<n>`` table name for one filter execution.

        Required when SQL-native filters can run in parallel at the same DAG
        level — sharing the canonical ``_filter_ws`` registration would let
        two stages clobber each other's working set.
        """
        return f"_filter_ws_{next(_TEMP_TABLE_COUNTER)}"

    def execute_filter_sql(
        self,
        sequence_ids: list[int],
        sql_query: str,
        ws_table_name: str = "_filter_ws",
    ) -> list[int]:
        """Execute a filter SQL query and return surviving sequence IDs.

        The *sql_query* must be a complete ``SELECT`` statement that returns
        ``sequence_id`` values and JOINs against *ws_table_name* (which
        contains the input *sequence_ids*) to scope to the working set.

        Pass a unique *ws_table_name* (use :meth:`make_filter_ws_name`) when
        running multiple SQL-native filters concurrently — the default
        ``_filter_ws`` name races otherwise.

        Args:
            sequence_ids: Input sequence IDs.
            sql_query: Complete SQL SELECT returning ``sequence_id`` values.
            ws_table_name: Name to register the working-set IDs under.  Must
                match the ``ws_table=`` argument passed to the filter's
                ``to_sql()``.

        Returns:
            List of sequence_ids that survive the filter.

        Raises:
            ValueError: If sql_query is not a single SELECT statement.
        """
        stripped = sql_query.strip()
        if not stripped.upper().startswith("SELECT"):
            raise ValueError("Filter SQL must be a single SELECT statement")
        if ";" in stripped:
            raise ValueError("Filter SQL must be a single SELECT statement")

        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register(ws_table_name, ids_df)
        try:
            result = self.conn.execute(sql_query).fetchall()
        finally:
            self.conn.unregister(ws_table_name)
        return [r[0] for r in result]

    def store_sequence_attributes(
        self,
        seq_ids: list[int],
        attr_name: str,
        attr_values: list[str],
    ):
        """Persist a per-sequence attribute column (e.g. heavy_chain, light_chain).

        Args:
            seq_ids: Sequence IDs.
            attr_name: Attribute name (column name from the input DataFrame).
            attr_values: Corresponding values (one per sequence_id).
        """
        if not seq_ids:
            return
        df = pd.DataFrame(
            {
                "sequence_id": seq_ids,
                "attr_name": attr_name,
                "attr_value": [str(v) if v is not None else None for v in attr_values],
            }
        )
        _attr_tmp = f"_attr_batch_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_attr_tmp, df)
        try:
            # use DuckDB ON CONFLICT DO UPDATE SET (INSERT OR REPLACE is SQLite syntax)
            self.conn.execute(
                f"""
                INSERT INTO sequence_attributes (sequence_id, attr_name, attr_value)
                SELECT sequence_id, attr_name, attr_value FROM {_attr_tmp}
                ON CONFLICT (sequence_id, attr_name) DO UPDATE SET attr_value = excluded.attr_value
            """
            )
        finally:
            self.conn.unregister(_attr_tmp)

    def get_sequence_attributes_for_ids(
        self,
        sequence_ids: list[int],
        attr_names: list[str],
    ) -> dict[int, dict[str, str]]:
        """Retrieve per-sequence attributes, returning {seq_id: {attr: value}}.

        Args:
            sequence_ids: Sequence IDs to look up.
            attr_names: Attribute names to retrieve.

        Returns:
            Nested dict: ``{sequence_id: {attr_name: attr_value}}``.
        """
        if not sequence_ids or not attr_names:
            return {}
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        _attr_ids_tmp = f"_attr_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_attr_ids_tmp, ids_df)
        try:
            placeholders = ", ".join(["?"] * len(attr_names))
            rows = self.conn.execute(
                f"""
                SELECT a.sequence_id, a.attr_name, a.attr_value
                FROM sequence_attributes a
                INNER JOIN {_attr_ids_tmp} w ON a.sequence_id = w.sequence_id
                WHERE a.attr_name IN ({placeholders})
            """,
                attr_names,
            ).fetchall()
        finally:
            self.conn.unregister(_attr_ids_tmp)

        result: dict[int, dict[str, str]] = {}
        for sid, name, val in rows:
            result.setdefault(int(sid), {})[name] = val
        return result

    # ------------------------------------------------------------------
    # Pipeline context methods (inter-stage key-value store)
    # ------------------------------------------------------------------

    def set_context(self, run_id: str, key: str, value: Any):
        """Store a key-value pair in the pipeline context table."""
        self.conn.execute(
            """
            INSERT INTO pipeline_context (run_id, key, value, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (run_id, key) DO UPDATE SET
                value = excluded.value, created_at = excluded.created_at
        """,
            [
                run_id,
                key,
                json.dumps(value) if not isinstance(value, str) else value,
                datetime.now(),
            ],
        )

    def get_context(self, run_id: str, key: str) -> Optional[Any]:
        """Retrieve a value from the pipeline context table."""
        result = self.conn.execute(
            "SELECT value FROM pipeline_context WHERE run_id = ? AND key = ?",
            [run_id, key],
        ).fetchone()
        if result is None:
            return None
        try:
            return json.loads(result[0])
        except (json.JSONDecodeError, TypeError):
            return result[0]

    def get_sequences_for_ids_with_columns(
        self,
        sequence_ids: list[int],
        columns: list[str],
    ) -> dict[int, dict[str, str]]:
        """Fetch column values from the sequences table for given IDs.

        This reads columns stored directly on the ``sequences`` table
        (via ``ensure_input_columns``), NOT from ``sequence_attributes``.

        Args:
            sequence_ids: Sequence IDs to look up.
            columns: Column names to fetch (must exist on the sequences table).

        Returns:
            ``{sequence_id: {col: value, ...}}``
        """
        if not sequence_ids or not columns:
            return {}
        for col in columns:
            _validate_column_name(col)
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        # unique temp name to avoid concurrent coroutine collision
        _col_ids_tmp = f"_col_ids_{next(_TEMP_TABLE_COUNTER)}"
        self.conn.register(_col_ids_tmp, ids_df)
        try:
            col_list = ", ".join(f's."{c}"' for c in columns)
            rows = self.conn.execute(
                f"""
                SELECT s.sequence_id, {col_list}
                FROM sequences s
                INNER JOIN {_col_ids_tmp} w ON s.sequence_id = w.sequence_id
            """
            ).fetchall()
        finally:
            self.conn.unregister(_col_ids_tmp)

        result: dict[int, dict[str, str]] = {}
        for row in rows:
            sid = int(row[0])
            result[sid] = {col: row[i + 1] for i, col in enumerate(columns)}

        # Warn if any returned rows have NULL in the requested extra columns.
        # This can happen when sequences were originally added without those columns.
        # We do NOT drop these rows — they may be intentionally sparse — but downstream
        # API calls will receive empty/None fields which is almost never desired.
        extra_cols = [c for c in columns if c not in ("sequence_id", "sequence", "length", "hash")]
        if extra_cols and result:
            bad_ids = [
                sid
                for sid, col_vals in result.items()
                if any(col_vals.get(c) is None for c in extra_cols)
            ]
            if bad_ids:
                warnings.warn(
                    f"Sequences {bad_ids} have NULL values in input columns "
                    f"{extra_cols}. API calls may receive empty fields.",
                    UserWarning,
                    stacklevel=2,
                )

        return result

    # ------------------------------------------------------------------
    # Pipeline definition methods (for from_db() recovery)
    # ------------------------------------------------------------------

    def save_pipeline_definition(
        self,
        definition_id: str,
        pipeline_type: str,
        input_schema_json: Optional[str],
        stages_json: str,
    ):
        """Persist a pipeline definition. Updates stages/schema if definition_id already exists."""
        # SAVE-01: ON CONFLICT DO UPDATE so the most recent definition always wins
        self.conn.execute(
            """
            INSERT INTO pipeline_definitions
                (definition_id, pipeline_type, input_schema_json, stages_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (definition_id) DO UPDATE SET
                stages_json = excluded.stages_json,
                input_schema_json = excluded.input_schema_json
        """,
            [definition_id, pipeline_type, input_schema_json, stages_json, datetime.now()],
        )

    def store_blob(self, content: str) -> str:
        """Store a large string value, returning its blob_id (SHA-256[:32]).

        Content-addressed: identical content always maps to the same blob_id.
        Safe to call multiple times with the same content (INSERT OR IGNORE).
        """
        import hashlib
        blob_id = hashlib.sha256(content.encode()).hexdigest()[:32]
        self.conn.execute(
            "INSERT INTO pipeline_blobs (blob_id, content, created_at) "
            "VALUES (?, ?, ?) ON CONFLICT (blob_id) DO NOTHING",
            [blob_id, content, datetime.now()],
        )
        return blob_id

    def load_blob(self, blob_id: str) -> Optional[str]:
        """Retrieve a stored blob by its blob_id. Returns None if not found."""
        row = self.conn.execute(
            "SELECT content FROM pipeline_blobs WHERE blob_id = ?", [blob_id]
        ).fetchone()
        return row[0] if row else None

    def load_pipeline_definition(self, definition_id: Optional[str] = None) -> Optional[dict]:
        """Load a pipeline definition by ID, or the latest one if ID is None."""
        if definition_id is not None:
            row = self.conn.execute(
                "SELECT definition_id, pipeline_type, input_schema_json, stages_json "
                "FROM pipeline_definitions WHERE definition_id = ?",
                [definition_id],
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT definition_id, pipeline_type, input_schema_json, stages_json "
                "FROM pipeline_definitions ORDER BY created_at DESC, definition_id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "definition_id": row[0],
            "pipeline_type": row[1],
            "input_schema_json": row[2],
            "stages_json": row[3],
        }

    def get_latest_definition_id(self) -> Optional[str]:
        """Return the definition_id of the most recently created pipeline definition."""
        row = self.conn.execute(
            "SELECT definition_id FROM pipeline_definitions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def get_latest_run_id(self, definition_id: Optional[str] = None) -> Optional[str]:
        """Return the run_id of the most recent pipeline run.

        ``from_db()`` must reuse the existing run_id so that resume
        can find already-completed stages.  A new run_id means no stages are
        marked complete, causing everything to re-run.

        Args:
            definition_id: If provided, restrict to runs that used this definition.
                           If None, return the most recent run across all definitions.

        Returns:
            The run_id string, or None if no runs exist.
        """
        if definition_id is not None:
            row = self.conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE definition_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                [definition_id],
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT run_id FROM pipeline_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def register_column(
        self,
        column_name: str,
        model_name: str,
        action: str,
        definition_id: str,
        stage_name: str,
    ):
        """Register an output column in the prediction_column_registry. Idempotent."""
        self.conn.execute(
            """
            INSERT INTO prediction_column_registry
                (column_name, model_name, action, definition_id, stage_name, registered_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (column_name, model_name, action) DO NOTHING
        """,
            [column_name, model_name, action, definition_id, stage_name],
        )

    def get_column_registry_entry(self, column_name: str) -> Optional[dict]:
        """Return the registry entry for a column, or None if not registered."""
        # ORDER BY registered_at DESC so we get the most recently registered
        # entry by insertion time, not by lexicographic sort of definition_id (a content hash).
        row = self.conn.execute(
            "SELECT model_name, action, definition_id, stage_name "
            "FROM prediction_column_registry WHERE column_name = ? "
            "ORDER BY registered_at DESC, definition_id DESC LIMIT 1",
            [column_name],
        ).fetchone()
        if row is None:
            return None
        return {
            "model_name": row[0],
            "action": row[1],
            "definition_id": row[2],
            "stage_name": row[3],
        }

    def get_existing_input_columns(self) -> list[str]:
        """Return extra columns on the sequences table (beyond the base schema).

        Used for input-schema validation when connecting to an existing DB.
        Returns an empty list if only the base columns are present.
        """
        return sorted(self._extra_columns)

    def close(self):
        """Close the DuckDB connection and release the file lock.

        Safe to call multiple times — subsequent calls are no-ops.
        Called automatically by ``__exit__`` and ``__del__``.
        """
        # guard against double-close and missing attribute
        if hasattr(self, "conn") and self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def __del__(self):
        """Close connection on garbage collection to prevent file-lock leaks."""
        self.close()

    def __repr__(self):
        """String representation."""
        if not hasattr(self, "conn") or self.conn is None:
            return f"DuckDBDataStore(db='{self.db_path}', closed=True)"
        try:
            seq_count = self.conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
            pred_count = self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            emb_count = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            struct_count = self.conn.execute("SELECT COUNT(*) FROM structures").fetchone()[
                0
            ]
        except Exception:
            return f"DuckDBDataStore(db='{self.db_path}', closed=True)"

        return (
            f"DuckDBDataStore(db='{self.db_path}', "
            f"sequences={seq_count}, predictions={pred_count}, "
            f"embeddings={emb_count}, structures={struct_count})"
        )
