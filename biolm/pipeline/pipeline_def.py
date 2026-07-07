"""
Pipeline definition persistence and reconstruction.

Provides factory functions for serializing pipeline definitions to DuckDB
and reconstructing pipeline objects from stored definitions (for kernel-death
recovery via DataPipeline.from_db() / GenerativePipeline.from_db()).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from biolm.pipeline.base import BasePipeline, Stage
    from biolm.pipeline.datastore_duckdb import DuckDBDataStore
    from biolm.pipeline.filters import BaseFilter


# Strings longer than this are externalized to pipeline_blobs rather than
# stored inline in stages_json.  5 000 chars is safely above any protein
# sequence (~3 000 AA max) but well below any PDB file (~10 000+ chars).
_BLOB_THRESHOLD = 5_000


def _extract_blobs(obj: Any, datastore: "DuckDBDataStore") -> Any:
    """Recursively walk *obj*, replacing large strings with blob references.

    Modifies nothing in-place — returns a new structure.  Any string longer
    than ``_BLOB_THRESHOLD`` chars is stored in ``pipeline_blobs`` and replaced
    with ``{"_blob_ref": blob_id}``.
    """
    if isinstance(obj, str):
        if len(obj) > _BLOB_THRESHOLD:
            blob_id = datastore.store_blob(obj)
            return {"_blob_ref": blob_id}
        return obj
    if isinstance(obj, dict):
        return {k: _extract_blobs(v, datastore) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_extract_blobs(item, datastore) for item in obj]
    return obj


def _resolve_blobs(obj: Any, datastore: "DuckDBDataStore") -> Any:
    """Inverse of ``_extract_blobs``: replace blob references with their content.

    Raises ``ValueError`` if a blob_id is not found in the datastore (DB was
    truncated or the blob was deleted).
    """
    if isinstance(obj, dict):
        if "_blob_ref" in obj and len(obj) == 1:
            blob_id = obj["_blob_ref"]
            content = datastore.load_blob(blob_id)
            if content is None:
                raise ValueError(
                    f"Pipeline definition references blob '{blob_id}' which is "
                    "not in the pipeline_blobs table.  The database may be "
                    "incomplete or the blob was deleted."
                )
            return content
        return {k: _resolve_blobs(v, datastore) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_blobs(item, datastore) for item in obj]
    return obj


def _pipeline_def_hash(
    pipeline_type: str,
    input_schema_cols: Optional[list[str]],
    stages_specs: list[dict],
) -> str:
    """Content-hash of the pipeline definition (32 hex chars / 128 bits).

    Same stages + same params = same hash = one row in ``pipeline_definitions``.
    Multiple runs share one definition row.

    Using 32 hex chars (128 bits) for collision resistance — 16 chars (64 bits)
    has a non-trivial birthday-attack risk when many definitions are stored.
    """
    payload = json.dumps(
        {
            "pipeline_type": pipeline_type,
            "input_schema": sorted(input_schema_cols) if input_schema_cols else None,
            "stages": stages_specs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def filter_from_spec(spec: dict) -> "BaseFilter":
    """Reconstruct a :class:`BaseFilter` from its ``to_spec()`` dict.

    Raises:
        NotImplementedError: For ``CustomFilter`` (func is not serializable).
        ValueError: For unknown filter types.
    """
    from biolm.pipeline.filters import (
        ConservedResidueFilter,
        DiversitySamplingFilter,
        HammingDistanceFilter,
        RankingFilter,
        SequenceLengthFilter,
        ThresholdFilter,
        ValidAminoAcidFilter,
    )

    ftype = spec.get("type")
    if ftype == "ThresholdFilter":
        return ThresholdFilter(
            column=spec["column"],
            min_value=spec.get("min_value"),
            max_value=spec.get("max_value"),
            keep_na=spec.get("keep_na", False),
        )
    elif ftype == "SequenceLengthFilter":
        return SequenceLengthFilter(
            min_length=spec.get("min_length"),
            max_length=spec.get("max_length"),
        )
    elif ftype == "HammingDistanceFilter":
        return HammingDistanceFilter(
            reference_sequence=spec["reference_sequence"],
            max_distance=spec.get("max_distance"),
            min_distance=spec.get("min_distance"),
            normalize=spec.get("normalize", False),
        )
    elif ftype == "RankingFilter":
        return RankingFilter(
            column=spec["column"],
            n=spec.get("n"),
            ascending=spec.get("ascending", False),
            method=spec.get("method", "top"),
            percentile=spec.get("percentile"),
        )
    elif ftype == "CustomFilter":
        raise NotImplementedError(
            f"CustomFilter (name='{spec.get('name', 'unknown')}') cannot be "
            "reconstructed automatically — the filter function (callable) is not "
            "serializable. Re-create this filter manually and call add_filter() again."
        )
    elif ftype == "ConservedResidueFilter":
        # Keys were stored as strings for JSON compat; convert back to int
        conserved_positions = {
            int(k): v for k, v in spec["conserved_positions"].items()
        }
        return ConservedResidueFilter(
            conserved_positions=conserved_positions,
            reference_length=spec.get("reference_length"),
        )
    elif ftype == "DiversitySamplingFilter":
        return DiversitySamplingFilter(
            n_samples=spec["n_samples"],
            method=spec.get("method", "random"),
            score_column=spec.get("score_column"),
            random_seed=spec.get("random_seed", 42),
            resample=spec.get("resample", True),
        )
    elif ftype == "ValidAminoAcidFilter":
        return ValidAminoAcidFilter(
            alphabet=spec.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"),
            verbose=spec.get("verbose", True),
            column=spec.get("column", "sequence"),
        )
    elif ftype == "CompositeFilter":
        from biolm.pipeline.filters import CompositeFilter
        sub_filters = [filter_from_spec(f) for f in spec.get("filters", [])]
        return CompositeFilter(*sub_filters)
    else:
        raise ValueError(
            f"Unknown filter type '{ftype}' in spec. "
            "Cannot reconstruct filter — re-create it manually."
        )


def _config_from_spec(spec: dict) -> Any:
    """Reconstruct a generation config object from its ``to_spec()`` dict."""
    from biolm.pipeline.generative import (
        DirectGenerationConfig,
        IterativeMaskingDMSConfig,
        SaturationMutagenesisConfig,
        SequenceSourceConfig,
    )
    from biolm.pipeline.mlm_remasking import RemaskingConfig

    ctype = spec.get("type")
    if ctype == "SequenceSourceConfig":
        # PD-01: honour serialized from_db / sequences values.
        # When from_db=False and a sequences list was stored, restore it directly
        # instead of unconditionally forcing from_db=True (which would silently
        # discard the stored sequence list).
        from_db = spec.get("from_db", True)
        sequences = spec.get("sequences")
        if not from_db and isinstance(sequences, list):
            return SequenceSourceConfig(
                sequences=sequences,
                column=spec.get("column", "sequence"),
                from_db=False,
            )
        return SequenceSourceConfig(
            sequences=None,
            column=spec.get("column", "sequence"),
            from_db=True,
        )
    elif ctype == "DirectGenerationConfig":
        return DirectGenerationConfig(
            model_name=spec["model_name"],
            structure_path=spec.get("structure_path"),
            structure_column=spec.get("structure_column"),
            sequence=spec.get("sequence"),
            item_field=spec.get("item_field", "pdb"),
            params=spec.get("params", {}),
            num_sequences=spec.get("num_sequences", 100),
            temperature=spec.get("temperature", 1.0),
            structure_from_stage=spec.get("structure_from_stage"),
            structure_from_model=spec.get("structure_from_model"),
            n_runs=spec.get("n_runs", 1),
            label=spec.get("label"),
        )
    elif ctype == "SaturationMutagenesisConfig":
        return SaturationMutagenesisConfig(
            parent_sequence=spec["parent_sequence"],
            scoring_model=spec["scoring_model"],
            positions=spec.get("positions"),
            alphabet=spec.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"),
            scoring_action=spec.get("scoring_action", "predict"),
            scoring_params=spec.get("scoring_params", {}),
            score_field=spec.get("score_field", "ddg"),
            top_n=spec.get("top_n", 50),
            ascending=spec.get("ascending", True),
            exclude_synonymous=spec.get("exclude_synonymous", True),
            batch_size=spec.get("batch_size", 8),
            label=spec.get("label"),
            pdb_str=spec.get("pdb_str"),
            chain=spec.get("chain", "A"),
        )
    elif ctype == "IterativeMaskingDMSConfig":
        return IterativeMaskingDMSConfig(
            parent_sequence=spec["parent_sequence"],
            model_name=spec["model_name"],
            positions=spec.get("positions"),
            rounds=spec.get("rounds", 2),
            mask_token=spec.get("mask_token", "<mask>"),
            alphabet=spec.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"),
            exclude_synonymous=spec.get("exclude_synonymous", True),
            batch_size=spec.get("batch_size", 32),
            label=spec.get("label"),
            action=spec.get("action", "predict"),
        )
    elif ctype == "RemaskingConfig":
        return RemaskingConfig(
            model_name=spec["model_name"],
            action=spec.get("action", "predict"),
            mask_fraction=spec.get("mask_fraction", 0.15),
            mask_positions=spec.get("mask_positions", "auto"),
            num_iterations=spec.get("num_iterations", 1),
            temperature=spec.get("temperature", 1.0),
            top_k=spec.get("top_k"),
            top_p=spec.get("top_p"),
            mask_token=spec.get("mask_token", "<mask>"),
            conserved_positions=spec.get("conserved_positions"),
            mask_strategy=spec.get("mask_strategy", "random"),
            block_size=spec.get("block_size", 3),
            confidence_threshold=spec.get("confidence_threshold", 0.8),
            # PD-13: restore parent_sequence and num_variants
            parent_sequence=spec.get("parent_sequence"),
            num_variants=spec.get("num_variants", 100),
        )
    else:
        raise ValueError(
            f"Unknown generation config type '{ctype}'. "
            "Supported types: SequenceSourceConfig, RemaskingConfig, "
            "DirectGenerationConfig, SaturationMutagenesisConfig, "
            "IterativeMaskingDMSConfig."
        )


def _embedding_extractor_from_spec(spec: Optional[dict]):
    """Reconstruct an EmbeddingSpec (or raise NotImplementedError for callables)."""
    if spec is None:
        return None
    if spec.get("type") == "EmbeddingSpec":
        from biolm.pipeline.data import EmbeddingSpec
        return EmbeddingSpec(
            key=spec["key"],
            layer=spec.get("layer"),
            reduction=spec.get("reduction"),
        )
    raise NotImplementedError(
        "Custom callable embedding_extractor cannot be reconstructed automatically. "
        "Use EmbeddingSpec(key=...) instead of a lambda/function, or re-attach "
        "embedding_extractor manually after calling from_db()."
    )


def stage_from_spec(spec: dict) -> "Stage":
    """Reconstruct a :class:`Stage` from its ``to_spec()`` dict.

    Raises:
        NotImplementedError: For stage types that cannot be auto-reconstructed.
        ValueError: For unknown stage types.
    """
    from biolm.pipeline.data import (
        ClusteringStage,
        CofoldingPredictionStage,
        EmbeddingSpec,
        FilterStage,
        PredictionStage,
        _ResolvedExtraction,
    )
    from biolm.pipeline.generative import GenerationStage

    stype = spec.get("type")

    # detect specs that were stored with a serialization error flag and
    # raise a clear, actionable error instead of trying to reconstruct a broken stage.
    if spec.get("_serialization_error"):
        err_msg = spec.get("_serialization_error_msg", "unknown error")
        raise NotImplementedError(
            f"Stage '{spec.get('name', '?')}' (type='{stype}') could not be "
            f"serialized when the pipeline definition was saved: {err_msg}. "
            "Re-create this stage manually and call add_stage()."
        )

    if stype == "PredictionStage":
        from biolm.pipeline.data import MatrixExtractionSpec, StructureSpec

        resolved = [
            _ResolvedExtraction(
                response_key=r["response_key"],
                column=r["column"],
                reduction=r.get("reduction"),
            )
            for r in spec.get("resolved", [])
        ]
        action = spec.get("action", "predict")
        emb_extractor = _embedding_extractor_from_spec(spec.get("embedding_extractor"))

        # Reconstruct structure_output / structure_input / matrix_extraction
        # before deciding whether the stage is "scalar-extraction-shaped".
        structure_output = None
        so_spec = spec.get("structure_output")
        if so_spec:
            structure_output = StructureSpec(
                key=so_spec["key"],
                format=so_spec.get("format"),
                plddt_key=so_spec.get("plddt_key"),
                index=so_spec.get("index", 0),
            )

        structure_input = spec.get("structure_input")

        matrix_extraction = None
        mx_spec = spec.get("matrix_extraction")
        if mx_spec:
            matrix_extraction = MatrixExtractionSpec(
                prefix=mx_spec["prefix"],
                values_key=mx_spec.get("values_key"),
                row_labels_key=mx_spec.get("row_labels_key"),
                col_labels_key=mx_spec.get("col_labels_key"),
                mutation_key=mx_spec.get("mutation_key"),
                value_key=mx_spec.get("value_key"),
            )

        # Build extractions + columns from resolved list (for __init__ call).
        # A stage with no resolved scalar extractions is valid when it produces
        # only structures or matrix predictions (structure_output / matrix_extraction)
        # — don't reject those as "incomplete".
        produces_non_scalar = structure_output is not None or matrix_extraction is not None
        if resolved and action in ("predict", "score"):
            extractions = [r.response_key for r in resolved]
            columns = {r.response_key: r.column for r in resolved}
        elif not resolved and action in ("predict", "score") and not produces_non_scalar:
            raise ValueError(
                f"PredictionStage '{spec.get('name', '?')}' (action='{action}') "
                "has no 'resolved' extractions in its stored spec and produces no "
                "structure or matrix output.  Re-create the stage manually and "
                "call add_stage()."
            )
        else:
            extractions = None
            columns = None

        stage = PredictionStage(
            name=spec["name"],
            model_name=spec["model_name"],
            action=action,
            extractions=extractions,
            columns=columns,
            params=spec.get("params", {}),
            batch_size=spec.get("batch_size", 32),
            max_concurrent=spec.get("max_concurrent", 5),
            max_connections=spec.get("max_connections", 10),
            item_columns=spec.get("item_columns"),
            embedding_extractor=emb_extractor,
            structure_output=structure_output,
            structure_input=structure_input,
            matrix_extraction=matrix_extraction,
            skip_on_error=spec.get("skip_on_error", False),
            depends_on=spec.get("depends_on", []),
        )
        return stage

    elif stype == "FilterStage":
        # PD-15: use safe .get() with an explicit, actionable error rather than
        # a raw KeyError that gives no context about which stage is broken.
        filter_spec = spec.get("filter_spec")
        if filter_spec is None:
            raise ValueError(
                f"FilterStage '{spec.get('name')}' spec is missing 'filter_spec' key. "
                "This stage cannot be reconstructed — re-create it manually and "
                "call add_stage()."
            )
        filter_obj = filter_from_spec(filter_spec)
        return FilterStage(
            name=spec["name"],
            filter_func=filter_obj,
            depends_on=spec.get("depends_on", []),
        )

    elif stype == "ClusteringStage":
        # PD-09: pass cluster_kwargs back to the constructor so custom
        # algorithm parameters (e.g. eps for DBSCAN) are restored.
        return ClusteringStage(
            name=spec["name"],
            method=spec["method"],
            n_clusters=spec.get("n_clusters"),
            similarity_metric=spec.get("similarity_metric", "embedding"),
            embedding_model=spec.get("embedding_model"),
            max_sample=spec.get("max_sample"),
            depends_on=spec.get("depends_on", []),
            **spec.get("cluster_kwargs", {}),
        )

    elif stype == "CofoldingPredictionStage":
        # PD-10: static_entities are not serializable (they contain structure data).
        # Warn the user so they know to re-attach them before calling .run().
        warnings.warn(
            f"CofoldingPredictionStage '{spec['name']}' was reconstructed without "
            "static_entities. Re-attach them via stage.static_entities = [...] "
            "before calling .run().",
            UserWarning,
            stacklevel=2,
        )
        return CofoldingPredictionStage(
            name=spec["name"],
            model_name=spec["model_name"],
            action=spec.get("action", "predict"),
            prediction_type=spec.get("prediction_type", "structure"),
            sequence_chain_id=spec.get("sequence_chain_id", "A"),
            sequence_entity_type=spec.get("sequence_entity_type", "protein"),
            static_entities=[],  # not serializable — re-attach manually
            params=spec.get("params", {}),
            batch_size=spec.get("batch_size", 1),
            depends_on=spec.get("depends_on", []),
        )

    elif stype == "GenerationStage":
        configs = [_config_from_spec(c) for c in spec.get("configs", [])]
        return GenerationStage(
            name=spec["name"],
            configs=configs,
            deduplicate=spec.get("deduplicate", True),
            depends_on=spec.get("depends_on", []),
        )

    else:
        raise NotImplementedError(
            f"Stage type '{stype}' (name='{spec.get('name', '?')}') cannot be "
            "reconstructed automatically. Re-create it manually and call add_stage()."
        )


def pipeline_from_definition(
    defn: dict,
    datastore: "DuckDBDataStore",
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> "BasePipeline":
    """Reconstruct a pipeline from a stored definition dict.

    Args:
        defn: Dict as returned by ``DuckDBDataStore.load_pipeline_definition()``.
        datastore: Open DuckDB datastore to attach to the pipeline.
        run_id: Run ID to resume (``None`` = look up the most recent run for this
            definition, or generate a fresh one if no prior runs exist).
            when resuming via ``from_db()``, we must reuse the existing
            run_id so that ``stage_id = f"{run_id}_{stage_name}"`` matches the
            stage_completions rows written by the previous run.  Passing a new
            run_id causes all stages to appear incomplete and re-run.
        verbose: Enable verbose output.

    Returns:
        A fully configured :class:`DataPipeline` or :class:`GenerativePipeline`
        ready to call ``.run(resume=True)`` on.

    Raises:
        NotImplementedError: If any stage cannot be auto-reconstructed
            (e.g. CustomFilter, custom callable embedding_extractor).
        ValueError: For unknown pipeline types.
    """
    from biolm.pipeline.data import DataPipeline
    from biolm.pipeline.generative import GenerativePipeline

    pipeline_type = defn["pipeline_type"]
    input_schema_json = defn.get("input_schema_json")
    stages_json = defn["stages_json"]
    definition_id = defn.get("definition_id")

    input_cols = json.loads(input_schema_json) if input_schema_json else None
    stages_specs = json.loads(stages_json)
    # Resolve any blob references (large strings stored externally in pipeline_blobs)
    if datastore is not None:
        stages_specs = _resolve_blobs(stages_specs, datastore)

    # if no run_id supplied, look up the most recent run for this
    # definition so that resume finds already-completed stages correctly.
    if run_id is None and datastore is not None:
        # PD-04: when definition_id is None, warn that multiple definitions sharing
        # the same DB may cause the wrong run to be selected.
        if definition_id is None:
            warnings.warn(
                "pipeline_from_definition() called without a definition_id — "
                "selecting the most recent run across ALL pipeline definitions in "
                "this database. If multiple definitions share the same DB, the wrong "
                "run may be selected. Call BasePipeline.from_db(definition_id=...) "
                "explicitly to avoid ambiguity.",
                UserWarning,
                stacklevel=2,
            )
        run_id = datastore.get_latest_run_id(definition_id=definition_id)

    # PD-08: validate that the resolved run_id actually exists in pipeline_runs.
    # A stale or mis-typed run_id would silently cause all stages to appear
    # incomplete and re-run from scratch.
    if run_id is not None and datastore is not None:
        row = datastore.get_pipeline_run(run_id)
        if row is None:
            warnings.warn(
                f"run_id '{run_id}' not found in pipeline_runs. "
                "Starting fresh run.",
                UserWarning,
                stacklevel=2,
            )
            run_id = None

    # Create the pipeline without sequences (resume=True path).
    # PD-16: pass resume=True as an explicit named keyword argument (not buried
    # in **kwargs) so it is visible and type-checkable by DataPipeline.__init__.
    kwargs: dict[str, Any] = {
        "datastore": datastore,
        "verbose": verbose,
    }
    if run_id is not None:
        kwargs["run_id"] = run_id
    if input_cols is not None:
        kwargs["input_columns"] = input_cols

    if pipeline_type == "DataPipeline":
        pipeline = DataPipeline(sequences=None, resume=True, **kwargs)
    elif pipeline_type == "GenerativePipeline":
        # GenerativePipeline passes **kwargs to BasePipeline which does not
        # accept input_columns — convert to input_schema here.
        # PD-16: pass resume=True explicitly (not buried in **kwargs).
        gp_kwargs = dict(kwargs)
        if "input_columns" in gp_kwargs:
            from biolm.pipeline.base import InputSchema
            gp_kwargs["input_schema"] = InputSchema(columns=gp_kwargs.pop("input_columns"))
        pipeline = GenerativePipeline(resume=True, **gp_kwargs)
    else:
        raise ValueError(
            f"Unknown pipeline type '{pipeline_type}'. "
            "Cannot reconstruct — only DataPipeline and GenerativePipeline are supported."
        )

    # Reconstruct and add stages.
    # PD-02: collect ALL reconstruction errors before raising so the caller sees
    # every broken stage at once (not just the first one).
    reconstruction_errors: list[str] = []
    reconstructed_stages = []
    for stage_spec in stages_specs:
        try:
            stage = stage_from_spec(stage_spec)
            reconstructed_stages.append(stage)
        except (NotImplementedError, ValueError) as exc:
            reconstruction_errors.append(
                f"  - Stage '{stage_spec.get('name', '?')}' "
                f"(type='{stage_spec.get('type', '?')}'): {exc}"
            )
        except Exception as exc:
            reconstruction_errors.append(
                f"  - Stage '{stage_spec.get('name', '?')}' "
                f"(type='{stage_spec.get('type', '?')}'): unexpected error: {exc}"
            )

    if reconstruction_errors:
        raise RuntimeError(
            f"Failed to reconstruct {len(reconstruction_errors)} stage(s) from the "
            "stored pipeline definition:\n"
            + "\n".join(reconstruction_errors)
            + "\n\nTo recover: re-create the failing stages manually and call "
            "pipeline.add_stage() on the reconstructed pipeline, then call "
            "pipeline.run(resume=True)."
        )

    for stage in reconstructed_stages:
        pipeline.add_stage(stage)

    return pipeline
