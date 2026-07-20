"""response_mapping → PredictionStage extraction specs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from biolm.core.expression_evaluator import extract_template_expr

_STRUCTURE_KEYS = frozenset({"pdb", "cif", "pdbs", "structure"})
_JMESPATH_TAIL = re.compile(r"^response\.(.+)$")


@dataclass
class MappingSpec:
    """Parsed response_mapping entry."""

    column: str
    kind: str  # scalar | structure | embedding
    response_key: str
    reduction: Optional[str] = None
    jmespath: Optional[str] = None


def _tail_from_jmespath(path: str) -> str:
    """Extract final field name from a JMESPath like results[*].pdb → pdb."""
    path = path.strip()
    if not path:
        return path
    # Strip common response wrapper prefix used in protocol YAML
    m = _JMESPATH_TAIL.match(path)
    if m:
        path = m.group(1)
    # Take last segment after . or [*].
    parts = re.split(r"\.|\[\*\]", path)
    parts = [p for p in parts if p]
    return parts[-1] if parts else path


def parse_mapping_entry(column: str, value: Any) -> MappingSpec:
    """Parse one response_mapping key/value pair."""
    if isinstance(value, dict):
        path = value.get("path") or value.get("jmespath") or ""
        if not path:
            raise ValueError(
                f"response_mapping['{column}'] object must include 'path'."
            )
        key = _tail_from_jmespath(str(path))
        return _classify(column, key, jmespath=str(path))

    if isinstance(value, str):
        is_tmpl, expr = extract_template_expr(value)
        if is_tmpl:
            key = _tail_from_jmespath(expr)
            return _classify(column, key, jmespath=expr)
        return _classify(column, value)

    raise ValueError(
        f"response_mapping['{column}'] must be a string or object with 'path'."
    )


def _classify(
    column: str,
    response_key: str,
    *,
    jmespath: Optional[str] = None,
    reduction: Optional[str] = None,
) -> MappingSpec:
    key_lower = response_key.lower()
    if key_lower in _STRUCTURE_KEYS or "pdb" in key_lower or "cif" in key_lower:
        return MappingSpec(
            column=column,
            kind="structure",
            response_key=response_key,
            jmespath=jmespath,
        )
    if key_lower in ("embedding", "embeddings", "mean_representations", "seqcoding"):
        return MappingSpec(
            column=column,
            kind="embedding",
            response_key=response_key,
            jmespath=jmespath,
        )
    return MappingSpec(
        column=column,
        kind="scalar",
        response_key=response_key,
        reduction=reduction,
        jmespath=jmespath,
    )


def mapping_to_stage_kwargs(
    action: str,
    response_mapping: dict[str, Any],
) -> dict[str, Any]:
    """Convert protocol response_mapping to PredictionStage constructor kwargs."""
    from biolm.pipeline.data import EmbeddingSpec, StructureSpec

    if not response_mapping:
        if action == "encode":
            return {"embedding_extractor": EmbeddingSpec("mean_representations")}
        raise ValueError("response_mapping is required for predict/score tasks.")

    extractions: list[str] = []
    columns: dict[str, str] = {}
    structure_output = None
    embedding_extractor = None
    plddt_key = None

    for col_name, raw in response_mapping.items():
        spec = parse_mapping_entry(col_name, raw)
        if spec.kind == "structure":
            structure_output = StructureSpec(key=spec.response_key)
        elif spec.kind == "embedding" or action == "encode":
            embedding_extractor = EmbeddingSpec(key=spec.response_key)
        else:
            extractions.append(spec.response_key)
            columns[spec.response_key] = col_name
            if spec.response_key in ("mean_plddt", "plddt") and structure_output:
                plddt_key = spec.response_key

    if structure_output and plddt_key:
        structure_output = StructureSpec(
            key=structure_output.key,
            format=structure_output.format,
            plddt_key=plddt_key,
            index=structure_output.index,
        )

    kwargs: dict[str, Any] = {}
    if extractions:
        kwargs["extractions"] = extractions
        if columns:
            kwargs["columns"] = columns
    if structure_output is not None:
        kwargs["structure_output"] = structure_output
    if embedding_extractor is not None:
        kwargs["embedding_extractor"] = embedding_extractor
    return kwargs
