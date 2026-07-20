"""Protocol output rules — result selection shared by local runs and MLflow logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biolm.core.expression_evaluator import (
    evaluate_expression,
    evaluate_where_clause,
    extract_template_expr,
)


@dataclass
class OutputSelection:
    """Rows selected by a single protocol ``outputs[]`` rule."""

    rule_index: int
    rule: dict[str, Any]
    records: list[dict[str, Any]]


def select_results(results: list[dict[str, Any]], output_rule: dict[str, Any]) -> list[dict[str, Any]]:
    """Select results based on an output rule (where, order_by, limit)."""
    selected = list(results)

    where_expr = output_rule.get("where")
    if where_expr:
        selected = [
            row for row in selected if evaluate_where_clause(where_expr, row)
        ]

    order_by = output_rule.get("order_by", [])
    if order_by:
        for order_spec in reversed(order_by):
            field = order_spec.get("field")
            order = order_spec.get("order", "asc")
            reverse = order == "desc"
            selected.sort(key=lambda x: x.get(field), reverse=reverse)

    limit = output_rule.get("limit", 200)
    if limit is not None:
        is_template, expr = extract_template_expr(str(limit))
        if is_template:
            if selected:
                limit = int(evaluate_expression(expr, selected[0]))
            else:
                limit = 200
        else:
            limit = int(limit)
        selected = selected[:limit]

    return selected


def apply_protocol_outputs(
    records: list[dict[str, Any]],
    output_rules: list[dict[str, Any]] | None,
) -> tuple[list[OutputSelection], list[dict[str, Any]]]:
    """Apply all protocol output rules and return per-rule and union selections."""
    if not output_rules:
        return [], []

    selections: list[OutputSelection] = []
    union: list[dict[str, Any]] = []

    for idx, rule in enumerate(output_rules):
        selected = select_results(records, rule)
        selections.append(OutputSelection(rule_index=idx, rule=rule, records=selected))
        for row in selected:
            if not any(row is existing or row == existing for existing in union):
                union.append(row)

    return selections, union
