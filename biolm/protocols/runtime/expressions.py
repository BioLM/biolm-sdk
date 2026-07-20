"""Recursive evaluation of protocol template expressions."""

from __future__ import annotations

from typing import Any

from biolm.core.expression_evaluator import evaluate_template_value


def evaluate_value(value: Any, context: dict[str, Any]) -> Any:
    """Evaluate ${{ }} templates in a scalar value."""
    return evaluate_template_value(value, context)


def evaluate_tree(obj: Any, context: dict[str, Any]) -> Any:
    """Recursively evaluate template expressions in dicts/lists."""
    if isinstance(obj, dict):
        return {k: evaluate_tree(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [evaluate_tree(v, context) for v in obj]
    return evaluate_value(obj, context)
