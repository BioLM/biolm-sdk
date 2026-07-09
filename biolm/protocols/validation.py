"""Protocol YAML validation (schema + semantic checks)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROTOCOL_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "schema",
    "protocol_schema.json",
)


@dataclass
class ValidationError:
    """Represents a single validation error."""

    message: str
    path: str = ""
    error_type: str = "unknown"


@dataclass
class ProtocolValidationResult:
    """Result of protocol validation."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str, path: str = "", error_type: str = "unknown"):
        self.errors.append(ValidationError(message=message, path=path, error_type=error_type))
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)


def load_yaml(yaml_path: str) -> dict:
    """Load a protocol YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for protocol support. Install with: pip install pyyaml"
        ) from exc
    try:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Failed to load protocol YAML: {e}") from e


def validate_protocol_file(yaml_path: str) -> ProtocolValidationResult:
    """Validate a protocol YAML file."""
    result = ProtocolValidationResult(is_valid=True)

    try:
        data = load_yaml(yaml_path)
    except Exception as e:
        result.add_error(f"Failed to parse YAML: {e}", path="", error_type="syntax")
        return result

    _validate_schema(data, result)
    if data:
        _validate_task_references(data, result)
        _validate_circular_dependencies(data, result)
        _validate_template_expressions(data, result)
        _collect_statistics(data, result)

    return result


def _validate_schema(data: dict, result: ProtocolValidationResult) -> None:
    if not os.path.exists(PROTOCOL_SCHEMA_PATH):
        result.add_warning("Schema file not found, skipping schema validation")
        return

    try:
        import jsonschema
    except ImportError:
        result.add_warning("jsonschema not installed, skipping schema validation")
        return

    try:
        with open(PROTOCOL_SCHEMA_PATH, "r") as f:
            schema = json.load(f)

        validator = jsonschema.Draft202012Validator(schema)
        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.path)
            result.add_error(
                error.message,
                path=path if path else "root",
                error_type="schema",
            )
    except Exception as e:
        result.add_error(f"Schema validation error: {e}", path="", error_type="schema")


def _validate_task_references(data: dict, result: ProtocolValidationResult) -> None:
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return

    task_ids: set[str] = set()
    for i, task in enumerate(tasks):
        if isinstance(task, dict):
            task_id = task.get("id")
            if task_id:
                if task_id in task_ids:
                    result.add_error(
                        f"Duplicate task ID: '{task_id}'",
                        path=f"tasks[{i}].id",
                        error_type="semantic",
                    )
                task_ids.add(task_id)

    input_names: set[str] = set()
    inputs = data.get("inputs", {})
    if isinstance(inputs, dict):
        input_names = set(inputs.keys())
    valid_from_targets = task_ids | input_names

    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue

        task_id = task.get("id", f"tasks[{i}]")
        base_path = f"tasks[{i}]"

        depends_on = task.get("depends_on", [])
        if isinstance(depends_on, list):
            for j, dep in enumerate(depends_on):
                if isinstance(dep, str) and dep not in task_ids:
                    result.add_error(
                        f"Task '{task_id}' references unknown task ID '{dep}' in depends_on",
                        path=f"{base_path}.depends_on[{j}]",
                        error_type="semantic",
                    )

        from_task = task.get("from")
        if isinstance(from_task, str) and from_task not in valid_from_targets:
            result.add_error(
                f"Task '{task_id}' references unknown task or input '{from_task}' in from",
                path=f"{base_path}.from",
                error_type="semantic",
            )

        foreach = task.get("foreach")
        if isinstance(foreach, str) and not foreach.startswith("${{"):
            if foreach not in task_ids:
                result.add_error(
                    f"Task '{task_id}' references unknown task ID '{foreach}' in foreach",
                    path=f"{base_path}.foreach",
                    error_type="semantic",
                )


def _validate_circular_dependencies(data: dict, result: ProtocolValidationResult) -> None:
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return

    graph: dict[str, list[str]] = {}
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_id = task.get("id")
        if not task_id:
            continue
        depends_on = task.get("depends_on", [])
        graph[task_id] = [dep for dep in depends_on if isinstance(dep, str)] if isinstance(depends_on, list) else []

    visited: set[str] = set()
    rec_stack: set[str] = set()

    def has_cycle(node: str, path: List[str]) -> Optional[List[str]]:
        if node in rec_stack:
            cycle_start = path.index(node)
            return path[cycle_start:] + [node]
        if node in visited:
            return None
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        for neighbor in graph.get(node, []):
            if neighbor in graph:
                cycle = has_cycle(neighbor, path)
                if cycle:
                    return cycle
        rec_stack.remove(node)
        path.pop()
        return None

    for task_id in graph:
        if task_id not in visited:
            cycle = has_cycle(task_id, [])
            if cycle:
                result.add_error(
                    f"Circular dependency detected: {' -> '.join(cycle)}",
                    path="tasks",
                    error_type="semantic",
                )
                break


def _validate_template_expressions(data: dict, result: ProtocolValidationResult) -> None:
    def check_value(value: Any, path: str) -> None:
        if isinstance(value, str):
            if value.startswith("${{") and value.endswith("}}"):
                inner = value[3:-2].strip()
                if not inner:
                    result.add_error("Empty template expression", path=path, error_type="semantic")
                brace_count = 0
                for char in inner:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count < 0:
                            result.add_error(
                                "Unbalanced braces in template expression",
                                path=path,
                                error_type="semantic",
                            )
                            break
                if brace_count != 0:
                    result.add_error(
                        "Unbalanced braces in template expression",
                        path=path,
                        error_type="semantic",
                    )
            elif "${{" in value or "}}" in value:
                result.add_error(
                    "Malformed template expression (missing ${{ or }})",
                    path=path,
                    error_type="semantic",
                )
        elif isinstance(value, dict):
            for key, val in value.items():
                check_value(val, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                check_value(item, f"{path}[{i}]" if path else f"[{i}]")

    check_value(data, "")


def _collect_statistics(data: dict, result: ProtocolValidationResult) -> None:
    stats: dict[str, Any] = {}

    inputs = data.get("inputs", {})
    stats["input_count"] = len(inputs) if isinstance(inputs, dict) else 0

    tasks = data.get("tasks", [])
    if isinstance(tasks, list):
        stats["task_count"] = len(tasks)
        stats["model_task_count"] = sum(
            1
            for t in tasks
            if isinstance(t, dict)
            and t.get("type") != "gather"
            and ("slug" in t or "app" in t or "class" in t)
        )
        stats["gather_task_count"] = sum(
            1 for t in tasks if isinstance(t, dict) and t.get("type") == "gather"
        )
    else:
        stats["task_count"] = 0
        stats["model_task_count"] = 0
        stats["gather_task_count"] = 0

    outputs = data.get("outputs", [])
    stats["output_rule_count"] = len(outputs) if isinstance(outputs, list) else 0
    stats["protocol_name"] = data.get("name", "unknown")
    result.statistics = stats
