"""Local Protocol Profile v1 — supported-feature checks."""

from __future__ import annotations

import re
from typing import Any

PROFILE_DOC = "docs/guide/protocol-local-profile.rst"

_UNSUPPORTED_TASK_KEYS = frozenset(
    {"class", "app", "method", "gather", "foreach", "subtasks"}
)
_UNSUPPORTED_TASK_FIELDS = frozenset({"skip_if", "skip_if_empty"})
_TASK_OUTPUT_EXPR = re.compile(r"\btasks\.")


class UnsupportedProtocolFeature(ValueError):
    """Raised when a protocol uses features outside Local Protocol Profile v1."""

    def __init__(self, message: str, *, feature: str | None = None):
        self.feature = feature
        full = f"{message}\n\nSee {PROFILE_DOC} for supported features."
        super().__init__(full)


def _iter_tasks(protocol: dict) -> list[dict]:
    tasks = protocol.get("tasks") or []
    if not isinstance(tasks, list):
        raise UnsupportedProtocolFeature(
            "Protocol 'tasks' must be a list.", feature="tasks"
        )
    return tasks


def _walk_values(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_values(v)
    elif isinstance(obj, str):
        yield obj


def _check_expressions(protocol: dict) -> None:
    for value in _walk_values(protocol):
        if not isinstance(value, str):
            continue
        if value.startswith("${{") and _TASK_OUTPUT_EXPR.search(value):
            raise UnsupportedProtocolFeature(
                "Task-output expressions (tasks.<id>...) are not supported locally.",
                feature="task_output_expression",
            )


def _check_task(task: dict, index: int) -> None:
    for key in _UNSUPPORTED_TASK_KEYS:
        if key in task and task[key] is not None:
            raise UnsupportedProtocolFeature(
                f"Task feature '{key}' is not supported in Local Protocol Profile v1.",
                feature=key,
            )

    for field in _UNSUPPORTED_TASK_FIELDS:
        if field in task and task[field] is not None:
            raise UnsupportedProtocolFeature(
                f"Task field '{field}' is not supported in Local Protocol Profile v1.",
                feature=field,
            )

    task_type = task.get("type", "ApiTask")
    if task_type != "ApiTask":
        raise UnsupportedProtocolFeature(
            f"Task type '{task_type}' is not supported; only ApiTask is supported.",
            feature=task_type,
        )

    if not task.get("id"):
        raise UnsupportedProtocolFeature(
            f"Task at index {index} is missing required 'id'.",
            feature="missing_task_id",
        )

    if not task.get("slug") or not task.get("action"):
        raise UnsupportedProtocolFeature(
            f"Task '{task.get('id')}' must define both 'slug' and 'action'.",
            feature="api_task",
        )


def check_supported(protocol: dict) -> None:
    """Validate protocol dict against Local Protocol Profile v1.

    Raises:
        UnsupportedProtocolFeature: if any unsupported construct is present.
    """
    if not isinstance(protocol, dict):
        raise UnsupportedProtocolFeature("Protocol must be a dict.", feature="protocol")

    tasks = _iter_tasks(protocol)
    if not tasks:
        raise UnsupportedProtocolFeature(
            "Protocol must define at least one task.", feature="tasks"
        )

    _check_expressions(protocol)

    seen_ids: set[str] = set()
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise UnsupportedProtocolFeature(
                f"Task at index {i} must be a dict.", feature="tasks"
            )
        _check_task(task, i)
        task_id = task["id"]
        if task_id in seen_ids:
            raise UnsupportedProtocolFeature(
                f"Duplicate task id '{task_id}'.", feature="duplicate_task_id"
            )
        seen_ids.add(task_id)

    for task in tasks:
        for dep in task.get("depends_on") or []:
            if dep not in seen_ids:
                raise UnsupportedProtocolFeature(
                    f"Task '{task['id']}' depends on unknown task '{dep}'.",
                    feature="depends_on",
                )
