"""Compiled protocol execution plan dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompiledTask:
    """One ApiTask after compile-time resolution."""

    task_id: str
    slug: str
    action: str
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    response_mapping: dict[str, Any] = field(default_factory=dict)
    stage_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Compile output: resolved sequences + ordered tasks."""

    protocol_name: str
    sequences: list[str]
    tasks: list[CompiledTask]
    inputs: dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
