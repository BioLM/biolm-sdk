"""Protocol dict → ExecutionPlan → DataPipeline."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from biolm.pipeline.data import DataPipeline, PredictionStage

from biolm.protocols.runtime.context import InputContext
from biolm.protocols.runtime.expressions import evaluate_tree
from biolm.protocols.runtime.mapping import mapping_to_stage_kwargs
from biolm.protocols.runtime.profile import check_supported
from biolm.protocols.runtime.spec import CompiledTask, ExecutionPlan


def _topo_sort_tasks(tasks: list[dict]) -> list[dict]:
    """Order tasks respecting depends_on (stable for independent tasks)."""
    by_id = {t["id"]: t for t in tasks}
    in_degree = {t["id"]: len(t.get("depends_on") or []) for t in tasks}
    dependents: dict[str, list[str]] = {t["id"]: [] for t in tasks}
    for t in tasks:
        for dep in t.get("depends_on") or []:
            dependents[dep].append(t["id"])

    queue = [t for t in tasks if in_degree[t["id"]] == 0]
    ordered: list[dict] = []
    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for child_id in dependents[current["id"]]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(by_id[child_id])

    if len(ordered) != len(tasks):
        raise ValueError("Circular task dependency detected.")
    return ordered


def build_execution_plan(
    protocol: dict,
    inputs: dict[str, Any],
    *,
    run_id: Optional[str] = None,
) -> ExecutionPlan:
    """Compile protocol + inputs to an ExecutionPlan (no pipeline construction)."""
    check_supported(protocol)
    ctx = InputContext(inputs, protocol)
    eval_ctx = ctx.as_eval_context()
    sequences = ctx.resolve_sequences()

    compiled: list[CompiledTask] = []
    for task in _topo_sort_tasks(protocol.get("tasks") or []):
        request_body = task.get("request_body") or {}
        params = evaluate_tree(request_body.get("params") or {}, eval_ctx)

        stage_kwargs = mapping_to_stage_kwargs(
            task["action"],
            task.get("response_mapping") or {},
        )

        compiled.append(
            CompiledTask(
                task_id=task["id"],
                slug=task["slug"],
                action=task["action"],
                depends_on=list(task.get("depends_on") or []),
                params=params,
                response_mapping=dict(task.get("response_mapping") or {}),
                stage_kwargs=stage_kwargs,
            )
        )

    return ExecutionPlan(
        protocol_name=protocol.get("name") or protocol.get("id") or "protocol",
        sequences=sequences,
        tasks=compiled,
        inputs=dict(ctx.inputs),
        run_id=run_id or str(uuid.uuid4())[:8],
    )


def compile_to_pipeline(
    protocol: dict,
    inputs: dict[str, Any],
    *,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    verbose: bool = False,
    **pipeline_kwargs: Any,
) -> tuple[ExecutionPlan, DataPipeline]:
    """Compile protocol and build a configured DataPipeline."""
    plan = build_execution_plan(protocol, inputs, run_id=run_id)

    ds_kwargs: dict[str, Any] = dict(pipeline_kwargs)
    if output_dir is not None:
        ds_kwargs["output_dir"] = str(output_dir)

    pipeline = DataPipeline(
        sequences=plan.sequences,
        run_id=plan.run_id,
        verbose=verbose,
        **ds_kwargs,
    )

    for task in plan.tasks:
        stage = PredictionStage(
            name=task.task_id,
            model_name=task.slug,
            action=task.action,
            params=task.params,
            depends_on=task.depends_on or None,
            **task.stage_kwargs,
        )
        pipeline.add_stage(stage)

    return plan, pipeline


# Public alias
compile_protocol = compile_to_pipeline
