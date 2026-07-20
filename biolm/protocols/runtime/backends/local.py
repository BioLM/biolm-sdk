"""Local backend: run compiled pipeline and materialize results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from biolm.pipeline.data import DataPipeline

from biolm.protocols.runtime.compile import compile_to_pipeline
from biolm.protocols.runtime.results import dataframe_to_records
from biolm.protocols.runtime.spec import ExecutionPlan


@dataclass
class LocalRunResult:
    """Result of a local protocol run."""

    dataframe: pd.DataFrame
    records: list[dict[str, Any]]
    plan: ExecutionPlan
    pipeline: Optional[DataPipeline] = None
    run_id: Optional[str] = None

    def to_seqframe(self, *, molecule_type: Optional[str] = None, path: Optional[str | Path] = None):
        """Materialize results as a :class:`~biolm.seqframe.SeqFrame` (requires seqframe extra)."""
        try:
            from biolm.seqframe import SeqFrame
        except ImportError as exc:
            raise ImportError(
                "LocalRunResult.to_seqframe() requires the seqframe extra.\n\n"
                "Install with:\n\n"
                "    pip install 'biolm[seqframe]'\n"
            ) from exc
        return SeqFrame.from_dataframe(
            self.dataframe,
            molecule_type=molecule_type,
            path=path,
        )


def run_local(
    protocol: dict,
    inputs: dict[str, Any],
    *,
    output_dir: Optional[str | Path] = None,
    verbose: bool = False,
    **pipeline_kwargs: Any,
) -> LocalRunResult:
    """Compile and execute a protocol locally via DataPipeline."""
    plan, pipeline = compile_to_pipeline(
        protocol,
        inputs,
        output_dir=str(output_dir) if output_dir else None,
        verbose=verbose,
        **pipeline_kwargs,
    )

    pipeline.run()
    df = pipeline.get_final_data()
    records = dataframe_to_records(df)

    return LocalRunResult(
        dataframe=df,
        records=records,
        plan=plan,
        pipeline=pipeline,
        run_id=plan.run_id,
    )
