"""Persist LLTP lab runs under ``.biolm/lltp/<run_id>.json``."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_RUNS_DIR = Path(".biolm") / "lltp"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


@dataclass
class LabRun:
    run_id: str
    connector: str
    service_id: str
    status: str = "submitted"
    experiment: Optional[str] = None
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    handle: Dict[str, Any] = field(default_factory=dict)
    last_status: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabRun":
        return cls(
            run_id=str(data["run_id"]),
            connector=str(data["connector"]),
            service_id=str(data["service_id"]),
            status=str(data.get("status") or "submitted"),
            experiment=data.get("experiment"),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
            handle=dict(data.get("handle") or {}),
            last_status=data.get("last_status"),
            result=data.get("result"),
            extras=dict(data.get("extras") or {}),
        )


def runs_dir(root: Optional[Union[str, Path]] = None) -> Path:
    base = Path(root) if root is not None else Path.cwd()
    return (base / DEFAULT_RUNS_DIR).resolve()


def run_path(run_id: str, *, root: Optional[Union[str, Path]] = None) -> Path:
    return runs_dir(root) / f"{run_id}.json"


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_run(run: LabRun, *, root: Optional[Union[str, Path]] = None) -> Path:
    run.updated_at = _utcnow()
    path = run_path(run.run_id, root=root)
    _atomic_write(path, run.to_dict())
    return path


def load_run(run_id: str, *, root: Optional[Union[str, Path]] = None) -> LabRun:
    path = run_path(run_id, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"Lab run not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return LabRun.from_dict(data)


def list_run_ids(*, root: Optional[Union[str, Path]] = None) -> List[str]:
    directory = runs_dir(root)
    if not directory.is_dir():
        return []
    ids = [p.stem for p in directory.glob("*.json") if p.is_file()]
    return sorted(ids)


def list_runs(*, root: Optional[Union[str, Path]] = None) -> List[LabRun]:
    return [load_run(rid, root=root) for rid in list_run_ids(root=root)]
