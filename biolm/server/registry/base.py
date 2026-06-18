"""Model deployment registry types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol


class ModelStatus(str, Enum):
    READY = "ready"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


@dataclass
class ModelEntry:
    slug: str
    base_url: str
    status: ModelStatus = ModelStatus.UNKNOWN
    source: str = "config"
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "base_url": self.base_url,
            "status": self.status.value,
            "source": self.source,
            "actions": self.actions,
        }


class ModelRegistry(Protocol):
    async def refresh(self) -> None: ...

    def get(self, slug: str) -> Optional[ModelEntry]: ...

    def list(self) -> List[ModelEntry]: ...

    def snapshot(self) -> Dict[str, ModelEntry]: ...
