"""SeqFrame Parquet metadata schema (version 0.1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SEQFRAME_VERSION = "0.1"
METADATA_VERSION_KEY = b"seqframe.version"
METADATA_SCHEMA_KEY = b"seqframe.schema"


@dataclass(frozen=True)
class SeqFrameMetadata:
    """Biological semantics and column mapping for a SeqFrame."""

    sequence_column: str = "sequence"
    id_column: str = "id"
    molecule_type: str = "unknown"
    alphabet: Optional[str] = None
    created_by: str = ""
    extensions: List[str] = field(default_factory=list)
    version: str = SEQFRAME_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("alphabet") is None:
            data.pop("alphabet", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeqFrameMetadata":
        return cls(
            sequence_column=data.get("sequence_column", "sequence"),
            id_column=data.get("id_column", "id"),
            molecule_type=data.get("molecule_type", "unknown"),
            alphabet=data.get("alphabet"),
            created_by=data.get("created_by", ""),
            extensions=list(data.get("extensions") or []),
            version=data.get("version", SEQFRAME_VERSION),
        )

    def to_parquet_metadata(self) -> Dict[bytes, bytes]:
        schema_json = json.dumps(self.to_dict(), sort_keys=True)
        return {
            METADATA_VERSION_KEY: self.version.encode("utf-8"),
            METADATA_SCHEMA_KEY: schema_json.encode("utf-8"),
        }

    @classmethod
    def from_parquet_metadata(cls, metadata: Dict[bytes, bytes]) -> "SeqFrameMetadata":
        if METADATA_VERSION_KEY not in metadata or METADATA_SCHEMA_KEY not in metadata:
            raise ValueError(
                "File is not a SeqFrame Parquet file (missing seqframe metadata). "
                "Use SeqFrame.from_fasta(), SeqFrame.from_csv(), or SeqFrame.from_jsonl() "
                "to import from other formats."
            )
        version = metadata[METADATA_VERSION_KEY].decode("utf-8")
        schema_raw = metadata[METADATA_SCHEMA_KEY].decode("utf-8")
        data = json.loads(schema_raw)
        data["version"] = version
        return cls.from_dict(data)
