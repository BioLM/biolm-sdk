"""sf.lab — Lab-in-the-Loop (LLTP) bridge (convert / merge only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame


class LabNamespace:
    """Convert SeqFrame rows ↔ connector order payloads / LLTP result datasets.

    Orchestration (submit / status / results) lives in :mod:`biolm.lab`.
    """

    def __init__(self, sf: "SeqFrame"):
        self._sf = sf

    def to_lltp(
        self,
        *,
        service_id: str,
        name: Optional[str] = None,
        sequence_column: str = "sequence",
        id_column: str = "id",
        name_column: Optional[str] = "name",
        **extras: Any,
    ) -> Dict[str, Any]:
        """Build a connector ``order_payload`` from this SeqFrame.

        Each row becomes a sequence entry with ``id`` = SeqFrame id (round-trips
        to ``entity.entity_id`` on results). Extra kwargs are merged into the
        payload (e.g. ``wait_for_scoring``, ``n_replicates``, payment ids).
        """
        df = self._sf.collect()
        if id_column not in df.columns:
            raise ValueError(
                f"SeqFrame missing id column {id_column!r}; "
                f"columns={list(df.columns)}"
            )
        if sequence_column not in df.columns:
            raise ValueError(
                f"SeqFrame missing sequence column {sequence_column!r}; "
                f"columns={list(df.columns)}"
            )

        sequences: List[Dict[str, str]] = []
        use_name = name_column and name_column in df.columns
        for _, row in df.iterrows():
            entry: Dict[str, str] = {
                "id": str(row[id_column]),
                "sequence": str(row[sequence_column]),
            }
            if use_name and row[name_column] is not None:
                entry["name"] = str(row[name_column])
            else:
                entry["name"] = entry["id"]
            sequences.append(entry)

        payload: Dict[str, Any] = {
            "service_id": service_id,
            "sequences": sequences,
        }
        if name is not None:
            payload["name"] = name
        payload.update(extras)
        return payload

    @classmethod
    def from_lltp(
        cls,
        dataset: Mapping[str, Any],
        *,
        molecule_type: Optional[str] = None,
        path: Optional[Union[str, Any]] = None,
    ) -> "SeqFrame":
        """Build a SeqFrame from a connector ``to_lltp_result`` dataset dict.

        Sets ``id`` from ``entity.entity_id`` so :meth:`merge` can join on ``id``.
        """
        from biolm.seqframe.core import SeqFrame

        records = dataset.get("records") or []
        rows: List[Dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            entity = rec.get("entity") or {}
            entity_id = str(
                entity.get("entity_id")
                or (rec.get("tags") or {}).get("construct_name")
                or ""
            )
            representation = entity.get("representation")
            seq = ""
            if isinstance(representation, str) and representation != entity_id:
                # Heuristic: raw sequence representations are usable as sequence
                seq = representation
            row: Dict[str, Any] = {
                "id": entity_id or f"row_{len(rows)}",
                "sequence": seq or entity_id or "X",
            }
            tags = rec.get("tags") or {}
            metrics = rec.get("metrics") or {}
            params = rec.get("parameters") or {}
            for key, value in tags.items():
                row[f"tag_{key}"] = value
            for key, value in metrics.items():
                row[f"metric_{key}"] = value
            for key, value in params.items():
                row[f"param_{key}"] = value
            if entity.get("type"):
                row["entity_type"] = entity["type"]
            rows.append(row)

        if not rows:
            raise ValueError(
                "LLTP dataset has no records; cannot build SeqFrame"
            )

        mt = molecule_type
        if mt is None:
            types = {r.get("entity_type") for r in rows}
            if "dna_sequence" in types:
                mt = "dna"
            elif "protein" in types or "aa_sequence" in types:
                mt = "protein"
            else:
                mt = "unknown"

        return SeqFrame.from_rows(rows, molecule_type=mt, path=path)

    def merge(
        self,
        other: "SeqFrame",
        *,
        on: str = "id",
        how: str = "left",
    ) -> "SeqFrame":
        """Join lab results onto this SeqFrame (default ``on='id'``)."""
        return self._sf.query.join(other, on=on, how=how)
