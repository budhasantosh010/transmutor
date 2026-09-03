from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PrimitiveRecord:
    primitive_id: str
    created_from: list[str]
    embedding: list[float]
    structural_signature: dict[str, Any]
    dynamic_signature: dict[str, Any]
    input_signature: dict[str, Any]
    output_signature: dict[str, Any]
    internal_cost: dict[str, Any]
    discovery_cost: dict[str, Any]
    source_motif_hash: str
    internal_graph: dict[str, Any]
    state_dim: int
    message_dim: int
    obs_dim: int
    state_path: str
    abstraction_depth: int = 1
    usage_history: list[dict[str, Any]] = field(default_factory=list)
    success_history: list[dict[str, Any]] = field(default_factory=list)
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True


class PrimitiveArchive:
    def __init__(self, records: list[PrimitiveRecord] | None = None):
        self.records: dict[str, PrimitiveRecord] = {record.primitive_id: record for record in (records or [])}

    def add(self, record: PrimitiveRecord) -> None:
        if record.primitive_id in self.records:
            raise ValueError(f"duplicate primitive {record.primitive_id}")
        self.records[record.primitive_id] = record

    def remove(self, primitive_id: str) -> None:
        self.records[primitive_id].active = False

    def list(self, *, active_only: bool = False) -> list[PrimitiveRecord]:
        values = sorted(self.records.values(), key=lambda record: record.primitive_id)
        return [record for record in values if record.active] if active_only else values

    def retrieve(self, query_embedding: list[float], top_k: int = 3) -> list[PrimitiveRecord]:
        query = np.asarray(query_embedding, dtype=float)
        scored = []
        for record in self.list(active_only=True):
            embedding = np.asarray(record.embedding, dtype=float)
            if embedding.shape != query.shape:
                continue
            denom = np.linalg.norm(query) * np.linalg.norm(embedding)
            score = float(np.dot(query, embedding) / denom) if denom > 1e-12 else 0.0
            scored.append((score, record.primitive_id, record))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [record for _, _, record in scored[:top_k]]

    def increment_usage(self, primitive_id: str, event: dict[str, Any]) -> None:
        self.records[primitive_id].usage_history.append(dict(event))

    def record_success(self, primitive_id: str, event: dict[str, Any]) -> None:
        self.records[primitive_id].success_history.append(dict(event))

    def record_failure(self, primitive_id: str, event: dict[str, Any]) -> None:
        self.records[primitive_id].failure_history.append(dict(event))

    def prune_simulation(self, *, max_recent_events: int = 20) -> list[str]:
        inactive = []
        for record in self.list(active_only=True):
            recent = record.success_history[-max_recent_events:]
            if len(record.usage_history) < 2 and not recent and record.internal_cost.get("cells", 0) > 0:
                inactive.append(record.primitive_id)
        return inactive

    def serialize(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(record) for record in self.list()], indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def deserialize(cls, path: str | Path) -> "PrimitiveArchive":
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([PrimitiveRecord(**row) for row in rows])

    def summary(self) -> dict[str, Any]:
        return {
            "archive_size": len(self.records),
            "active_size": sum(1 for record in self.records.values() if record.active),
            "primitive_ids": sorted(self.records),
            "max_abstraction_depth": max((record.abstraction_depth for record in self.records.values()), default=0),
        }
