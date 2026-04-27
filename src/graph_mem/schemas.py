from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EventType = Literal["emergence", "supplement", "task", "unknown"]
UpdateType = Literal["new", "update", "revision", "conflict", "unknown"]
NodeState = Literal["active", "inactive", "superseded", "conflicted"]
EdgeType = Literal["semantic", "temporal", "co_usage", "supersedes"]


@dataclass(slots=True)
class StructuredRecord:
    text: str
    event_type: EventType = "unknown"
    topic: str = "general"
    preference_slot: str = "other"
    value: str = ""
    update_type: UpdateType = "unknown"
    polarity: Literal["positive", "negative", "uncertain"] = "uncertain"
    time_index: int = 0
    condition_tag: str = ""
    evidence_span: str = ""
    source_turn_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class MemoryNode:
    node_id: str
    structured_record: StructuredRecord
    structured_text: str
    embedding: list[float]
    created_at: int
    updated_at: int
    state: NodeState = "active"
    valid_from: int = 0
    valid_to: int | None = None
    supersedes: str | None = None
    version: int = 1
    retrieve_count: int = 0
    core_score: float = 0.0


@dataclass(slots=True)
class MemoryEdge:
    src_node_id: str
    dst_node_id: str
    edge_type: EdgeType
    weight: float
    created_at: int
    updated_at: int
    usage_count: int = 1
