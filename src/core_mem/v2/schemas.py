"""Structured schemas for stage-2 memory records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core_mem.v2.relations import normalize_relation

_SPEAKERS = {"user", "assistant", "system"}
_VALUE_TYPES = {"preference", "constraint", "profile", "goal", "event", "location", "occupation", "other"}
_TIME_SCOPES = {"current", "past", "future", "recent_change", "unknown"}
_STATUS_HINTS = {"active", "stale", "conflicted", "unknown"}
_POLARITIES = {"positive", "negative", "neutral"}
_BANKS = {"core", "residual"}
_QUERY_TYPES = {"single_fact", "multi_fact", "update_sensitive", "temporal"}
_GLOBAL_CONSISTENCY = {"high", "medium", "low"}
_FACET_SCOPES = {"entity", "style", "reason", "state", "temporal"}


def _require(choice: str, valid: set[str], field_name: str) -> str:
    if choice not in valid:
        raise ValueError(f"{field_name} must be one of {sorted(valid)}.")
    return choice


def _bounded_float(value: float, field_name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be within [0.0, 1.0].")
    return numeric


@dataclass(frozen=True)
class Observation:
    obs_id: str
    source_dataset: str
    source_dialogue_id: str
    source_turn_id: str
    session_id: str
    speaker: str
    entity: str
    relation: str
    value: str
    value_type: str
    time_scope: str
    status_hint: str
    polarity: str
    confidence: float
    evidence_text: str
    canonical_gloss: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.speaker, _SPEAKERS, "speaker")
        _require(self.value_type, _VALUE_TYPES, "value_type")
        _require(self.time_scope, _TIME_SCOPES, "time_scope")
        _require(self.status_hint, _STATUS_HINTS, "status_hint")
        _require(self.polarity, _POLARITIES, "polarity")
        object.__setattr__(self, "confidence", _bounded_float(self.confidence, "confidence"))
        object.__setattr__(self, "relation", normalize_relation(self.relation))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Observation:
        return cls(
            obs_id=str(payload["obs_id"]),
            source_dataset=str(payload.get("source_dataset", "")),
            source_dialogue_id=str(payload.get("source_dialogue_id", "")),
            source_turn_id=str(payload.get("source_turn_id", "")),
            session_id=str(payload.get("session_id", "")),
            speaker=str(payload.get("speaker", "user")),
            entity=str(payload.get("entity", "user")),
            relation=str(payload.get("relation", "other_fact")),
            value=str(payload.get("value", "")),
            value_type=str(payload.get("value_type", "other")),
            time_scope=str(payload.get("time_scope", "unknown")),
            status_hint=str(payload.get("status_hint", "unknown")),
            polarity=str(payload.get("polarity", "neutral")),
            confidence=float(payload.get("confidence", 0.0)),
            evidence_text=str(payload.get("evidence_text", "")),
            canonical_gloss=str(payload.get("canonical_gloss", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class FacetRecord:
    facet_id: str
    source_observation_id: str
    relation: str
    facet_type: str
    facet_value: str
    facet_scope: str
    facet_polarity: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", normalize_relation(self.relation))
        _require(self.facet_scope, _FACET_SCOPES, "facet_scope")
        _require(self.facet_polarity, _POLARITIES, "facet_polarity")
        object.__setattr__(self, "confidence", _bounded_float(self.confidence, "confidence"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FacetRecord:
        return cls(
            facet_id=str(payload["facet_id"]),
            source_observation_id=str(payload.get("source_observation_id", "")),
            relation=str(payload.get("relation", "other_fact")),
            facet_type=str(payload.get("facet_type", "generic_fact")),
            facet_value=str(payload.get("facet_value", "")),
            facet_scope=str(payload.get("facet_scope", "entity")),
            facet_polarity=str(payload.get("facet_polarity", "neutral")),
            confidence=float(payload.get("confidence", 0.0)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class SoftRoleScores:
    stable: float = 0.0
    preference: float = 0.0
    constraint: float = 0.0
    goal: float = 0.0
    temporal: float = 0.0
    social: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("stable", "preference", "constraint", "goal", "temporal", "social"):
            object.__setattr__(self, field_name, _bounded_float(getattr(self, field_name), field_name))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SoftRoleScores:
        return cls(**{key: float(payload.get(key, 0.0)) for key in cls().__dict__})


@dataclass(frozen=True)
class SlotRecord:
    slot_id: str
    bank: str
    entity: str
    relation: str
    retrieval_key: list[float]
    latent_tokens: list[list[float]]
    soft_role_scores: SoftRoleScores
    confidence: float
    first_seen_ts: str
    last_update_ts: str
    revision_count: int
    active_flag: bool
    revision_parent: str | None
    canonical_gloss: str

    def __post_init__(self) -> None:
        _require(self.bank, _BANKS, "bank")
        object.__setattr__(self, "confidence", _bounded_float(self.confidence, "confidence"))
        object.__setattr__(self, "relation", normalize_relation(self.relation))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["soft_role_scores"] = self.soft_role_scores.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SlotRecord:
        return cls(
            slot_id=str(payload["slot_id"]),
            bank=str(payload.get("bank", "residual")),
            entity=str(payload.get("entity", "user")),
            relation=str(payload.get("relation", "other_fact")),
            retrieval_key=[float(value) for value in payload.get("retrieval_key", [])],
            latent_tokens=[
                [float(item) for item in token_row]
                for token_row in payload.get("latent_tokens", [])
            ],
            soft_role_scores=SoftRoleScores.from_dict(payload.get("soft_role_scores", {})),
            confidence=float(payload.get("confidence", 0.0)),
            first_seen_ts=str(payload.get("first_seen_ts", "")),
            last_update_ts=str(payload.get("last_update_ts", "")),
            revision_count=int(payload.get("revision_count", 0)),
            active_flag=bool(payload.get("active_flag", True)),
            revision_parent=str(payload["revision_parent"]) if payload.get("revision_parent") is not None else None,
            canonical_gloss=str(payload.get("canonical_gloss", "")),
        )


@dataclass(frozen=True)
class BeliefItem:
    relation: str
    value: str
    status: str
    time_scope: str
    confidence: float
    support_slot_ids: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", normalize_relation(self.relation))
        _require(self.status, _STATUS_HINTS, "status")
        _require(self.time_scope, _TIME_SCOPES, "time_scope")
        object.__setattr__(self, "confidence", _bounded_float(self.confidence, "confidence"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BeliefItem:
        return cls(
            relation=str(payload.get("relation", "other_fact")),
            value=str(payload.get("value", "")),
            status=str(payload.get("status", "unknown")),
            time_scope=str(payload.get("time_scope", "unknown")),
            confidence=float(payload.get("confidence", 0.0)),
            support_slot_ids=[str(value) for value in payload.get("support_slot_ids", [])],
        )


@dataclass(frozen=True)
class BeliefState:
    query_id: str
    entity: str
    query_type: str
    belief_items: list[BeliefItem]
    global_consistency: str

    def __post_init__(self) -> None:
        _require(self.query_type, _QUERY_TYPES, "query_type")
        _require(self.global_consistency, _GLOBAL_CONSISTENCY, "global_consistency")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "entity": self.entity,
            "query_type": self.query_type,
            "belief_items": [item.to_dict() for item in self.belief_items],
            "global_consistency": self.global_consistency,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BeliefState:
        return cls(
            query_id=str(payload.get("query_id", "")),
            entity=str(payload.get("entity", "user")),
            query_type=str(payload.get("query_type", "single_fact")),
            belief_items=[BeliefItem.from_dict(item) for item in payload.get("belief_items", [])],
            global_consistency=str(payload.get("global_consistency", "medium")),
        )
