"""Minimal vanilla CoRe Memory workflow primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core_mem.embeddings.base import TextEmbedder

def _pad_vector(values: list[float], size: int) -> list[float]:
    if len(values) >= size:
        return list(values[:size])
    return list(values) + [0.0] * (size - len(values))


def _match_dimensions(left: list[float], right: list[float]) -> tuple[list[float], list[float]]:
    size = max(len(left), len(right))
    return _pad_vector(left, size), _pad_vector(right, size)


def _dot_product(left: list[float], right: list[float]) -> float:
    matched_left, matched_right = _match_dimensions(left, right)
    return sum(lhs * rhs for lhs, rhs in zip(matched_left, matched_right))


@dataclass(frozen=True)
class ResidualSlot:
    latent: list[float]
    timestamp: str
    salience: float
    support: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryObservation:
    latent: list[float]
    timestamp: str
    salience: float = 1.0
    support: list[str] = field(default_factory=list)


@dataclass
class CoreMemoryState:
    core_vector: list[float]
    residual_slots: list[ResidualSlot] = field(default_factory=list)

    @classmethod
    def empty(cls, dimension: int = 0) -> CoreMemoryState:
        return cls(core_vector=[0.0] * max(dimension, 0))

    def residual_count(self) -> int:
        return len(self.residual_slots)

    def as_dict(self) -> dict[str, Any]:
        return {
            "core_vector": self.core_vector,
            "residual_slots": [
                {
                    "latent": slot.latent,
                    "timestamp": slot.timestamp,
                    "salience": slot.salience,
                    "support": slot.support,
                }
                for slot in self.residual_slots
            ],
        }


@dataclass(frozen=True)
class MemoryReadResult:
    core_vector: list[float]
    supporting_slots: list[ResidualSlot]


@dataclass(frozen=True)
class CoreMemoryUpdater:
    learning_rate: float = 0.5

    def update(self, current: list[float], incoming: list[float]) -> list[float]:
        matched_current, matched_incoming = _match_dimensions(current, incoming)
        if not matched_current:
            return list(matched_incoming)
        return [
            (1.0 - self.learning_rate) * current_value + self.learning_rate * incoming_value
            for current_value, incoming_value in zip(matched_current, matched_incoming)
        ]


@dataclass(frozen=True)
class ResidualMemoryManager:
    max_slots: int = 4

    def insert(self, slots: list[ResidualSlot], slot: ResidualSlot) -> list[ResidualSlot]:
        ranked = list(slots) + [slot]
        ranked.sort(key=lambda item: (item.salience, item.timestamp), reverse=True)
        return ranked[: self.max_slots]


@dataclass(frozen=True)
class MemoryWriter:
    updater: CoreMemoryUpdater = field(default_factory=CoreMemoryUpdater)
    residual_manager: ResidualMemoryManager = field(default_factory=ResidualMemoryManager)
    residual_salience_floor: float = 0.0

    def write(self, state: CoreMemoryState, observation: MemoryObservation) -> CoreMemoryState:
        next_core = self.updater.update(state.core_vector, observation.latent)
        next_slots = list(state.residual_slots)
        if observation.salience >= self.residual_salience_floor:
            next_slots = self.residual_manager.insert(
                next_slots,
                ResidualSlot(
                    latent=list(observation.latent),
                    timestamp=observation.timestamp,
                    salience=observation.salience,
                    support=list(observation.support),
                ),
            )
        return CoreMemoryState(core_vector=next_core, residual_slots=next_slots)


@dataclass(frozen=True)
class MemoryReader:
    top_k: int = 3

    def read(self, state: CoreMemoryState, query_latent: list[float]) -> MemoryReadResult:
        ranked = sorted(
            state.residual_slots,
            key=lambda slot: (_dot_product(query_latent, slot.latent), slot.salience, slot.timestamp),
            reverse=True,
        )
        return MemoryReadResult(core_vector=list(state.core_vector), supporting_slots=ranked[: self.top_k])


@dataclass
class VanillaCoreMemory:
    writer: MemoryWriter = field(default_factory=MemoryWriter)
    reader: MemoryReader = field(default_factory=MemoryReader)
    embedder: TextEmbedder | None = None
    state: CoreMemoryState = field(default_factory=CoreMemoryState.empty)

    def observe(self, observation: MemoryObservation) -> CoreMemoryState:
        self.state = self.writer.write(self.state, observation)
        return self.state

    def query(self, query_latent: list[float]) -> MemoryReadResult:
        return self.reader.read(self.state, query_latent)

    def observe_text(
        self,
        text: str,
        *,
        timestamp: str,
        salience: float = 1.0,
        support: list[str] | None = None,
    ) -> CoreMemoryState:
        latent = self._embed_text(text)
        return self.observe(
            MemoryObservation(
                latent=latent,
                timestamp=timestamp,
                salience=salience,
                support=list(support or []),
            )
        )

    def query_text(self, text: str) -> MemoryReadResult:
        latent = self._embed_text(text)
        return self.query(latent)

    def _embed_text(self, text: str) -> list[float]:
        if self.embedder is None:
            raise RuntimeError("Text embedding requires an embedder to be configured.")
        return self.embedder.embed(text)
