"""Core memory primitives for the vanilla CoRe Memory skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResidualSlot:
    latent: list[float]
    timestamp: str
    salience: float
    support: list[str] = field(default_factory=list)


@dataclass
class CoreMemoryState:
    core_vector: list[float]
    residual_slots: list[ResidualSlot] = field(default_factory=list)

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

