"""MemorySlot data structure and vector operations shared by Core and Residual."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MemorySlot:
    """A single memory slot used by both core and residual stores."""

    embedding: np.ndarray
    merge_count: int = 1
    timestamp: float = field(default_factory=time.time)
    salience: float = 1.0
    provenance: str = ""
    _recent_deltas: deque = field(default_factory=lambda: deque(maxlen=10))

    def __post_init__(self) -> None:
        self.embedding = _l2_normalize(np.asarray(self.embedding, dtype=np.float32))

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

    def record_change(self, magnitude: float) -> None:
        """Append the L2 magnitude of the latest merge change."""
        self._recent_deltas.append(magnitude)

    def is_stable(self, window: int = 3, epsilon: float = 0.05) -> bool:
        """Return True if the last *window* merges each changed < *epsilon*."""
        if len(self._recent_deltas) < window:
            return False
        return all(d < epsilon for d in list(self._recent_deltas)[-window:])

    def to_dict(self) -> dict:
        """Serialize metadata (excluding embedding) to a JSON-friendly dict."""
        return {
            "merge_count": self.merge_count,
            "timestamp": self.timestamp,
            "salience": self.salience,
            "provenance": self.provenance,
            "recent_deltas": list(self._recent_deltas),
        }

    @classmethod
    def from_dict(cls, meta: dict, embedding: np.ndarray) -> MemorySlot:
        """Reconstruct a slot from persisted metadata + embedding."""
        slot = cls(
            embedding=embedding,
            merge_count=meta["merge_count"],
            timestamp=meta["timestamp"],
            salience=meta["salience"],
            provenance=meta.get("provenance", ""),
        )
        for d in meta.get("recent_deltas", []):
            slot._recent_deltas.append(d)
        return slot


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def online_centroid_merge(
    slot: MemorySlot,
    incoming: np.ndarray,
    recency_weight: float = 1.5,
) -> float:
    """Merge *incoming* into *slot* using the online centroid formula.

    Returns the L2 magnitude of the change (before re-normalization).
    """
    n = slot.merge_count
    old = slot.embedding.copy()
    combined = (n * old + recency_weight * incoming) / (n + recency_weight)
    new_emb = _l2_normalize(combined)

    change_magnitude = float(np.linalg.norm(new_emb - old))

    slot.embedding = new_emb
    slot.merge_count += 1
    slot.timestamp = time.time()
    slot.record_change(change_magnitude)
    return change_magnitude
