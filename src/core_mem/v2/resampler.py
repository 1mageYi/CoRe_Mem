"""Light cross-attention-style resampler skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from core_mem.v2.schemas import SlotRecord
from core_mem.v2.vector_ops import dot_product, mean_vectors


@dataclass(frozen=True)
class LightResampler:
    latent_queries: int = 8

    def compose(self, query_vector: list[float], slots: list[SlotRecord]) -> list[list[float]]:
        if not slots:
            return []
        weighted_tokens: list[list[float]] = []
        for slot in slots:
            weight = max(dot_product(query_vector, slot.retrieval_key), 0.0) + 1.0
            for token in slot.latent_tokens:
                weighted_tokens.append([value * weight for value in token])
        anchor = mean_vectors(weighted_tokens)
        return [[value * ((idx + 1) / self.latent_queries) for value in anchor] for idx in range(self.latent_queries)]
