"""Light cross-attention-style resampler for stage-2 latent composition."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn

from core_mem.v2.schemas import SlotRecord


def _seeded_linear(in_features: int, out_features: int, *, seed: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        for row in range(out_features):
            for col in range(in_features):
                phase = (seed + 1) * (row + 1) * (col + 1)
                layer.weight[row, col] = math.sin(phase) * 0.1
            layer.bias[row] = math.cos((seed + 1) * (row + 1)) * 0.03
    return layer


def _normalize(vector: torch.Tensor) -> list[float]:
    norm = torch.linalg.vector_norm(vector).item()
    if norm <= 0.0:
        return [0.0 for _ in range(vector.numel())]
    return [float(value / norm) for value in vector.tolist()]


def _fit_width(values: list[float], width: int) -> torch.Tensor:
    payload = list(values[:width])
    if len(payload) < width:
        payload.extend([0.0] * (width - len(payload)))
    return torch.tensor(payload, dtype=torch.float32)


@dataclass(frozen=True)
class LightResampler:
    latent_queries: int = 8
    token_dim: int = 8
    seed: int = 43
    query_projection: nn.Linear = field(init=False, repr=False)
    token_projection: nn.Linear = field(init=False, repr=False)
    role_projection: nn.Linear = field(init=False, repr=False)
    latent_query_bank: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_projection", _seeded_linear(self.token_dim, self.token_dim, seed=self.seed))
        object.__setattr__(self, "token_projection", _seeded_linear(self.token_dim, self.token_dim, seed=self.seed + 1))
        object.__setattr__(self, "role_projection", _seeded_linear(6, self.token_dim, seed=self.seed + 2))
        bank = torch.zeros((self.latent_queries, self.token_dim), dtype=torch.float32)
        for row in range(self.latent_queries):
            for col in range(self.token_dim):
                bank[row, col] = math.sin((row + 1) * (col + 1) * (self.seed + 1)) * 0.2
        object.__setattr__(self, "latent_query_bank", bank)

    def compose(self, query_vector: list[float], slots: list[SlotRecord]) -> list[list[float]]:
        if not slots:
            return []
        query_tensor = _fit_width(query_vector, self.token_dim)
        base_query = torch.tanh(self.query_projection(query_tensor))
        enriched_tokens: list[torch.Tensor] = []
        for slot in slots:
            role_tensor = torch.tensor(
                [
                    slot.soft_role_scores.stable,
                    slot.soft_role_scores.preference,
                    slot.soft_role_scores.constraint,
                    slot.soft_role_scores.goal,
                    slot.soft_role_scores.temporal,
                    slot.soft_role_scores.social,
                ],
                dtype=torch.float32,
            )
            role_bias = self.role_projection(role_tensor)
            for token in slot.latent_tokens:
                token_tensor = _fit_width(token, self.token_dim)
                enriched_tokens.append(torch.tanh(self.token_projection(token_tensor) + role_bias))
        token_stack = torch.stack(enriched_tokens)
        composed_tokens: list[list[float]] = []
        for query_idx in range(self.latent_queries):
            latent_query = torch.tanh(base_query + self.latent_query_bank[query_idx])
            attention_logits = token_stack @ latent_query
            attention = torch.softmax(attention_logits, dim=0)
            composed = torch.sum(attention.unsqueeze(1) * token_stack, dim=0)
            composed_tokens.append(_normalize(composed))
        return composed_tokens
