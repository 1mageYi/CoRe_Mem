"""Slot/query encoders for the stage-2 latent-memory path."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re

import torch
from torch import nn

from core_mem.v2.schemas import Observation, SlotRecord, SoftRoleScores

_FEATURE_DIM = 12


def _seeded_linear(in_features: int, out_features: int, *, seed: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        for row in range(out_features):
            for col in range(in_features):
                phase = (seed + 1) * (row + 1) * (col + 1)
                layer.weight[row, col] = math.sin(phase) * 0.12
            layer.bias[row] = math.cos((seed + 1) * (row + 1)) * 0.05
    return layer


def _normalize(vector: torch.Tensor) -> list[float]:
    norm = torch.linalg.vector_norm(vector).item()
    if norm <= 0.0:
        return [0.0 for _ in range(vector.numel())]
    return [float(value / norm) for value in vector.tolist()]


def _ratios(text: str) -> tuple[float, float, float]:
    length = max(len(text), 1)
    digits = sum(char.isdigit() for char in text) / length
    punctuation = sum(not char.isalnum() and not char.isspace() for char in text) / length
    uppercase = sum(char.isupper() for char in text) / length
    return digits, punctuation, uppercase


def _lexical_features(text: str) -> list[float]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9']+", lowered)
    token_count = len(tokens)
    unique_tokens = len(set(tokens))
    char_count = len(lowered)
    avg_token_length = sum(len(token) for token in tokens) / token_count if token_count else 0.0
    digits_ratio, punctuation_ratio, uppercase_ratio = _ratios(text)
    vocab = {
        "preference": float(any(token in {"like", "love", "prefer", "favorite"} for token in tokens)),
        "update": float(any(token in {"now", "currently", "changed", "instead"} for token in tokens)),
        "temporal": float(any(token in {"before", "after", "future", "past", "when"} for token in tokens)),
        "negation": float(any(token in {"not", "never", "don't", "no"} for token in tokens)),
        "question": float("?" in text or any(token in {"what", "which", "who", "where", "when"} for token in tokens)),
    }
    return [
        min(char_count / 128.0, 1.0),
        min(token_count / 24.0, 1.0),
        min(unique_tokens / 24.0, 1.0),
        min(avg_token_length / 12.0, 1.0),
        digits_ratio,
        punctuation_ratio,
        uppercase_ratio,
        vocab["preference"],
        vocab["update"],
        vocab["temporal"],
        vocab["negation"],
        vocab["question"],
    ]


def _observation_features(observation: Observation) -> list[float]:
    relation_tokens = observation.relation.replace("_", " ")
    value_tokens = observation.value
    combined = f"{relation_tokens} {value_tokens} {observation.canonical_gloss}"
    features = _lexical_features(combined)
    features[7] = max(features[7], float(observation.value_type == "preference" or observation.relation.endswith("preference")))
    features[8] = max(features[8], float(observation.time_scope == "recent_change"))
    features[9] = max(features[9], float(observation.time_scope in {"past", "future", "recent_change"}))
    features[10] = max(features[10], float(observation.polarity == "negative"))
    features[11] = max(features[11], float(observation.status_hint in {"active", "conflicted"}))
    return features


def _token_features(base: list[float], token_text: str, token_idx: int, total_tokens: int) -> list[float]:
    token_features = _lexical_features(token_text)
    return [
        (left + right) / 2.0
        for left, right in zip(
            base,
            [
                token_features[0],
                token_features[1],
                token_features[2],
                min((token_idx + 1) / max(total_tokens, 1), 1.0),
                token_features[4],
                token_features[5],
                token_features[6],
                token_features[7],
                token_features[8],
                token_features[9],
                token_features[10],
                token_features[11],
            ],
        )
    ]


@dataclass(frozen=True)
class SlotEncoderConfig:
    retrieval_dim: int = 8
    slot_tokens: int = 8
    token_dim: int = 8


@dataclass(frozen=True)
class QueryEncoder:
    dimension: int = 8
    feature_dim: int = _FEATURE_DIM
    seed: int = 17
    projection: nn.Linear = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "projection", _seeded_linear(self.feature_dim, self.dimension, seed=self.seed))

    def encode(self, text: str) -> list[float]:
        features = torch.tensor(_lexical_features(text), dtype=torch.float32)
        encoded = torch.tanh(self.projection(features))
        return _normalize(encoded)


@dataclass(frozen=True)
class SlotEncoder:
    config: SlotEncoderConfig = SlotEncoderConfig()
    feature_dim: int = _FEATURE_DIM
    retrieval_seed: int = 29
    token_seed: int = 31
    retrieval_projection: nn.Linear = field(init=False, repr=False)
    token_projection: nn.Linear = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "retrieval_projection",
            _seeded_linear(self.feature_dim, self.config.retrieval_dim, seed=self.retrieval_seed),
        )
        object.__setattr__(
            self,
            "token_projection",
            _seeded_linear(self.feature_dim, self.config.token_dim, seed=self.token_seed),
        )

    def encode(
        self,
        observation: Observation,
        *,
        timestamp: str,
        existing_slot: SlotRecord | None = None,
        bank: str = "residual",
        revision_parent: str | None = None,
    ) -> SlotRecord:
        base_features = _observation_features(observation)
        retrieval_input = torch.tensor(base_features, dtype=torch.float32)
        retrieval_key = _normalize(torch.tanh(self.retrieval_projection(retrieval_input)))
        token_texts = re.findall(r"[a-z0-9']+", observation.value.lower()) or re.findall(
            r"[a-z0-9']+",
            observation.canonical_gloss.lower(),
        )
        if not token_texts:
            token_texts = [observation.relation.replace("_", " ")]
        latent_tokens = [
            _normalize(
                torch.tanh(
                    self.token_projection(
                        torch.tensor(
                            _token_features(base_features, token_texts[token_idx % len(token_texts)], token_idx, self.config.slot_tokens),
                            dtype=torch.float32,
                        )
                    )
                )
            )
            for token_idx in range(self.config.slot_tokens)
        ]
        digest = hashlib.sha1(f"{observation.obs_id}|{bank}|{timestamp}".encode("utf-8")).hexdigest()[:12]
        return SlotRecord(
            slot_id=f"slot_{digest}",
            bank=bank,
            entity=observation.entity,
            relation=observation.relation,
            retrieval_key=retrieval_key,
            latent_tokens=latent_tokens,
            soft_role_scores=self._soft_roles(observation),
            confidence=observation.confidence,
            first_seen_ts=existing_slot.first_seen_ts if existing_slot else timestamp,
            last_update_ts=timestamp,
            revision_count=(existing_slot.revision_count + 1) if existing_slot else 0,
            active_flag=True,
            revision_parent=revision_parent,
            canonical_gloss=observation.canonical_gloss,
        )

    @staticmethod
    def _soft_roles(observation: Observation) -> SoftRoleScores:
        relation = observation.relation
        return SoftRoleScores(
            stable=0.85 if observation.time_scope == "current" and observation.status_hint == "active" else 0.35,
            preference=0.95 if relation.endswith("preference") or relation == "hobby" else 0.05,
            constraint=0.95 if relation == "constraint" or observation.value_type == "constraint" else 0.0,
            goal=0.95 if relation == "goal" or observation.value_type == "goal" else 0.0,
            temporal=0.9 if observation.time_scope in {"past", "future", "recent_change"} else 0.1,
            social=0.9 if relation == "relationship" else 0.0,
        )
