"""Encoder comparison harness for the v5 latent-substrate line."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

import numpy as np

from core_mem.v2.encoder import _lexical_features


@dataclass(frozen=True)
class EncoderCandidate:
    name: str
    model_id: str
    family: str


V5_ENCODER_CANDIDATES: tuple[EncoderCandidate, ...] = (
    EncoderCandidate(name="bge", model_id="BAAI/bge-base-en-v1.5", family="bge"),
    EncoderCandidate(name="e5", model_id="intfloat/e5-base-v2", family="e5"),
    EncoderCandidate(name="contriever", model_id="facebook/contriever", family="contriever"),
)


def render_context_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{row.get('role', 'unknown')}: {row.get('content', '')}".strip() for row in messages)


def encode_text_proxy(text: str, *, family: str, role: str, dimension: int = 64) -> np.ndarray:
    prefix = ""
    if family == "bge" and role == "query":
        prefix = "Represent this sentence for searching relevant passages: "
    elif family == "e5":
        prefix = "query: " if role == "query" else "passage: "
    prepared = f"{prefix}{text}".lower()
    values = np.zeros(dimension, dtype=np.float32)
    for idx, feature in enumerate(_lexical_features(prepared)):
        values[idx % dimension] += float(feature)
    for token in re.findall(r"[a-z0-9']+", prepared):
        digest = hashlib.sha256(f"{family}:{role}:{token}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[bucket] += sign / math.sqrt(max(len(token), 1))
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _rank_positive(query_text: str, candidate_texts: list[str], *, family: str) -> int:
    query_vec = encode_text_proxy(query_text, family=family, role="query")
    scores = [
        cosine(query_vec, encode_text_proxy(candidate, family=family, role="passage"))
        for candidate in candidate_texts
    ]
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return ranked.index(0) + 1


def evaluate_proxy_encoder(
    samples: list[dict[str, Any]],
    *,
    candidate: EncoderCandidate,
    negatives_per_query: int = 4,
) -> dict[str, float]:
    if not samples:
        return {"top1_accuracy": 0.0, "mrr": 0.0, "shuffled_target_mrr": 0.0, "num_examples": 0.0}
    hits = 0.0
    reciprocal_ranks: list[float] = []
    shuffled_reciprocal_ranks: list[float] = []
    targets = [str(sample.get("target_text", "")) for sample in samples]
    for idx, sample in enumerate(samples):
        query_text = render_context_text(sample.get("input_context", []))
        if not query_text:
            query_text = str(sample.get("shared_context_id", ""))
        positive = targets[idx]
        negatives = [targets[(idx + offset + 1) % len(targets)] for offset in range(min(negatives_per_query, max(len(targets) - 1, 0)))]
        rank = _rank_positive(query_text, [positive, *negatives], family=candidate.family)
        hits += float(rank == 1)
        reciprocal_ranks.append(1.0 / rank)

        shuffled_positive = targets[(idx + max(len(targets) // 2, 1)) % len(targets)]
        shuffled_rank = _rank_positive(query_text, [shuffled_positive, *negatives], family=candidate.family)
        shuffled_reciprocal_ranks.append(1.0 / shuffled_rank)
    return {
        "top1_accuracy": hits / len(samples),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "shuffled_target_mrr": sum(shuffled_reciprocal_ranks) / len(shuffled_reciprocal_ranks),
        "num_examples": float(len(samples)),
    }
