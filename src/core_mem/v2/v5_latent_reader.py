"""Query-conditioned latent reader evaluation for v5."""

from __future__ import annotations

from typing import Any

from core_mem.v2.v5_encoder_harness import EncoderCandidate, cosine, encode_text_proxy, render_context_text


def _candidate_targets(samples: list[dict[str, Any]], idx: int, negatives_per_query: int) -> list[str]:
    targets = [str(sample.get("target_text", "")) for sample in samples]
    positive = targets[idx]
    negatives = [targets[(idx + offset + 1) % len(targets)] for offset in range(min(negatives_per_query, max(len(targets) - 1, 0)))]
    return [positive, *negatives]


def _query_text(sample: dict[str, Any]) -> str:
    rendered = render_context_text(sample.get("input_context", []))
    return rendered or str(sample.get("shared_context_id", ""))


def _score_query_overlap(query_text: str, candidate_text: str) -> float:
    query_tokens = set(query_text.lower().split())
    candidate_tokens = set(candidate_text.lower().split())
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)


def _score_latent(query_text: str, candidate_text: str, *, candidate: EncoderCandidate) -> float:
    query_vec = encode_text_proxy(query_text, family=candidate.family, role="query")
    candidate_vec = encode_text_proxy(candidate_text, family=candidate.family, role="passage")
    return cosine(query_vec, candidate_vec) + 0.20 * _score_query_overlap(query_text, candidate_text)


def _score_text_only(query_text: str, candidate_text: str) -> float:
    if not query_text or not candidate_text:
        return 0.0
    return -abs(len(query_text) - len(candidate_text)) / max(len(query_text), len(candidate_text), 1)


def _rank(scores: list[float]) -> int:
    ranked = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
    return ranked.index(0) + 1


def _metrics_from_ranks(ranks: list[int], *, candidate_count: int) -> dict[str, float]:
    if not ranks:
        return {"top1_accuracy": 0.0, "mrr": 0.0, "random_top1": 0.0, "num_examples": 0.0}
    return {
        "top1_accuracy": sum(1.0 for rank in ranks if rank == 1) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "random_top1": 1.0 / candidate_count,
        "num_examples": float(len(ranks)),
    }


def evaluate_v5_latent_reader(
    samples: list[dict[str, Any]],
    *,
    candidate: EncoderCandidate,
    negatives_per_query: int = 4,
) -> dict[str, Any]:
    if not samples:
        return {
            "latent_only": {"top1_accuracy": 0.0, "mrr": 0.0, "random_top1": 0.0, "num_examples": 0.0},
            "shuffled_latent": {"top1_accuracy": 0.0, "mrr": 0.0, "random_top1": 0.0, "num_examples": 0.0},
            "text_only": {"top1_accuracy": 0.0, "mrr": 0.0, "random_top1": 0.0, "num_examples": 0.0},
            "full": {"top1_accuracy": 0.0, "mrr": 0.0, "random_top1": 0.0, "num_examples": 0.0},
        }
    latent_ranks: list[int] = []
    shuffled_ranks: list[int] = []
    text_ranks: list[int] = []
    full_ranks: list[int] = []
    candidate_count = min(negatives_per_query + 1, len(samples))
    for idx, sample in enumerate(samples):
        query_text = _query_text(sample)
        candidates = _candidate_targets(samples, idx, negatives_per_query)
        latent_scores = [_score_latent(query_text, text, candidate=candidate) for text in candidates]
        text_scores = [_score_text_only(query_text, text) for text in candidates]
        full_scores = [latent + 0.25 * text for latent, text in zip(latent_scores, text_scores)]
        shuffled_positive = str(samples[(idx + max(len(samples) // 2, 1)) % len(samples)].get("target_text", ""))
        shuffled_scores = [
            _score_latent(query_text, text, candidate=candidate)
            for text in [shuffled_positive, *candidates[1:]]
        ]
        latent_ranks.append(_rank(latent_scores))
        shuffled_ranks.append(_rank(shuffled_scores))
        text_ranks.append(_rank(text_scores))
        full_ranks.append(_rank(full_scores))
    return {
        "latent_only": _metrics_from_ranks(latent_ranks, candidate_count=candidate_count),
        "shuffled_latent": _metrics_from_ranks(shuffled_ranks, candidate_count=candidate_count),
        "text_only": _metrics_from_ranks(text_ranks, candidate_count=candidate_count),
        "full": _metrics_from_ranks(full_ranks, candidate_count=candidate_count),
    }
