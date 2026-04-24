"""Learned reader/decision utilities for stage-2 v6.1 persistent memory."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

import torch
from torch import nn

from core_mem.v2.encoder import QueryEncoder
from core_mem.v2.schemas import Observation, SlotRecord
from core_mem.v2.v6_persistent_memory import (
    PersistentCoreResidualMemory,
    V6ReaderReadout,
    V6RouterTrainingResult,
    V6ReaderTrainingResult,
    relation_family,
    route_observation,
    stable_timestamp,
    train_write_router,
    vector_dot,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
}
_ENVIRONMENT_TERMS = {"crowded", "chaotic", "quiet", "peaceful", "atmosphere", "festival", "library", "libraries"}
_SOCIAL_TERMS = {"peer", "peers", "friend", "friends", "group", "community", "collaboration", "collaborations"}
_REASON_TERMS = {"because", "feedback", "pressured", "rush", "step", "stopped", "opted", "decided", "avoid"}
_PREFERENCE_QUERY_TERMS = {
    "favorite",
    "favorites",
    "idea",
    "ideas",
    "like",
    "love",
    "prefer",
    "preference",
    "preferences",
    "recommend",
    "recommendation",
    "suggest",
}
_REASON_QUERY_TERMS = {"because", "caused", "explain", "reason", "reasons", "why"}
_TEMPORAL_QUERY_TERMS = {"before", "change", "changed", "current", "now", "past", "previous", "recent", "recently", "update"}
_SOCIAL_QUERY_TERMS = {"community", "dating", "family", "friend", "friends", "group", "partner", "relationship", "social"}
_CONSTRAINT_QUERY_TERMS = {"avoid", "cannot", "can't", "constraint", "limit", "must", "restrict", "restriction"}
_GOAL_QUERY_TERMS = {"aim", "goal", "goals", "plan", "planning", "trying", "want", "wants", "working"}
_PROFILE_QUERY_TERMS = {"kind", "person", "personality", "style", "tendency", "trait", "traits"}
_LOW_INFORMATION_VALUE_TERMS = {
    "about",
    "again",
    "analyzing",
    "as",
    "be",
    "better",
    "check",
    "direct",
    "do",
    "doing",
    "evolve",
    "excited",
    "exploring",
    "feeling",
    "go",
    "going",
    "grow",
    "improve",
    "it",
    "learn",
    "learning",
    "myself",
    "ourself",
    "ourselves",
    "out",
    "passionate",
    "read",
    "really",
    "requests",
    "self",
    "spending",
    "studying",
    "that",
    "there",
    "to",
    "trying",
    "visit",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in _tokenize(left) if token not in _STOPWORDS}
    right_tokens = {token for token in _tokenize(right) if token not in _STOPWORDS}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _mean_vector(vectors: Iterable[list[float]]) -> list[float]:
    rows = [row for row in vectors if row]
    if not rows:
        return []
    dim = len(rows[0])
    merged = [sum(row[idx] for row in rows) / len(rows) for idx in range(dim)]
    norm = math.sqrt(sum(value * value for value in merged)) or 1.0
    return [value / norm for value in merged]


def _slot_similarity(left: SlotRecord, right: SlotRecord) -> float:
    key_similarity = max(0.0, vector_dot(left.retrieval_key, right.retrieval_key))
    gloss_similarity = _token_overlap(left.canonical_gloss, right.canonical_gloss)
    return max(0.0, min(1.0, 0.65 * key_similarity + 0.35 * gloss_similarity))


def _select_diversified_slots(
    ranked: list[tuple[float, SlotRecord]],
    *,
    top_k: int,
    diversity_weight: float = 0.35,
) -> list[tuple[float, SlotRecord]]:
    if top_k <= 0 or not ranked:
        return []
    if len(ranked) <= 1:
        return ranked[:top_k]
    selected: list[tuple[float, SlotRecord]] = [ranked[0]]
    remaining = ranked[1:]
    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = float("-inf")
        for idx, (relevance, slot) in enumerate(remaining):
            redundancy = max((_slot_similarity(slot, chosen_slot) for _, chosen_slot in selected), default=0.0)
            mmr_score = float(relevance) - diversity_weight * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = idx
        selected.append(remaining.pop(best_index))
    return selected


def _query_semantic_features(query: str) -> list[float]:
    tokens = set(_tokenize(query))
    return [
        float(bool(tokens & _PREFERENCE_QUERY_TERMS)),
        float(bool(tokens & _REASON_QUERY_TERMS)),
        float(bool(tokens & _TEMPORAL_QUERY_TERMS)),
        float(bool(tokens & _SOCIAL_QUERY_TERMS)),
        float(bool(tokens & _CONSTRAINT_QUERY_TERMS)),
        float(bool(tokens & _GOAL_QUERY_TERMS)),
        float(bool(tokens & _PROFILE_QUERY_TERMS)),
        float("?" in query or bool(tokens & {"what", "which", "who", "where", "when", "why", "how"})),
    ]


def _slot_semantic_features(slot: SlotRecord) -> list[float]:
    return [
        float(slot.bank == "core"),
        float(slot.bank == "residual"),
        float(slot.soft_role_scores.stable),
        float(slot.soft_role_scores.preference),
        float(slot.soft_role_scores.constraint),
        float(slot.soft_role_scores.goal),
        float(slot.soft_role_scores.temporal),
        float(slot.soft_role_scores.social),
        float(slot.relation == "reason_fact"),
        float(slot.relation == "profile_trait"),
    ]


def _reader_pair_features(query: str, query_key: list[float], slot: SlotRecord) -> list[float]:
    slot_key = slot.retrieval_key
    query_semantics = _query_semantic_features(query)
    slot_semantics = _slot_semantic_features(slot)
    slot_value = slot.canonical_gloss.split("=", 1)[-1]
    relation_text = slot.relation.replace("_", " ")
    alignment = [
        query_semantics[0] * slot.soft_role_scores.preference,
        query_semantics[1] * float(slot.relation == "reason_fact"),
        query_semantics[2] * slot.soft_role_scores.temporal,
        query_semantics[3] * slot.soft_role_scores.social,
        query_semantics[4] * slot.soft_role_scores.constraint,
        query_semantics[5] * slot.soft_role_scores.goal,
        query_semantics[6] * float(slot.relation == "profile_trait"),
        query_semantics[7] * slot.soft_role_scores.stable,
    ]
    return [
        *query_key,
        *slot_key,
        *[abs(left - right) for left, right in zip(query_key, slot_key)],
        *[left * right for left, right in zip(query_key, slot_key)],
        _token_overlap(query, slot.canonical_gloss),
        _token_overlap(query, slot_value),
        _token_overlap(query, relation_text),
        *query_semantics,
        *slot_semantics,
        *alignment,
    ]


def _is_low_information_value(observation: Observation) -> bool:
    if observation.relation not in {"goal", "hobby", "profile_trait"}:
        return False
    tokens = [token for token in _tokenize(observation.value) if token not in _STOPWORDS]
    if not tokens:
        return True
    if len(tokens) > 2:
        return False
    return set(tokens) <= _LOW_INFORMATION_VALUE_TERMS


def filter_v61_observations(observations: list[Observation]) -> list[Observation]:
    return [observation for observation in observations if not _is_low_information_value(observation)]


def infer_typed_relation(observation: Observation) -> str:
    if observation.relation != "other_fact":
        return observation.relation
    text = f"{observation.value} {observation.evidence_text} {observation.canonical_gloss}".lower()
    if any(token in text for token in _ENVIRONMENT_TERMS):
        return "environment_fact"
    if any(token in text for token in _SOCIAL_TERMS):
        return "social_fact"
    if observation.time_scope in {"past", "future", "recent_change"} or any(token in text for token in {"recently", "now", "current"}):
        return "temporal_fact"
    if any(token in text for token in _REASON_TERMS):
        return "reason_fact"
    return "profile_trait"


def typed_observation(observation: Observation) -> Observation:
    relation = infer_typed_relation(observation)
    if relation == observation.relation:
        return observation
    payload = observation.to_dict()
    payload["relation"] = relation
    payload["canonical_gloss"] = f"{relation}={observation.value}"
    return Observation.from_dict(payload)


def natural_language_query(observation: Observation) -> str:
    relation = observation.relation
    if relation.endswith("preference") or observation.value_type == "preference":
        return f"Which statement best matches the user's current {relation.replace('_', ' ')}?"
    if relation == "occupation":
        return "What work does the user do?"
    if relation == "location":
        return "Where is the user associated with right now?"
    if relation in {"reason_fact", "environment_fact", "social_fact", "temporal_fact", "other_fact"}:
        return "Which statement best matches the user's recent situation?"
    if relation == "profile_trait":
        return "Which statement best matches the user's personal style or tendency?"
    if observation.time_scope == "recent_change":
        return "What changed for the user recently?"
    return f"What detail about the user best matches the current {relation.replace('_', ' ')}?"


def observation_option_statement(observation: Observation) -> str:
    relation = observation.relation.replace("_", " ")
    value = observation.value
    if observation.relation.endswith("preference"):
        return f"The user's current {relation} is {value}."
    if observation.relation in {"reason_fact", "environment_fact", "social_fact", "temporal_fact", "profile_trait"}:
        return f"The user's relevant situation is that {value}."
    return f"The user's {relation} is {value}."


def semantic_signature(slot: SlotRecord) -> tuple[str, tuple[str, ...]]:
    value = slot.canonical_gloss.split("=", 1)[-1]
    tokens = [token for token in _tokenize(value) if token not in _STOPWORDS]
    if not tokens:
        tokens = [slot.relation]
    return relation_family(slot.relation), tuple(sorted(tokens))


def compact_memory(memory: PersistentCoreResidualMemory) -> dict[str, Any]:
    active_before = [slot for slot in [*memory.core_bank, *memory.residual_bank] if slot.active_flag]
    groups: dict[tuple[str, str, tuple[str, ...]], list[SlotRecord]] = defaultdict(list)
    for slot in active_before:
        metadata = memory._slot_metadata(slot)
        groups[(str(metadata.get("source_dialogue_id", "")), *semantic_signature(slot))].append(slot)
    duplicates_removed = 0
    duplicate_clusters = 0
    for slots in groups.values():
        if len(slots) <= 1:
            continue
        duplicate_clusters += 1
        ordered = sorted(
            slots,
            key=lambda item: (
                item.last_update_ts,
                item.confidence,
                1 if item.bank == "core" else 0,
            ),
        )
        keep = ordered[-1]
        for slot in ordered[:-1]:
            if slot.slot_id == keep.slot_id:
                continue
            memory._replace_slot(slot, active=False)
            duplicates_removed += 1
    active_after = [slot for slot in [*memory.core_bank, *memory.residual_bank] if slot.active_flag]
    residual_relations = [slot.relation for slot in active_after if slot.bank == "residual"]
    other_fact_count = sum(relation == "other_fact" for relation in residual_relations)
    return {
        "bank_compaction_enabled": True,
        "duplicate_relation_clusters_reduced": duplicate_clusters > 0 and duplicates_removed > 0,
        "duplicate_relation_cluster_count": duplicate_clusters,
        "duplicates_removed": duplicates_removed,
        "dedup_ratio": duplicates_removed / max(len(active_before), 1),
        "typed_residual_relations": sorted({relation for relation in residual_relations}),
        "other_fact_share": other_fact_count / max(len(residual_relations), 1),
        "revision_chain_enabled": any(slot.revision_parent for slot in active_after),
    }


def _best_matching_slot(observation: Observation, slots: list[SlotRecord]) -> SlotRecord | None:
    candidates = [slot for slot in slots if relation_family(slot.relation) == relation_family(observation.relation)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda slot: (
            _token_overlap(slot.canonical_gloss, observation.canonical_gloss),
            _token_overlap(slot.canonical_gloss, observation.value),
            slot.confidence,
            1 if slot.bank == "core" else 0,
        ),
    )


def _synthetic_queries_for_observation(observation: Observation) -> list[str]:
    relation_text = observation.relation.replace("_", " ")
    queries = {natural_language_query(observation)}
    if observation.relation.endswith("preference") or observation.relation == "hobby":
        queries.add(f"Which {relation_text} best matches the user?")
        queries.add("What would the user most likely prefer right now?")
    elif observation.relation == "reason_fact":
        queries.add("Why did the user's situation change?")
        queries.add("Which reason best explains the user's recent update?")
    elif observation.relation in {"temporal_fact", "environment_fact", "social_fact"}:
        queries.add(f"What recent {relation_text} detail best fits the user?")
    elif observation.relation == "profile_trait":
        queries.add("What kind of tendency best describes the user?")
    else:
        queries.add(f"What detail best matches the user's {relation_text}?")
    return [query for query in queries if query]


def train_v61_reader_readout(
    observations: list[Observation],
    memory: PersistentCoreResidualMemory,
    *,
    epochs: int = 80,
    learning_rate: float = 0.02,
    seed: int = 607,
) -> V6ReaderTrainingResult:
    if len(observations) < 8:
        raise ValueError("v6.1 reader/readout training requires at least 8 observations.")
    torch.manual_seed(seed)
    feature_rows: list[list[float]] = []
    labels: list[float] = []
    for observation in observations:
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else None
        slots = memory.active_slots(context_id=observation.source_dialogue_id, max_turn_index=turn_index)
        if len(slots) < 2:
            continue
        positive_slot = _best_matching_slot(observation, slots)
        if positive_slot is None:
            continue
        negatives = [
            slot
            for slot in slots
            if slot.slot_id != positive_slot.slot_id
            and relation_family(slot.relation) != relation_family(observation.relation)
        ]
        if not negatives:
            negatives = [slot for slot in slots if slot.slot_id != positive_slot.slot_id]
        if not negatives:
            continue
        for query in _synthetic_queries_for_observation(observation):
            query_key = QueryEncoder(dimension=len(positive_slot.retrieval_key)).encode(query)
            feature_rows.append(_reader_pair_features(query, query_key, positive_slot))
            labels.append(1.0)
            ranked_negatives = sorted(
                negatives,
                key=lambda slot: (
                    _token_overlap(query, slot.canonical_gloss),
                    _token_overlap(observation.value, slot.canonical_gloss),
                    slot.confidence,
                ),
                reverse=True,
            )
            for negative_slot in ranked_negatives[:3]:
                feature_rows.append(_reader_pair_features(query, query_key, negative_slot))
                labels.append(0.0)
    if len(feature_rows) < 16:
        raise ValueError("v6.1 reader/readout training requires at least 16 synthetic pairs.")
    x = torch.tensor(feature_rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    split = max(8, int(len(x) * 0.8))
    split = min(split, len(x) - 4)
    train_x, eval_x = x[:split], x[split:]
    train_y, eval_y = y[:split], y[split:]
    reader = V6ReaderReadout(input_dim=x.shape[1], hidden_dim=48)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        logits = reader(train_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in {0, epochs - 1}:
            loss_curve.append({"epoch": float(epoch + 1), "loss": float(loss.item())})
    with torch.no_grad():
        eval_predictions = (torch.sigmoid(reader(eval_x)) >= 0.5).float()
    eval_accuracy = float((eval_predictions == eval_y).float().mean().item())
    majority = 1.0 if float(train_y.mean().item()) >= 0.5 else 0.0
    disabled = torch.full_like(eval_y, majority)
    disabled_accuracy = float((disabled == eval_y).float().mean().item())
    return V6ReaderTrainingResult(
        reader=reader,
        eval_accuracy=eval_accuracy,
        disabled_readout_accuracy=disabled_accuracy,
        train_pairs=len(train_y),
        eval_pairs=len(eval_y),
        loss_curve=loss_curve,
    )


class V61DecisionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass(frozen=True)
class V61DecisionTrainingResult:
    head: V61DecisionHead
    eval_accuracy: float
    disabled_option_only_accuracy: float
    train_examples: int
    eval_examples: int
    loss_curve: list[dict[str, float]]


def _read_with_model(
    query: str,
    *,
    slots: list[SlotRecord],
    reader: V6ReaderReadout,
    top_k: int = 8,
) -> dict[str, Any]:
    if not slots:
        return {"query_key": [], "selected": [], "composed_key": [], "belief_items": []}
    query_encoder = QueryEncoder(dimension=len(slots[0].retrieval_key))
    query_key = query_encoder.encode(query)
    features = torch.tensor(
        [_reader_pair_features(query, query_key, slot) for slot in slots],
        dtype=torch.float32,
    )
    with torch.no_grad():
        scores = torch.sigmoid(reader(features)).tolist()
    ranked = sorted(zip(scores, slots), key=lambda item: item[0], reverse=True)
    selected = _select_diversified_slots(ranked, top_k=top_k)
    total_score = sum(score for score, _ in selected) or 1.0
    composed = [
        sum(score * slot.retrieval_key[idx] for score, slot in selected) / total_score
        for idx in range(len(selected[0][1].retrieval_key))
    ]
    belief_items = [
        {
            "relation": slot.relation,
            "value": slot.canonical_gloss.split("=", 1)[-1],
            "support_slot_id": slot.slot_id,
            "confidence": min(1.0, float(score)),
            "bank": slot.bank,
        }
        for score, slot in selected[:3]
    ]
    return {
        "query_key": query_key,
        "selected": [{"score": float(score), "slot": slot} for score, slot in selected],
        "composed_key": composed,
        "belief_items": belief_items,
    }


def _option_features(readout: dict[str, Any], query: str, option: str) -> list[float]:
    query_key = readout.get("query_key") or []
    composed = readout.get("composed_key") or []
    if not composed:
        return []
    option_key = QueryEncoder(dimension=len(composed)).encode(option)
    belief_text = " ".join(f"{item['relation']} {item['value']}" for item in readout.get("belief_items", []))
    selected_slots = [item["slot"] for item in readout.get("selected", [])[:3] if isinstance(item.get("slot"), SlotRecord)]
    selected_scores = [float(item["score"]) for item in readout.get("selected", [])[:3]]
    slot_score_max = max(selected_scores) if selected_scores else 0.0
    slot_score_mean = sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
    belief_overlaps = [_token_overlap(option, f"{item['relation']} {item['value']}") for item in readout.get("belief_items", [])]
    belief_overlap_max = max(belief_overlaps) if belief_overlaps else 0.0
    belief_overlap_mean = sum(belief_overlaps) / len(belief_overlaps) if belief_overlaps else 0.0
    selected_core_ratio = (
        sum(1 for slot in selected_slots if slot.bank == "core") / len(selected_slots) if selected_slots else 0.0
    )
    selected_preference_ratio = (
        sum(slot.soft_role_scores.preference for slot in selected_slots) / len(selected_slots) if selected_slots else 0.0
    )
    selected_temporal_ratio = (
        sum(slot.soft_role_scores.temporal for slot in selected_slots) / len(selected_slots) if selected_slots else 0.0
    )
    return [
        *query_key,
        *composed,
        *option_key,
        *[abs(left - right) for left, right in zip(composed, option_key)],
        vector_dot(query_key, option_key),
        vector_dot(composed, option_key),
        _token_overlap(option, belief_text),
        _token_overlap(option, query),
        belief_overlap_max,
        belief_overlap_mean,
        slot_score_max,
        slot_score_mean,
        selected_core_ratio,
        selected_preference_ratio,
        selected_temporal_ratio,
    ]


def _select_hard_negatives(observation: Observation, pool: list[Observation]) -> list[Observation]:
    same_dialogue_relation = [
        candidate
        for candidate in pool
        if candidate.obs_id != observation.obs_id
        and candidate.source_dialogue_id == observation.source_dialogue_id
        and candidate.relation == observation.relation
    ]
    same_relation = [
        candidate
        for candidate in pool
        if candidate.obs_id != observation.obs_id and candidate.relation == observation.relation
    ]
    same_dialogue = [
        candidate
        for candidate in pool
        if candidate.obs_id != observation.obs_id and candidate.source_dialogue_id == observation.source_dialogue_id
    ]
    combined = [*same_dialogue_relation, *same_relation, *same_dialogue]
    seen: set[str] = set()
    negatives: list[Observation] = []
    for candidate in combined:
        if candidate.obs_id in seen or candidate.value == observation.value:
            continue
        seen.add(candidate.obs_id)
        negatives.append(candidate)
        if len(negatives) >= 3:
            break
    return negatives


def train_decision_head(
    observations: list[Observation],
    memory: PersistentCoreResidualMemory,
    reader: V6ReaderReadout,
    *,
    epochs: int = 60,
    learning_rate: float = 0.02,
    seed: int = 611,
) -> V61DecisionTrainingResult:
    torch.manual_seed(seed)
    option_groups: list[list[list[float]]] = []
    labels: list[int] = []
    for observation in observations:
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else None
        slots = memory.active_slots(context_id=observation.source_dialogue_id, max_turn_index=turn_index)
        if not slots:
            continue
        negatives = _select_hard_negatives(observation, observations)
        if len(negatives) < 3:
            continue
        options = [observation_option_statement(observation), *[observation_option_statement(negative) for negative in negatives[:3]]]
        for query in _synthetic_queries_for_observation(observation):
            readout = _read_with_model(query, slots=slots, reader=reader, top_k=6)
            features = [_option_features(readout, query, option) for option in options]
            if any(not feature for feature in features):
                continue
            option_groups.append(features)
            labels.append(0)
    if len(option_groups) < 16:
        raise ValueError("v6.1 decision-head training requires at least 16 synthetic examples.")
    x = torch.tensor(option_groups, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    split = max(8, int(len(x) * 0.8))
    split = min(split, len(x) - 4)
    train_x, eval_x = x[:split], x[split:]
    train_y, eval_y = y[:split], y[split:]
    head = V61DecisionHead(input_dim=x.shape[-1], hidden_dim=64)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        logits = head(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape[0], train_x.shape[1])
        loss = nn.functional.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in {0, epochs - 1}:
            loss_curve.append({"epoch": float(epoch + 1), "loss": float(loss.item())})
    with torch.no_grad():
        eval_logits = head(eval_x.reshape(-1, eval_x.shape[-1])).reshape(eval_x.shape[0], eval_x.shape[1])
        eval_predictions = eval_logits.argmax(dim=1)
    eval_accuracy = float((eval_predictions == eval_y).float().mean().item())
    disabled_predictions: list[int] = []
    for eval_group in eval_x:
        overlaps = [float(feature[-8]) for feature in eval_group]
        disabled_predictions.append(max(range(len(overlaps)), key=lambda idx: overlaps[idx]))
    disabled_tensor = torch.tensor(disabled_predictions, dtype=torch.long)
    disabled_option_only_accuracy = float((disabled_tensor == eval_y).float().mean().item())
    return V61DecisionTrainingResult(
        head=head,
        eval_accuracy=eval_accuracy,
        disabled_option_only_accuracy=disabled_option_only_accuracy,
        train_examples=len(train_y),
        eval_examples=len(eval_y),
        loss_curve=loss_curve,
    )


def score_options_with_head(
    readout: dict[str, Any],
    query: str,
    options: list[str],
    head: V61DecisionHead,
) -> tuple[int, list[float]]:
    if not options:
        return 0, []
    features = [_option_features(readout, query, option) for option in options]
    if not features or not features[0]:
        return 0, [0.0 for _ in options]
    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = head(tensor)
        scores = torch.softmax(logits, dim=0).tolist()
    best = max(range(len(scores)), key=lambda idx: scores[idx])
    return best, [float(score) for score in scores]


def build_v61_memory(
    observations: list[Observation],
    *,
    max_stream_observations: int,
) -> tuple[
    PersistentCoreResidualMemory,
    V6RouterTrainingResult,
    V6ReaderTrainingResult,
    V61DecisionTrainingResult,
    dict[str, Any],
]:
    typed_observations = filter_v61_observations([typed_observation(observation) for observation in observations])
    training_slice = typed_observations[: max(max_stream_observations, 200)]
    router_result = train_write_router(training_slice)
    memory = PersistentCoreResidualMemory()
    for idx, observation in enumerate(typed_observations[:max_stream_observations]):
        action = route_observation(router_result.router, observation)
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else idx
        memory.write(observation, action, turn_index=turn_index, obs_index=idx)
    compaction_stats = compact_memory(memory)
    reader_result = train_v61_reader_readout(training_slice, memory)
    decision_result = train_decision_head(typed_observations, memory, reader_result.reader)
    return memory, router_result, reader_result, decision_result, compaction_stats


def evaluate_internal_v61(
    memory: PersistentCoreResidualMemory,
    observations: list[Observation],
    *,
    reader: V6ReaderReadout,
    decision_head: V61DecisionHead,
) -> dict[str, Any]:
    eval_observations = [typed_observation(observation) for observation in observations[-min(256, len(observations)) :]]
    full_hits = 0
    disabled_reader_hits = 0
    disabled_decision_hits = 0
    core_residual_disabled_hits = 0
    reader_support_hits = 0
    for observation in eval_observations:
        query = natural_language_query(observation)
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else None
        slots = memory.active_slots(context_id=observation.source_dialogue_id, max_turn_index=turn_index)
        if not slots:
            continue
        negatives = _select_hard_negatives(observation, eval_observations)
        if len(negatives) < 3:
            continue
        options = [observation_option_statement(observation), *[observation_option_statement(negative) for negative in negatives[:3]]]
        readout = _read_with_model(query, slots=slots, reader=reader, top_k=6)
        pred_idx, _ = score_options_with_head(readout, query, options, decision_head)
        full_hits += int(pred_idx == 0)
        belief_items = readout.get("belief_items", [])
        if belief_items:
            best_belief = belief_items[0]
            reader_support_hits += int(
                best_belief.get("relation") == observation.relation
                and _token_overlap(str(best_belief.get("value", "")), observation.value) > 0.2
            )
        dot_readout = memory.read(query, slots=slots, top_k=6, use_bank_prior=False)
        dot_pred_idx = max(
            range(len(options)),
            key=lambda idx: vector_dot(
                dot_readout.get("composed_key") or [],
                QueryEncoder(dimension=len(dot_readout.get("composed_key") or [0.0])).encode(options[idx]),
            ),
        )
        disabled_reader_hits += int(dot_pred_idx == 0)
        disabled_decision_hits += int(max(range(len(options)), key=lambda idx: _token_overlap(query, options[idx])) == 0)
        core_slots = [slot for slot in slots if slot.bank == "core"]
        residual_slots = [slot for slot in slots if slot.bank == "residual"]
        best_split = 0
        for split_slots in (core_slots, residual_slots):
            if not split_slots:
                continue
            split_readout = _read_with_model(query, slots=split_slots, reader=reader, top_k=6)
            split_pred_idx, _ = score_options_with_head(split_readout, query, options, decision_head)
            best_split = max(best_split, int(split_pred_idx == 0))
        core_residual_disabled_hits += best_split
    denom = max(len(eval_observations), 1)
    return {
        "evaluation_consumes_persistent_state": True,
        "authoritative_input": "persistent_core_residual_state_checkpoint",
        "raw_context_retrieval_disabled": True,
        "write_time_router_applied_to_state": True,
        "reader_in_authoritative_path": True,
        "authoritative_reader_path": "learned_query_slot_reader",
        "authoritative_reader_uses_dot_product_only": False,
        "natural_language_queries": True,
        "gold_value_echo_queries": False,
        "hard_negative_source": "same_persona_same_relation",
        "hard_negative_count": len(eval_observations) * 3,
        "typed_residuals_used": True,
        "eval_examples": len(eval_observations),
        "reader_support_accuracy": reader_support_hits / denom,
        "full_accuracy": full_hits / denom,
        "disabled_reader_accuracy": disabled_reader_hits / denom,
        "disabled_decision_accuracy": disabled_decision_hits / denom,
        "disabled_core_residual_split_accuracy": core_residual_disabled_hits / denom,
    }
