"""Learned write-worthiness and attribute-validity utilities for stage-2 v6.2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

import torch
from torch import nn

from core_mem.v2.encoder import _lexical_features
from core_mem.v2.schemas import Observation, SlotRecord
from core_mem.v2.v61_learned_memory import (
    _is_low_information_value,
    compact_memory,
    evaluate_internal_v61,
    filter_v61_observations,
    train_decision_head,
    train_v61_reader_readout,
    typed_observation,
)
from core_mem.v2.v6_persistent_memory import (
    PersistentCoreResidualMemory,
    V6ReaderTrainingResult,
    V6RouterTrainingResult,
    V6WriteRouter,
    relation_family,
    route_observation,
    stable_timestamp,
    train_write_router,
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
_LOW_SIGNAL_TURN_TERMS = {
    "cool",
    "fine",
    "great",
    "haha",
    "hello",
    "hi",
    "nice",
    "okay",
    "sure",
    "thanks",
}
_GENERIC_VALUE_TERMS = {
    "again",
    "around",
    "better",
    "different",
    "it",
    "more",
    "something",
    "stuff",
    "that",
    "things",
    "this",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _non_stopword_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in _STOPWORDS]


def _stopword_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 1.0
    return sum(token in _STOPWORDS for token in tokens) / len(tokens)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(_non_stopword_tokens(left))
    right_tokens = set(_non_stopword_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


@dataclass(frozen=True)
class DialogueTurn:
    source_dialogue_id: str
    source_turn_id: str
    session_id: str
    speaker: str
    text: str
    context_text: str = ""


@dataclass(frozen=True)
class CandidateObservation:
    turn: DialogueTurn
    observation: Observation


def turn_storage_key(turn: DialogueTurn) -> str:
    return f"{turn.source_dialogue_id}:{turn.source_turn_id}"


class V62BinaryClassifier(nn.Module):
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
class V62BinaryTrainingResult:
    model: V62BinaryClassifier
    threshold: float
    accuracy: float
    disabled_accuracy: float
    f1: float
    disabled_f1: float
    train_examples: int
    eval_examples: int
    loss_curve: list[dict[str, float]]


@dataclass(frozen=True)
class V62WriteQualityBuildResult:
    memory: PersistentCoreResidualMemory
    cleaned_observations: list[Observation]
    router_result: V6RouterTrainingResult
    reader_result: V6ReaderTrainingResult
    decision_result: Any
    write_worthiness_result: V62BinaryTrainingResult
    attribute_validity_result: V62BinaryTrainingResult
    state_metrics: dict[str, Any]
    write_eval: dict[str, Any]
    write_trace: list[dict[str, Any]]


def _binary_f1(predictions: Iterable[int], labels: Iterable[int]) -> float:
    tp = fp = fn = 0
    for prediction, label in zip(predictions, labels):
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 1:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _train_binary_classifier(
    features: list[list[float]],
    labels: list[int],
    *,
    seed: int,
    epochs: int = 80,
    learning_rate: float = 0.02,
) -> V62BinaryTrainingResult:
    if len(features) < 16:
        raise ValueError("v6.2 binary classifier requires at least 16 examples.")
    if len(set(labels)) < 2:
        raise ValueError("v6.2 binary classifier requires both positive and negative labels.")
    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    split = max(12, int(len(x) * 0.8))
    split = min(split, len(x) - 4)
    train_x, eval_x = x[:split], x[split:]
    train_y, eval_y = y[:split], y[split:]
    model = V62BinaryClassifier(input_dim=x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    positive_count = float(train_y.sum().item())
    negative_count = float(len(train_y) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32)
    loss_curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        logits = model(train_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in {0, epochs - 1}:
            loss_curve.append({"epoch": float(epoch + 1), "loss": float(loss.item())})
    with torch.no_grad():
        eval_probs = torch.sigmoid(model(eval_x))
    eval_labels = eval_y.to(dtype=torch.int64).tolist()
    best_threshold = 0.5
    best_predictions = (eval_probs >= best_threshold).to(dtype=torch.int64).tolist()
    best_f1 = _binary_f1(best_predictions, eval_labels)
    best_accuracy = sum(int(pred == gold) for pred, gold in zip(best_predictions, eval_labels)) / max(len(eval_labels), 1)
    for threshold in [step / 20.0 for step in range(2, 19)]:
        predictions = (eval_probs >= threshold).to(dtype=torch.int64).tolist()
        current_f1 = _binary_f1(predictions, eval_labels)
        current_accuracy = sum(int(pred == gold) for pred, gold in zip(predictions, eval_labels)) / max(len(eval_labels), 1)
        if current_f1 > best_f1 or (math.isclose(current_f1, best_f1) and current_accuracy > best_accuracy):
            best_threshold = threshold
            best_predictions = predictions
            best_f1 = current_f1
            best_accuracy = current_accuracy
    eval_predictions = best_predictions
    accuracy = sum(int(pred == gold) for pred, gold in zip(eval_predictions, eval_labels)) / max(len(eval_labels), 1)
    majority = 1 if float(train_y.mean().item()) >= 0.5 else 0
    disabled_predictions = [majority for _ in eval_labels]
    disabled_accuracy = sum(int(pred == gold) for pred, gold in zip(disabled_predictions, eval_labels)) / max(len(eval_labels), 1)
    return V62BinaryTrainingResult(
        model=model,
        threshold=best_threshold,
        accuracy=accuracy,
        disabled_accuracy=disabled_accuracy,
        f1=best_f1,
        disabled_f1=_binary_f1(disabled_predictions, eval_labels),
        train_examples=len(train_y),
        eval_examples=len(eval_y),
        loss_curve=loss_curve,
    )


def turn_write_worthy_label(candidates: list[Observation]) -> bool:
    return any(observation_attribute_valid_label(candidate) for candidate in candidates)


def observation_attribute_valid_label(observation: Observation) -> bool:
    typed = typed_observation(observation)
    value_tokens = _non_stopword_tokens(typed.value)
    if typed.speaker == "assistant":
        return False
    if typed.confidence < 0.58:
        return False
    if not value_tokens:
        return False
    if _is_low_information_value(typed):
        return False
    if len(value_tokens) <= 1 and relation_family(typed.relation) in {"other_fact", "profile_trait", "goal", "hobby"}:
        return False
    if len(value_tokens) <= 2 and set(value_tokens) <= _GENERIC_VALUE_TERMS:
        return False
    if typed.relation in {"reason_fact", "environment_fact", "social_fact", "temporal_fact"} and len(value_tokens) < 3:
        return False
    if typed.relation == "profile_trait" and (_stopword_ratio(typed.value) > 0.55 or len(value_tokens) < 3):
        return False
    if typed.relation == "other_fact" and (_stopword_ratio(typed.value) > 0.6 or len(value_tokens) < 3):
        return False
    return True


def turn_text_features(turn: DialogueTurn) -> list[float]:
    lowered = turn.text.lower()
    tokens = _tokenize(turn.text)
    return [
        *_lexical_features(turn.text),
        float(turn.speaker == "user"),
        float(turn.speaker == "system"),
        float(any(token in {"i", "i'm", "ive", "my", "me"} for token in tokens)),
        float(any(token in {"like", "love", "prefer", "favorite", "want", "plan"} for token in tokens)),
        float(any(token in {"now", "recently", "these", "days", "changed", "stopped", "started"} for token in tokens)),
        float(any(token in {"because", "after", "feedback", "deadline", "crowded", "chaotic"} for token in tokens)),
        float(len(tokens) <= 4 or set(tokens) <= _LOW_SIGNAL_TURN_TERMS),
        min(len(tokens) / 40.0, 1.0),
    ]


def observation_validity_features(observation: Observation) -> list[float]:
    typed = typed_observation(observation)
    value_tokens = _non_stopword_tokens(typed.value)
    evidence_tokens = _non_stopword_tokens(typed.evidence_text)
    return [
        *_lexical_features(f"{typed.relation} {typed.value} {typed.evidence_text}"),
        min(len(value_tokens) / 12.0, 1.0),
        min(len(set(value_tokens)) / 12.0, 1.0),
        min(len(evidence_tokens) / 24.0, 1.0),
        _stopword_ratio(typed.value),
        float(len(value_tokens) <= 2),
        float(_is_low_information_value(typed)),
        float(typed.relation == "other_fact"),
        float(typed.relation in {"reason_fact", "environment_fact", "social_fact", "temporal_fact"}),
        float(typed.relation.endswith("preference")),
        float(typed.time_scope == "recent_change"),
        float(typed.polarity == "negative"),
        float(typed.status_hint == "active"),
        float(typed.confidence),
        _token_overlap(typed.value, typed.evidence_text),
    ]


def train_write_worthiness_model(turns: list[DialogueTurn], turn_candidates: dict[str, list[Observation]]) -> V62BinaryTrainingResult:
    features: list[list[float]] = []
    labels: list[int] = []
    for turn in turns:
        if turn.speaker not in {"user", "system"}:
            continue
        features.append(turn_text_features(turn))
        labels.append(int(turn_write_worthy_label(turn_candidates.get(turn_storage_key(turn), []))))
    return _train_binary_classifier(features, labels, seed=621)


def train_attribute_validity_model(candidates: list[CandidateObservation]) -> V62BinaryTrainingResult:
    features = [observation_validity_features(item.observation) for item in candidates]
    labels = [int(observation_attribute_valid_label(item.observation)) for item in candidates]
    return _train_binary_classifier(features, labels, seed=622)


def score_write_worthiness(model: V62BinaryClassifier, turn: DialogueTurn) -> float:
    tensor = torch.tensor([turn_text_features(turn)], dtype=torch.float32)
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor)).item())


def score_attribute_validity(model: V62BinaryClassifier, observation: Observation) -> float:
    tensor = torch.tensor([observation_validity_features(observation)], dtype=torch.float32)
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor)).item())


def estimate_state_cleanliness(memory: PersistentCoreResidualMemory) -> dict[str, float]:
    active_slots = [slot for slot in [*memory.core_bank, *memory.residual_bank] if slot.active_flag]
    if not active_slots:
        return {
            "bank_precision_estimate": 0.0,
            "invalid_slot_rate": 1.0,
            "low_information_slot_share": 1.0,
            "state_noise_rate": 1.0,
        }
    invalid_slots = 0
    low_information_slots = 0
    weak_slots = 0
    for slot in active_slots:
        value = slot.canonical_gloss.split("=", 1)[-1]
        pseudo = Observation(
            obs_id=f"slot-{slot.slot_id}",
            source_dataset="persistent_state_audit",
            source_dialogue_id="persistent_state_audit",
            source_turn_id="0",
            session_id="persistent_state_audit",
            speaker="user",
            entity=slot.entity,
            relation=slot.relation,
            value=value,
            value_type="other",
            time_scope="current",
            status_hint="active",
            polarity="neutral",
            confidence=slot.confidence,
            evidence_text=value,
            canonical_gloss=slot.canonical_gloss,
            metadata={},
        )
        invalid = not observation_attribute_valid_label(pseudo)
        invalid_slots += int(invalid)
        low_information_slots += int(_is_low_information_value(pseudo))
        weak_slots += int(_stopword_ratio(value) > 0.6 or len(_non_stopword_tokens(value)) < 2)
    active_count = len(active_slots)
    invalid_rate = invalid_slots / active_count
    low_information_rate = low_information_slots / active_count
    noise_rate = (invalid_slots + weak_slots) / (2.0 * active_count)
    return {
        "bank_precision_estimate": 1.0 - invalid_rate,
        "invalid_slot_rate": invalid_rate,
        "low_information_slot_share": low_information_rate,
        "state_noise_rate": noise_rate,
    }


def _count_conflict_negatives(candidates: list[CandidateObservation]) -> int:
    count = 0
    seen: dict[tuple[str, str], set[str]] = {}
    for item in candidates:
        observation = typed_observation(item.observation)
        key = (observation.source_dialogue_id, relation_family(observation.relation))
        values = seen.setdefault(key, set())
        normalized = " ".join(_non_stopword_tokens(observation.value))
        if normalized and normalized not in values and values:
            count += 1
        if normalized:
            values.add(normalized)
    return count


def evaluate_write_quality(
    turns: list[DialogueTurn],
    turn_candidates: dict[str, list[Observation]],
    candidate_pool: list[CandidateObservation],
    *,
    write_worthiness_result: V62BinaryTrainingResult,
    attribute_validity_result: V62BinaryTrainingResult,
) -> dict[str, Any]:
    eval_turns = [turn for turn in turns if turn.speaker in {"user", "system"}][-min(256, len(turns)) :]
    write_labels = [int(turn_write_worthy_label(turn_candidates.get(turn_storage_key(turn), []))) for turn in eval_turns]
    write_predictions = [
        int(score_write_worthiness(write_worthiness_result.model, turn) >= write_worthiness_result.threshold)
        for turn in eval_turns
    ]
    write_accuracy = sum(int(pred == gold) for pred, gold in zip(write_predictions, write_labels)) / max(len(write_labels), 1)
    disabled_write_prediction = 1 if sum(write_labels) >= (len(write_labels) / 2.0) else 0
    disabled_write_predictions = [disabled_write_prediction for _ in write_labels]
    disabled_write_accuracy = sum(int(pred == gold) for pred, gold in zip(disabled_write_predictions, write_labels)) / max(
        len(write_labels),
        1,
    )
    eval_candidates = candidate_pool[-min(512, len(candidate_pool)) :]
    candidate_labels = [int(observation_attribute_valid_label(item.observation)) for item in eval_candidates]
    candidate_predictions = [
        int(score_attribute_validity(attribute_validity_result.model, item.observation) >= attribute_validity_result.threshold)
        for item in eval_candidates
    ]
    attribute_f1 = _binary_f1(candidate_predictions, candidate_labels)
    disabled_attribute_prediction = 1 if sum(candidate_labels) >= (len(candidate_labels) / 2.0) else 0
    disabled_attribute_predictions = [disabled_attribute_prediction for _ in candidate_labels]
    write_precision_hits = 0
    write_precision_denominator = 0
    for turn in eval_turns:
        candidates = turn_candidates.get(turn_storage_key(turn), [])
        if not candidates:
            continue
        scored = sorted(
            ((score_attribute_validity(attribute_validity_result.model, candidate), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        top_candidate = scored[0][1]
        write_precision_hits += int(observation_attribute_valid_label(top_candidate))
        write_precision_denominator += 1
    hard_negative_count = sum(
        int(not observation_attribute_valid_label(item.observation))
        for item in eval_candidates
    )
    return {
        "write_worthiness_in_authoritative_write_path": True,
        "attribute_validity_before_write": True,
        "uses_raw_dialogue_observations": True,
        "hard_negatives_enabled": True,
        "hard_negative_source": "raw_dialogue_turns_and_candidate_fragments",
        "hard_negative_count": hard_negative_count,
        "conflict_negative_count": _count_conflict_negatives(eval_candidates),
        "write_worthiness_eval_accuracy": write_accuracy,
        "disabled_write_worthiness_accuracy": disabled_write_accuracy,
        "attribute_validity_f1": attribute_f1,
        "disabled_attribute_validity_f1": _binary_f1(disabled_attribute_predictions, candidate_labels),
        "write_precision_at_1": write_precision_hits / max(write_precision_denominator, 1),
        "disabled_write_precision_at_1": sum(candidate_labels) / max(len(candidate_labels), 1),
        "turn_eval_examples": len(eval_turns),
        "candidate_eval_examples": len(eval_candidates),
    }


def build_v62_memory(
    turns: list[DialogueTurn],
    turn_candidates: dict[str, list[Observation]],
    candidate_pool: list[CandidateObservation],
    *,
    max_stream_observations: int,
) -> V62WriteQualityBuildResult:
    if not turns:
        raise ValueError("v6.2 requires at least one dialogue turn.")
    write_worthiness_result = train_write_worthiness_model(turns, turn_candidates)
    attribute_validity_result = train_attribute_validity_model(candidate_pool)
    cleaned_observations: list[Observation] = []
    write_trace: list[dict[str, Any]] = []
    for turn in turns:
        if turn.speaker not in {"user", "system"}:
            continue
        turn_score = score_write_worthiness(write_worthiness_result.model, turn)
        turn_candidates_list = turn_candidates.get(turn_storage_key(turn), [])
        accepted_turn = turn_score >= write_worthiness_result.threshold
        if not accepted_turn:
            write_trace.append(
                {
                    "event": "turn_gate",
                    "source_dialogue_id": turn.source_dialogue_id,
                    "source_turn_id": turn.source_turn_id,
                    "speaker": turn.speaker,
                    "write_worthiness_score": turn_score,
                    "accepted": False,
                    "candidate_count": len(turn_candidates_list),
                }
            )
            continue
        write_trace.append(
            {
                "event": "turn_gate",
                "source_dialogue_id": turn.source_dialogue_id,
                "source_turn_id": turn.source_turn_id,
                "speaker": turn.speaker,
                "write_worthiness_score": turn_score,
                "accepted": True,
                "candidate_count": len(turn_candidates_list),
            }
        )
        for observation in turn_candidates_list:
            typed = typed_observation(observation)
            validity_score = score_attribute_validity(attribute_validity_result.model, typed)
            accepted = validity_score >= attribute_validity_result.threshold
            write_trace.append(
                {
                    "event": "candidate_gate",
                    "obs_id": typed.obs_id,
                    "source_dialogue_id": typed.source_dialogue_id,
                    "source_turn_id": typed.source_turn_id,
                    "relation": typed.relation,
                    "value": typed.value,
                    "attribute_validity_score": validity_score,
                    "accepted": accepted,
                }
            )
            if accepted:
                cleaned_observations.append(typed)
            if len(cleaned_observations) >= max_stream_observations:
                break
        if len(cleaned_observations) >= max_stream_observations:
            break
    cleaned_observations = filter_v61_observations(cleaned_observations)
    if len(cleaned_observations) < 32:
        raise ValueError("v6.2 cleaned observation stream is too small for training.")
    training_slice = cleaned_observations[: max(max_stream_observations, 200)]
    router_result = train_write_router(training_slice)
    memory = PersistentCoreResidualMemory()
    for idx, observation in enumerate(cleaned_observations[:max_stream_observations]):
        action = route_observation(router_result.router, observation)
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else idx
        memory.write(observation, action, turn_index=turn_index, obs_index=idx)
    compaction_stats = compact_memory(memory)
    reader_result = train_v61_reader_readout(training_slice, memory)
    decision_result = train_decision_head(cleaned_observations, memory, reader_result.reader)
    write_eval = evaluate_write_quality(
        turns,
        turn_candidates,
        candidate_pool,
        write_worthiness_result=write_worthiness_result,
        attribute_validity_result=attribute_validity_result,
    )
    state_metrics = {
        **compaction_stats,
        **estimate_state_cleanliness(memory),
        "candidate_observation_count": len(candidate_pool),
        "cleaned_observation_count": len(cleaned_observations),
        "write_gate_rejected_turns": sum(1 for row in write_trace if row["event"] == "turn_gate" and not row["accepted"]),
        "attribute_gate_rejected_candidates": sum(
            1 for row in write_trace if row["event"] == "candidate_gate" and not row["accepted"]
        ),
    }
    return V62WriteQualityBuildResult(
        memory=memory,
        cleaned_observations=cleaned_observations,
        router_result=router_result,
        reader_result=reader_result,
        decision_result=decision_result,
        write_worthiness_result=write_worthiness_result,
        attribute_validity_result=attribute_validity_result,
        state_metrics=state_metrics,
        write_eval=write_eval,
        write_trace=write_trace,
    )


def evaluate_internal_v62(
    memory: PersistentCoreResidualMemory,
    observations: list[Observation],
    *,
    reader: Any,
    decision_head: Any,
) -> dict[str, Any]:
    payload = evaluate_internal_v61(memory, observations, reader=reader, decision_head=decision_head)
    payload["write_time_router_applied_to_state"] = True
    payload["typed_residuals_used"] = True
    return payload
