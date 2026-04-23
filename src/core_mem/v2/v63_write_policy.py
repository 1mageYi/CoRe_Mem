"""Confidence-aware four-way write policy utilities for stage-2 v6.3."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from core_mem.v2.schemas import Observation
from core_mem.v2.v61_learned_memory import (
    compact_memory,
    train_decision_head,
    train_v61_reader_readout,
    typed_observation,
)
from core_mem.v2.v62_write_memory import (
    CandidateObservation,
    DialogueTurn,
    V62BinaryTrainingResult,
    _non_stopword_tokens,
    _stopword_ratio,
    estimate_state_cleanliness,
    observation_attribute_valid_label,
    score_attribute_validity,
    score_write_worthiness,
    train_attribute_validity_model,
    train_write_worthiness_model,
    turn_storage_key,
    turn_write_worthy_label,
)
from core_mem.v2.v6_persistent_memory import (
    PersistentCoreResidualMemory,
    V6ReaderTrainingResult,
    V6RouterTrainingResult,
    relation_family,
    route_observation,
    train_write_router,
)

V63_POLICY_LABELS = (
    "core-worthy",
    "residual-worthy",
    "weak-but-keep",
    "drop",
)

_DURABLE_RELATIONS = {
    "constraint",
    "education_degree",
    "goal",
    "hobby",
    "location",
    "occupation",
    "profile_trait",
}
_VOLATILE_RELATIONS = {
    "environment_fact",
    "reason_fact",
    "social_fact",
    "temporal_fact",
}


@dataclass(frozen=True)
class V63WritePolicyBuildResult:
    memory: PersistentCoreResidualMemory
    policy_observations: list[Observation]
    router_result: V6RouterTrainingResult
    reader_result: V6ReaderTrainingResult
    decision_result: Any
    write_worthiness_result: V62BinaryTrainingResult
    attribute_validity_result: V62BinaryTrainingResult
    state_metrics: dict[str, Any]
    policy_eval: dict[str, Any]
    write_trace: list[dict[str, Any]]


def _observation_with_metadata(
    observation: Observation,
    *,
    confidence: float,
    metadata: dict[str, Any],
) -> Observation:
    payload = observation.to_dict()
    payload["confidence"] = max(0.0, min(1.0, confidence))
    merged_metadata = dict(payload.get("metadata", {}) or {})
    merged_metadata.update(metadata)
    payload["metadata"] = merged_metadata
    return Observation.from_dict(payload)


def _durable_observation(observation: Observation) -> bool:
    relation = relation_family(observation.relation)
    return (
        observation.value_type in {"constraint", "location", "occupation", "preference", "profile"}
        or observation.relation.endswith("preference")
        or relation in _DURABLE_RELATIONS
    )


def _volatile_observation(observation: Observation) -> bool:
    relation = relation_family(observation.relation)
    return (
        observation.time_scope in {"past", "recent_change"}
        or observation.polarity == "negative"
        or relation in _VOLATILE_RELATIONS
    )


def _weak_keep_eligible(observation: Observation) -> bool:
    value_tokens = _non_stopword_tokens(observation.value)
    if observation.speaker == "assistant":
        return False
    if len(value_tokens) < 2:
        return False
    if _stopword_ratio(observation.value) > 0.72:
        return False
    return True


def classify_write_policy(
    *,
    turn_score: float,
    validity_score: float,
    observation: Observation,
) -> str:
    typed = typed_observation(observation)
    if not _weak_keep_eligible(typed):
        return "drop"
    if observation_attribute_valid_label(typed) and validity_score >= 0.78 and turn_score >= 0.52:
        if _durable_observation(typed) and not _volatile_observation(typed) and typed.confidence >= 0.72:
            return "core-worthy"
        return "residual-worthy"
    combined = 0.55 * validity_score + 0.45 * turn_score
    if validity_score >= 0.62 or combined >= 0.54:
        return "residual-worthy"
    if combined >= 0.33:
        return "weak-but-keep"
    return "drop"


def materialize_policy_observation(
    observation: Observation,
    *,
    policy_label: str,
    turn_score: float,
    validity_score: float,
) -> Observation:
    typed = typed_observation(observation)
    if policy_label == "weak-but-keep":
        adjusted_confidence = max(0.36, min(0.59, 0.4 * turn_score + 0.6 * validity_score))
    elif policy_label == "residual-worthy":
        adjusted_confidence = max(typed.confidence, min(0.82, 0.3 * turn_score + 0.7 * validity_score))
    elif policy_label == "core-worthy":
        adjusted_confidence = max(typed.confidence, min(0.95, 0.35 * turn_score + 0.65 * validity_score))
    else:
        adjusted_confidence = min(typed.confidence, 0.25)
    return _observation_with_metadata(
        typed,
        confidence=adjusted_confidence,
        metadata={
            "write_policy_label": policy_label,
            "turn_score": float(turn_score),
            "validity_score": float(validity_score),
        },
    )


def policy_action_from_router(
    *,
    policy_label: str,
    router_action: str,
    observation: Observation,
) -> str:
    if policy_label == "drop":
        return "ignore"
    if policy_label == "weak-but-keep":
        if router_action == "overwrite":
            return "overwrite"
        return "merge_residual"
    if policy_label == "residual-worthy":
        if router_action in {"mark_stale", "overwrite"}:
            return router_action
        if router_action in {"new_residual", "merge_residual"}:
            return router_action
        return "merge_residual"
    if _volatile_observation(observation):
        return "merge_residual"
    if router_action == "promote_to_core":
        return "promote_to_core"
    return "merge_core"


def evaluate_confidence_aware_write_policy(
    turns: list[DialogueTurn],
    turn_candidates: dict[str, list[Observation]],
    *,
    write_worthiness_result: V62BinaryTrainingResult,
    attribute_validity_result: V62BinaryTrainingResult,
) -> dict[str, Any]:
    eval_turns = [turn for turn in turns if turn.speaker in {"user", "system"}][-min(256, len(turns)) :]
    oracle_turns = 0
    strict_turn_hits = 0
    v63_turn_hits = 0
    oracle_positive_candidates = 0
    strict_positive_candidates = 0
    v63_positive_candidates = 0
    weak_saved_candidates = 0
    policy_counts: Counter[str] = Counter()
    for turn in eval_turns:
        turn_score = score_write_worthiness(write_worthiness_result.model, turn)
        candidates = [typed_observation(item) for item in turn_candidates.get(turn_storage_key(turn), [])]
        if not candidates:
            continue
        strict_turn_hit = False
        v63_turn_hit = False
        oracle_turn = False
        for candidate in candidates:
            validity_score = score_attribute_validity(attribute_validity_result.model, candidate)
            strict_keep = (
                turn_score >= write_worthiness_result.threshold
                and validity_score >= attribute_validity_result.threshold
            )
            policy_label = classify_write_policy(
                turn_score=turn_score,
                validity_score=validity_score,
                observation=candidate,
            )
            policy_counts[policy_label] += 1
            oracle_positive = observation_attribute_valid_label(candidate)
            if oracle_positive:
                oracle_turn = True
                oracle_positive_candidates += 1
                strict_positive_candidates += int(strict_keep)
                v63_positive_candidates += int(policy_label != "drop")
                weak_saved_candidates += int(policy_label == "weak-but-keep" and not strict_keep)
                strict_turn_hit = strict_turn_hit or strict_keep
                v63_turn_hit = v63_turn_hit or (policy_label != "drop")
        if oracle_turn:
            oracle_turns += 1
            strict_turn_hits += int(strict_turn_hit)
            v63_turn_hits += int(v63_turn_hit)
    denom_candidates = max(oracle_positive_candidates, 1)
    denom_turns = max(oracle_turns, 1)
    return {
        "confidence_aware_policy_in_authoritative_write_path": True,
        "policy_labels": list(V63_POLICY_LABELS),
        "support_coverage_recall": v63_positive_candidates / denom_candidates,
        "disabled_support_coverage_recall": strict_positive_candidates / denom_candidates,
        "write_recall": v63_turn_hits / denom_turns,
        "disabled_write_recall": strict_turn_hits / denom_turns,
        "oracle_positive_candidates": oracle_positive_candidates,
        "strict_positive_candidates": strict_positive_candidates,
        "v63_positive_candidates": v63_positive_candidates,
        "weak_but_keep_saved_candidates": weak_saved_candidates,
        "policy_label_counts": dict(policy_counts),
    }


def build_v63_memory(
    turns: list[DialogueTurn],
    turn_candidates: dict[str, list[Observation]],
    candidate_pool: list[CandidateObservation],
    *,
    max_stream_observations: int,
) -> V63WritePolicyBuildResult:
    if not turns:
        raise ValueError("v6.3 requires at least one dialogue turn.")
    write_worthiness_result = train_write_worthiness_model(turns, turn_candidates)
    attribute_validity_result = train_attribute_validity_model(candidate_pool)
    policy_observations: list[Observation] = []
    write_trace: list[dict[str, Any]] = []
    policy_counts: Counter[str] = Counter()
    for turn in turns:
        if turn.speaker not in {"user", "system"}:
            continue
        turn_score = score_write_worthiness(write_worthiness_result.model, turn)
        candidates = [typed_observation(item) for item in turn_candidates.get(turn_storage_key(turn), [])]
        write_trace.append(
            {
                "event": "turn_policy",
                "source_dialogue_id": turn.source_dialogue_id,
                "source_turn_id": turn.source_turn_id,
                "speaker": turn.speaker,
                "write_worthiness_score": turn_score,
                "candidate_count": len(candidates),
            }
        )
        for candidate in candidates:
            validity_score = score_attribute_validity(attribute_validity_result.model, candidate)
            policy_label = classify_write_policy(
                turn_score=turn_score,
                validity_score=validity_score,
                observation=candidate,
            )
            policy_counts[policy_label] += 1
            strict_keep = (
                turn_score >= write_worthiness_result.threshold
                and validity_score >= attribute_validity_result.threshold
            )
            write_trace.append(
                {
                    "event": "candidate_policy",
                    "obs_id": candidate.obs_id,
                    "source_dialogue_id": candidate.source_dialogue_id,
                    "source_turn_id": candidate.source_turn_id,
                    "relation": candidate.relation,
                    "value": candidate.value,
                    "turn_score": turn_score,
                    "attribute_validity_score": validity_score,
                    "policy_label": policy_label,
                    "strict_keep": strict_keep,
                }
            )
            if policy_label == "drop":
                continue
            policy_observations.append(
                materialize_policy_observation(
                    candidate,
                    policy_label=policy_label,
                    turn_score=turn_score,
                    validity_score=validity_score,
                )
            )
            if len(policy_observations) >= max_stream_observations:
                break
        if len(policy_observations) >= max_stream_observations:
            break
    if len(policy_observations) < 64:
        raise ValueError("v6.3 retained observation stream is too small for training.")
    training_slice = policy_observations[: max(max_stream_observations, 200)]
    router_result = train_write_router(training_slice)
    memory = PersistentCoreResidualMemory()
    weak_but_keep_residual_count = 0
    for idx, observation in enumerate(policy_observations[:max_stream_observations]):
        policy_label = str(observation.metadata.get("write_policy_label", "drop"))
        router_action = route_observation(router_result.router, observation)
        action = policy_action_from_router(
            policy_label=policy_label,
            router_action=router_action,
            observation=observation,
        )
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else idx
        written = memory.write(observation, action, turn_index=turn_index, obs_index=idx)
        weak_but_keep_residual_count += int(
            policy_label == "weak-but-keep" and written is not None and written.bank == "residual"
        )
        write_trace.append(
            {
                "event": "memory_write",
                "obs_id": observation.obs_id,
                "source_dialogue_id": observation.source_dialogue_id,
                "source_turn_id": observation.source_turn_id,
                "policy_label": policy_label,
                "router_action": router_action,
                "applied_action": action,
                "written_bank": written.bank if written is not None else None,
            }
        )
    compaction_stats = compact_memory(memory)
    reader_result = train_v61_reader_readout(training_slice, memory)
    decision_result = train_decision_head(policy_observations, memory, reader_result.reader)
    policy_eval = evaluate_confidence_aware_write_policy(
        turns,
        turn_candidates,
        write_worthiness_result=write_worthiness_result,
        attribute_validity_result=attribute_validity_result,
    )
    policy_eval["weak_but_keep_routed_to_residual"] = weak_but_keep_residual_count > 0
    state_metrics = {
        **compaction_stats,
        **estimate_state_cleanliness(memory),
        "candidate_observation_count": len(candidate_pool),
        "policy_retained_observation_count": len(policy_observations),
        "weak_but_keep_residual_count": weak_but_keep_residual_count,
        "strict_drop_candidate_count": policy_counts.get("drop", 0),
        "policy_label_counts": dict(policy_counts),
    }
    return V63WritePolicyBuildResult(
        memory=memory,
        policy_observations=policy_observations,
        router_result=router_result,
        reader_result=reader_result,
        decision_result=decision_result,
        write_worthiness_result=write_worthiness_result,
        attribute_validity_result=attribute_validity_result,
        state_metrics=state_metrics,
        policy_eval=policy_eval,
        write_trace=write_trace,
    )


def evaluate_internal_v63(
    memory: PersistentCoreResidualMemory,
    observations: list[Observation],
    *,
    reader: Any,
    decision_head: Any,
) -> dict[str, Any]:
    from core_mem.v2.v62_write_memory import evaluate_internal_v62

    payload = evaluate_internal_v62(
        memory,
        observations,
        reader=reader,
        decision_head=decision_head,
    )
    payload["confidence_aware_write_policy_used"] = True
    return payload
