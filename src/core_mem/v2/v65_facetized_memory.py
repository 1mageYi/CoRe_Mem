"""Facetized observation-to-memory utilities for stage-2 v6.5."""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any

from core_mem.v2.schemas import FacetRecord, Observation
from core_mem.v2.v61_learned_memory import compact_memory, train_decision_head, train_v61_reader_readout, typed_observation
from core_mem.v2.v62_write_memory import (
    CandidateObservation,
    DialogueTurn,
    score_attribute_validity,
    score_write_worthiness,
    train_attribute_validity_model,
    train_write_worthiness_model,
    turn_storage_key,
)
from core_mem.v2.v63_write_policy import (
    V63WritePolicyBuildResult,
    classify_write_policy,
    evaluate_confidence_aware_write_policy,
    materialize_policy_observation,
    policy_action_from_router,
)
from core_mem.v2.v6_persistent_memory import PersistentCoreResidualMemory, relation_family, route_observation, train_write_router


_STOPWORDS = {
    "a",
    "about",
    "again",
    "also",
    "am",
    "an",
    "and",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "i",
    "i'm",
    "i've",
    "in",
    "into",
    "is",
    "it",
    "it's",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "while",
    "with",
}
_PREFERENCE_MODE_TERMS = {"emotional", "experimental", "intimate", "personal", "private", "software", "spontaneous"}
_ENVIRONMENT_AVERSION_TERMS = {"chaotic", "crowded", "loud", "noisy", "overwhelmed", "pressure", "pressured", "scrutiny"}
_ENVIRONMENT_PREFERENCE_TERMS = {"calm", "cozy", "library", "peaceful", "quiet", "small"}
_SOCIAL_FEEDBACK_TERMS = {"community", "feedback", "mentor", "peer", "peers", "support"}
_REASON_TERMS = {"because", "decided", "feedback", "overwhelmed", "pressured", "reason", "why"}
_TEMPORAL_TERMS = {"before", "current", "now", "past", "recent", "recently", "started", "stopped"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def informative_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in _STOPWORDS]


def _top_tokens(text: str, *, limit: int = 5) -> str:
    tokens = informative_tokens(text)
    return " ".join(tokens[:limit]) if tokens else text.strip().lower()


def _facet_id(observation: Observation, facet_type: str, facet_value: str) -> str:
    digest = hashlib.sha1(f"{observation.obs_id}|{facet_type}|{facet_value}".encode("utf-8")).hexdigest()[:12]
    return f"facet_{digest}"


def _make_facet(
    observation: Observation,
    *,
    facet_type: str,
    facet_value: str,
    facet_scope: str,
    facet_polarity: str | None = None,
) -> FacetRecord | None:
    normalized = " ".join(informative_tokens(facet_value))
    if not normalized:
        return None
    return FacetRecord(
        facet_id=_facet_id(observation, facet_type, normalized),
        source_observation_id=observation.obs_id,
        relation=observation.relation,
        facet_type=facet_type,
        facet_value=normalized,
        facet_scope=facet_scope,
        facet_polarity=facet_polarity or observation.polarity,
        confidence=observation.confidence,
        metadata={
            "source_relation_family": relation_family(observation.relation),
            "source_canonical_gloss": observation.canonical_gloss,
        },
    )


def facetize_observation(observation: Observation) -> list[FacetRecord]:
    typed = typed_observation(observation)
    text = f"{typed.value} {typed.evidence_text}".lower()
    tokens = set(informative_tokens(text))
    facets: list[FacetRecord] = []
    seen: set[tuple[str, str]] = set()

    def add(record: FacetRecord | None) -> None:
        if record is None:
            return
        key = (record.facet_type, record.facet_value)
        if key in seen:
            return
        seen.add(key)
        facets.append(record)

    if typed.relation.endswith("preference") or typed.relation in {"hobby", "goal"}:
        add(_make_facet(typed, facet_type="preference_target", facet_value=_top_tokens(typed.value), facet_scope="entity"))
        mode_tokens = [token for token in informative_tokens(text) if token in _PREFERENCE_MODE_TERMS]
        if mode_tokens:
            add(_make_facet(typed, facet_type="preference_mode", facet_value=" ".join(mode_tokens[:4]), facet_scope="style"))
    if typed.relation in {"environment_fact", "location"} or tokens & (_ENVIRONMENT_AVERSION_TERMS | _ENVIRONMENT_PREFERENCE_TERMS):
        aversion_tokens = [token for token in informative_tokens(text) if token in _ENVIRONMENT_AVERSION_TERMS]
        if aversion_tokens:
            add(
                _make_facet(
                    typed,
                    facet_type="environment_aversion",
                    facet_value=" ".join(aversion_tokens[:4]),
                    facet_scope="state",
                    facet_polarity="negative",
                )
            )
        preference_tokens = [token for token in informative_tokens(text) if token in _ENVIRONMENT_PREFERENCE_TERMS]
        if preference_tokens:
            add(_make_facet(typed, facet_type="environment_preference", facet_value=" ".join(preference_tokens[:4]), facet_scope="state"))
    if typed.relation in {"social_fact", "reason_fact"} or tokens & _SOCIAL_FEEDBACK_TERMS:
        social_tokens = [token for token in informative_tokens(text) if token in _SOCIAL_FEEDBACK_TERMS]
        if social_tokens:
            add(_make_facet(typed, facet_type="social_feedback", facet_value=" ".join(social_tokens[:4]), facet_scope="reason"))
    if typed.relation in {"reason_fact", "temporal_fact"} or tokens & _REASON_TERMS:
        reason_tokens = [token for token in informative_tokens(text) if token in (_REASON_TERMS | _ENVIRONMENT_AVERSION_TERMS)]
        if reason_tokens:
            add(_make_facet(typed, facet_type="update_reason", facet_value=" ".join(reason_tokens[:5]), facet_scope="reason"))
    if typed.time_scope != "current" or tokens & _TEMPORAL_TERMS:
        temporal_tokens = [token for token in informative_tokens(text) if token in _TEMPORAL_TERMS]
        add(
            _make_facet(
                typed,
                facet_type="temporal_state",
                facet_value=" ".join(temporal_tokens[:4]) if temporal_tokens else typed.time_scope,
                facet_scope="temporal",
            )
        )
    if not facets:
        add(_make_facet(typed, facet_type="generic_fact", facet_value=_top_tokens(typed.value), facet_scope="entity"))
    return facets


def materialize_facet_observation(observation: Observation, facet: FacetRecord) -> Observation:
    payload = typed_observation(observation).to_dict()
    payload["obs_id"] = f"{observation.obs_id}::{facet.facet_id}"
    payload["value"] = facet.facet_value
    payload["canonical_gloss"] = f"{payload['relation']}={facet.facet_value}"
    payload["confidence"] = min(0.98, max(float(payload["confidence"]), facet.confidence))
    metadata = dict(payload.get("metadata", {}) or {})
    metadata.update(
        {
            "facet_id": facet.facet_id,
            "facet_type": facet.facet_type,
            "facet_scope": facet.facet_scope,
            "facet_polarity": facet.facet_polarity,
            "source_observation_id": facet.source_observation_id,
            "facet_match_key": f"{relation_family(facet.relation)}::{facet.facet_type}",
            "facet_schema_version": "v65",
        }
    )
    payload["metadata"] = metadata
    return Observation.from_dict(payload)


def facetize_turn_candidates(
    turn_candidates: dict[str, list[Observation]],
) -> tuple[dict[str, list[Observation]], list[FacetRecord], dict[str, int]]:
    facet_turn_candidates: dict[str, list[Observation]] = {}
    facet_records: list[FacetRecord] = []
    facet_counts: Counter[str] = Counter()
    for key, observations in turn_candidates.items():
        rows: list[Observation] = []
        for observation in observations:
            for facet in facetize_observation(observation):
                facet_records.append(facet)
                facet_counts[facet.facet_type] += 1
                rows.append(materialize_facet_observation(observation, facet))
        facet_turn_candidates[key] = rows
    return facet_turn_candidates, facet_records, dict(facet_counts)


def facetize_candidate_pool(
    candidate_pool: list[CandidateObservation],
) -> tuple[list[CandidateObservation], list[FacetRecord], dict[str, int]]:
    rows: list[CandidateObservation] = []
    facet_records: list[FacetRecord] = []
    facet_counts: Counter[str] = Counter()
    for candidate in candidate_pool:
        for facet in facetize_observation(candidate.observation):
            facet_records.append(facet)
            facet_counts[facet.facet_type] += 1
            rows.append(CandidateObservation(turn=candidate.turn, observation=materialize_facet_observation(candidate.observation, facet)))
    return rows, facet_records, dict(facet_counts)


def facet_schema_payload(facet_records: list[FacetRecord], facet_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "facet_schema_defined": bool(facet_counts),
        "facet_schema_version": "v65",
        "facet_types": sorted(facet_counts),
        "facet_scopes": sorted({facet.facet_scope for facet in facet_records}),
        "facet_type_counts": facet_counts,
        "authoritative_facetizer_in_path": True,
        "facetization_unit": "observation_to_1..k_facets",
    }


def facet_write_utility_score(observation: Observation) -> float:
    metadata = dict(observation.metadata)
    facet_type = str(metadata.get("facet_type", "generic_fact"))
    specificity = min(len(set(informative_tokens(observation.value))) / 5.0, 1.0)
    bonus = {
        "environment_aversion": 0.2,
        "preference_mode": 0.14,
        "preference_target": 0.18,
        "social_feedback": 0.12,
        "update_reason": 0.18,
        "temporal_state": 0.08,
    }.get(facet_type, 0.05)
    polarity_bonus = 0.06 if metadata.get("facet_polarity") == "negative" else 0.0
    return min(1.0, specificity + bonus + polarity_bonus)


def facet_aware_policy_action(observation: Observation, *, policy_label: str, router_action: str) -> str:
    base_action = policy_action_from_router(policy_label=policy_label, router_action=router_action, observation=observation)
    utility = facet_write_utility_score(observation)
    facet_type = str(observation.metadata.get("facet_type", "generic_fact"))
    if facet_type in {"preference_target", "preference_mode", "environment_aversion", "update_reason"}:
        if base_action in {"merge_core", "merge_residual", "overwrite"} and utility >= 0.45:
            return "new_residual"
        if policy_label == "core-worthy" and utility >= 0.62 and base_action in {"merge_core", "new_core"}:
            return "promote_to_core"
    return base_action


def build_v65_memory(
    turns: list[DialogueTurn],
    turn_candidates: dict[str, list[Observation]],
    candidate_pool: list[CandidateObservation],
    *,
    max_stream_observations: int,
) -> tuple[V63WritePolicyBuildResult, dict[str, Any]]:
    if not turns:
        raise ValueError("v6.5 requires at least one dialogue turn.")
    write_worthiness_result = train_write_worthiness_model(turns, turn_candidates)
    attribute_validity_result = train_attribute_validity_model(candidate_pool)
    policy_observations: list[Observation] = []
    write_trace: list[dict[str, Any]] = []
    policy_counts: Counter[str] = Counter()
    facet_policy_counts: Counter[str] = Counter()
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
            policy_label = classify_write_policy(turn_score=turn_score, validity_score=validity_score, observation=candidate)
            policy_counts[policy_label] += 1
            facet_type = str(candidate.metadata.get("facet_type", "generic_fact"))
            facet_policy_counts[f"{facet_type}:{policy_label}"] += 1
            write_trace.append(
                {
                    "event": "candidate_policy",
                    "obs_id": candidate.obs_id,
                    "source_dialogue_id": candidate.source_dialogue_id,
                    "source_turn_id": candidate.source_turn_id,
                    "relation": candidate.relation,
                    "value": candidate.value,
                    "facet_type": facet_type,
                    "turn_score": turn_score,
                    "attribute_validity_score": validity_score,
                    "policy_label": policy_label,
                    "facet_write_utility": facet_write_utility_score(candidate),
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
        raise ValueError("v6.5 retained facet observation stream is too small for training.")
    training_slice = policy_observations[: max(max_stream_observations, 200)]
    router_result = train_write_router(training_slice)
    memory = PersistentCoreResidualMemory()
    weak_but_keep_residual_count = 0
    facet_rerouted_actions = 0
    for idx, observation in enumerate(policy_observations[:max_stream_observations]):
        policy_label = str(observation.metadata.get("write_policy_label", "drop"))
        router_action = route_observation(router_result.router, observation)
        base_action = policy_action_from_router(policy_label=policy_label, router_action=router_action, observation=observation)
        action = facet_aware_policy_action(observation, policy_label=policy_label, router_action=router_action)
        facet_rerouted_actions += int(action != base_action)
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else idx
        written = memory.write(observation, action, turn_index=turn_index, obs_index=idx)
        weak_but_keep_residual_count += int(policy_label == "weak-but-keep" and written is not None and written.bank == "residual")
        write_trace.append(
            {
                "event": "memory_write",
                "obs_id": observation.obs_id,
                "source_dialogue_id": observation.source_dialogue_id,
                "source_turn_id": observation.source_turn_id,
                "facet_type": observation.metadata.get("facet_type"),
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
    policy_eval["facet_aware_write_utility_used"] = True
    state_metrics = {
        **compaction_stats,
        "candidate_observation_count": len(candidate_pool),
        "policy_retained_observation_count": len(policy_observations),
        "weak_but_keep_residual_count": weak_but_keep_residual_count,
        "strict_drop_candidate_count": policy_counts.get("drop", 0),
        "policy_label_counts": dict(policy_counts),
        "facet_policy_label_counts": dict(facet_policy_counts),
        "facet_rerouted_actions": facet_rerouted_actions,
    }
    return (
        V63WritePolicyBuildResult(
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
        ),
        {
            "facet_aware_write_utility_used": True,
            "facet_policy_label_counts": dict(facet_policy_counts),
            "facet_rerouted_actions": facet_rerouted_actions,
        },
    )


def classify_facet_failure(
    *,
    gold_option_text: str,
    all_slots: list[Any],
    active_slots: list[Any],
    selected_glosses: list[str],
    threshold: float,
) -> tuple[str, float, float, float]:
    def best_overlap(text: str, slots: list[Any]) -> float:
        option_tokens = set(informative_tokens(text))
        if not option_tokens:
            return 0.0
        best = 0.0
        for slot in slots:
            slot_tokens = set(informative_tokens(getattr(slot, "canonical_gloss", "")))
            if not slot_tokens:
                continue
            best = max(best, len(option_tokens & slot_tokens) / len(option_tokens | slot_tokens))
        return best

    best_any = best_overlap(gold_option_text, all_slots)
    best_active = best_overlap(gold_option_text, active_slots)
    best_selected = best_overlap(gold_option_text, [type("GlossSlot", (), {"canonical_gloss": gloss})() for gloss in selected_glosses])
    if best_any < threshold:
        return "needed_facet_missing", best_any, best_active, best_selected
    if best_active < threshold:
        return "wrong_sibling_facet_selected", best_any, best_active, best_selected
    if best_selected < threshold:
        return "facet_written_but_reader_missed", best_any, best_active, best_selected
    return "facet_selected_but_decision_failed", best_any, best_active, best_selected
