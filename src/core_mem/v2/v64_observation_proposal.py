"""Hybrid learned observation proposer utilities for stage-2 v6.4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import torch

from core_mem.v2.encoder import _lexical_features
from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.schemas import Observation
from core_mem.v2.v62_write_memory import (
    DialogueTurn,
    V62BinaryTrainingResult,
    _non_stopword_tokens,
    _stopword_ratio,
    _token_overlap,
    _train_binary_classifier,
)

_FIRST_PERSON_TERMS = {"i", "i'm", "i've", "me", "my", "mine"}
_PREFERENCE_CUES = {
    "enjoy",
    "favorite",
    "like",
    "love",
    "prefer",
    "passion",
    "music",
    "book",
    "reading",
    "library",
    "festival",
    "software",
    "midi",
}
_STATE_CUES = {
    "feedback",
    "criticism",
    "crowded",
    "chaotic",
    "comfortable",
    "community",
    "deadline",
    "library",
    "overwhelmed",
    "pressured",
    "private",
    "scrutiny",
    "souvenir",
    "travel",
}
_NEGATIVE_CUES = {
    "can't",
    "cannot",
    "chaotic",
    "crowded",
    "criticism",
    "hard",
    "not",
    "overwhelmed",
    "pressure",
    "pressured",
    "scrutiny",
    "stopped",
}
_TEMPORAL_CUES = {
    "again",
    "before",
    "currently",
    "future",
    "now",
    "past",
    "recently",
    "step back",
    "these days",
}
_VALUE_TYPE_BY_RELATION = {
    "constraint": "constraint",
    "goal": "goal",
    "hobby": "preference",
    "location": "location",
    "music_preference": "preference",
    "occupation": "occupation",
}


@dataclass(frozen=True)
class V64ObservationProposerResult:
    threshold: float
    train_examples: int
    eval_examples: int
    accuracy: float
    disabled_accuracy: float
    loss_curve: list[dict[str, float]]
    model: torch.nn.Module


@dataclass(frozen=True)
class V64MergeStats:
    rule_candidate_count: int
    learned_candidate_count: int
    merged_candidate_count: int
    deduped_candidate_count: int


def clause_proposer_features(clause: str, *, context_text: str = "") -> list[float]:
    tokens = re.findall(r"[a-z0-9']+", clause.lower())
    context_tokens = re.findall(r"[a-z0-9']+", context_text.lower())
    non_stop = _non_stopword_tokens(clause)
    lowered = clause.lower()
    cue_terms = set(tokens) & (_PREFERENCE_CUES | _STATE_CUES)
    return [
        *_lexical_features(clause),
        *_lexical_features(context_text),
        min(len(non_stop) / 16.0, 1.0),
        min(len(set(non_stop)) / 16.0, 1.0),
        float(any(token in _FIRST_PERSON_TERMS for token in tokens)),
        min(len(cue_terms) / 4.0, 1.0),
        float(any(cue in lowered for cue in _NEGATIVE_CUES)),
        float(any(cue in lowered for cue in _TEMPORAL_CUES)),
        float("?" in clause),
        _token_overlap(clause, context_text) if context_text else 0.0,
        min(len(set(tokens) & set(context_tokens)) / max(len(set(tokens)), 1), 1.0) if context_text else 0.0,
        1.0 - min(_stopword_ratio(clause), 1.0),
    ]


def silver_clause_positive_label(clause: str, *, matched_rule_observations: list[Observation]) -> bool:
    if matched_rule_observations:
        return True
    lowered = clause.lower()
    non_stop = _non_stopword_tokens(clause)
    if len(non_stop) < 4:
        return False
    if _stopword_ratio(clause) > 0.72:
        return False
    cue_hits = sum(term in lowered for term in (_PREFERENCE_CUES | _STATE_CUES))
    has_first_person = any(token in _FIRST_PERSON_TERMS for token in re.findall(r"[a-z0-9']+", lowered))
    return has_first_person and cue_hits > 0


def train_observation_proposer(
    turns: list[DialogueTurn],
    turn_rule_candidates: dict[str, list[Observation]],
    *,
    parser: Stage2ObservationParser | None = None,
) -> V64ObservationProposerResult:
    parser = parser or Stage2ObservationParser(minimum_confidence=0.0)
    features: list[list[float]] = []
    labels: list[int] = []
    for turn in turns:
        if turn.speaker not in {"user", "system"}:
            continue
        clauses = parser._split_clauses(turn.text)
        rule_candidates = turn_rule_candidates.get(f"{turn.source_dialogue_id}:{turn.source_turn_id}", [])
        for clause in clauses:
            matched = [item for item in rule_candidates if _rule_clause_match(clause, item)]
            features.append(clause_proposer_features(clause, context_text=turn.context_text))
            labels.append(int(silver_clause_positive_label(clause, matched_rule_observations=matched)))
    result = _train_binary_classifier(features, labels, seed=641)
    return V64ObservationProposerResult(
        threshold=result.threshold,
        train_examples=result.train_examples,
        eval_examples=result.eval_examples,
        accuracy=result.accuracy,
        disabled_accuracy=result.disabled_accuracy,
        loss_curve=result.loss_curve,
        model=result.model,
    )


def score_observation_proposer(
    proposer: V64ObservationProposerResult,
    clause: str,
    *,
    context_text: str = "",
) -> float:
    tensor = torch.tensor([clause_proposer_features(clause, context_text=context_text)], dtype=torch.float32)
    with torch.no_grad():
        return float(torch.sigmoid(proposer.model(tensor)).item())


def annotate_rule_candidates(observations: list[Observation]) -> list[Observation]:
    annotated: list[Observation] = []
    for observation in observations:
        metadata = dict(observation.metadata)
        metadata["candidate_provenance"] = sorted(set(metadata.get("candidate_provenance", [])) | {"rule"})
        annotated.append(Observation.from_dict({**observation.to_dict(), "metadata": metadata}))
    return annotated


def build_learned_observation(
    turn: DialogueTurn,
    clause: str,
    *,
    proposer_score: float,
    clause_index: int,
) -> Observation | None:
    value = normalize_learned_value(clause)
    if len(_non_stopword_tokens(value)) < 4:
        return None
    if _stopword_ratio(value) > 0.72:
        return None
    relation = infer_clause_relation(clause)
    time_scope = infer_clause_time_scope(clause)
    polarity = infer_clause_polarity(clause)
    value_type = _VALUE_TYPE_BY_RELATION.get(relation, "other")
    canonical_gloss = f"{relation}={value}"
    digest = hashlib.sha1(
        f"{turn.source_dialogue_id}|{turn.source_turn_id}|learned|{clause_index}|{canonical_gloss}".encode("utf-8")
    ).hexdigest()[:12]
    return Observation(
        obs_id=f"obs_{digest}",
        source_dataset="personamem_raw_context_no_gold",
        source_dialogue_id=turn.source_dialogue_id,
        source_turn_id=turn.source_turn_id,
        session_id=turn.session_id,
        speaker=turn.speaker,
        entity="user",
        relation=relation,
        value=value,
        value_type=value_type,
        time_scope=time_scope,
        status_hint="active" if time_scope != "past" else "stale",
        polarity=polarity,
        confidence=max(0.56, min(0.84, proposer_score)),
        evidence_text=clause.strip(),
        canonical_gloss=canonical_gloss,
        metadata={
            "parser": "learned_observation_proposer_v64",
            "candidate_provenance": ["learned"],
            "proposer_score": float(proposer_score),
            "context_window": "local_last_2_user_turns",
        },
    )


def propose_learned_observations(
    turn: DialogueTurn,
    *,
    parser: Stage2ObservationParser,
    proposer: V64ObservationProposerResult,
    existing_rule_candidates: list[Observation],
) -> list[Observation]:
    observations: list[Observation] = []
    seen_keys = {_normalized_candidate_key(candidate) for candidate in existing_rule_candidates}
    for clause_index, clause in enumerate(parser._split_clauses(turn.text)):
        score = score_observation_proposer(proposer, clause, context_text=turn.context_text)
        if score < proposer.threshold:
            continue
        learned = build_learned_observation(turn, clause, proposer_score=score, clause_index=clause_index)
        if learned is None:
            continue
        key = _normalized_candidate_key(learned)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        observations.append(learned)
    return observations


def merge_candidate_observations(
    rule_candidates: list[Observation],
    learned_candidates: list[Observation],
) -> tuple[list[Observation], V64MergeStats]:
    merged: list[Observation] = []
    deduped = 0
    for candidate in [*annotate_rule_candidates(rule_candidates), *learned_candidates]:
        match_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _candidates_equivalent(existing, candidate)
            ),
            None,
        )
        if match_index is None:
            merged.append(candidate)
            continue
        deduped += 1
        merged[match_index] = _merge_observations(merged[match_index], candidate)
    return merged, V64MergeStats(
        rule_candidate_count=len(rule_candidates),
        learned_candidate_count=len(learned_candidates),
        merged_candidate_count=len(merged),
        deduped_candidate_count=deduped,
    )


def infer_clause_relation(clause: str) -> str:
    lowered = clause.lower()
    if any(token in lowered for token in {"music", "album", "song", "software", "midi", "remix"}):
        return "music_preference"
    if any(token in lowered for token in {"library", "book", "reading", "novel", "story", "book club"}):
        return "hobby"
    if any(token in lowered for token in {"goal", "plan", "want to", "hope to", "aim"}):
        return "goal"
    if any(token in lowered for token in {"work as", "job", "profession"}):
        return "occupation"
    if any(token in lowered for token in {"live in", "from ", "city", "town"}):
        return "location"
    if any(token in lowered for token in {"crowded", "chaotic", "quiet", "library", "festival", "venue", "atmosphere"}):
        return "environment_fact"
    if any(token in lowered for token in {"feedback", "criticism", "peers", "collaboration"}):
        return "social_fact"
    if any(token in lowered for token in {"pressured", "pressure", "overwhelmed", "scrutiny", "deadline"}):
        return "reason_fact"
    return "other_fact"


def infer_clause_time_scope(clause: str) -> str:
    lowered = clause.lower()
    if any(token in lowered for token in {"used to", "previously", "before"}):
        return "past"
    if any(token in lowered for token in {"will", "going to", "plan to", "hope to"}):
        return "future"
    if any(token in lowered for token in {"now", "recently", "these days", "step back", "stopped"}):
        return "recent_change"
    return "current"


def infer_clause_polarity(clause: str) -> str:
    lowered = clause.lower()
    if any(token in lowered for token in _NEGATIVE_CUES):
        return "negative"
    if any(token in lowered for token in {"love", "enjoy", "favorite", "great", "helpful"}):
        return "positive"
    return "neutral"


def normalize_learned_value(clause: str) -> str:
    normalized = clause.strip().lower()
    normalized = re.sub(r"^(?:user|assistant|system)\s*:\s*", "", normalized)
    normalized = re.sub(
        r"^(?:i(?:'m| am| was| have| had| feel| felt| think| know| guess| decided to| want to| hope to)\s+)",
        "",
        normalized,
    )
    normalized = " ".join(normalized.split())
    return normalized.strip(" .,!?:;")


def _rule_clause_match(clause: str, observation: Observation) -> bool:
    if clause.strip().lower() == observation.evidence_text.strip().lower():
        return True
    return _token_overlap(clause, observation.evidence_text) >= 0.8 or _token_overlap(clause, observation.value) >= 0.8


def _normalized_candidate_key(observation: Observation) -> tuple[str, str]:
    normalized_value = " ".join(_non_stopword_tokens(observation.value.lower()))
    return observation.relation, normalized_value


def _candidates_equivalent(left: Observation, right: Observation) -> bool:
    if left.relation != right.relation:
        return False
    left_value = " ".join(_non_stopword_tokens(left.value))
    right_value = " ".join(_non_stopword_tokens(right.value))
    if not left_value or not right_value:
        return False
    if left_value == right_value:
        return True
    if left_value in right_value or right_value in left_value:
        return True
    return _token_overlap(left_value, right_value) >= 0.92


def _merge_observations(left: Observation, right: Observation) -> Observation:
    left_meta = dict(left.metadata)
    right_meta = dict(right.metadata)
    provenance = sorted(set(left_meta.get("candidate_provenance", [])) | set(right_meta.get("candidate_provenance", [])))
    prefer_right = len(right.value) > len(left.value)
    chosen = right if prefer_right else left
    payload = chosen.to_dict()
    payload["confidence"] = max(left.confidence, right.confidence)
    payload["metadata"] = {
        **left_meta,
        **right_meta,
        "candidate_provenance": provenance,
    }
    return Observation.from_dict(payload)
