"""End-to-end stage-2 structured memory skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re

from core_mem.v2.consolidation import ConsolidationManager
from core_mem.v2.decoder import BeliefDecoder
from core_mem.v2.encoder import QueryEncoder, SlotEncoder
from core_mem.v2.lifecycle import LifecycleDecision, LifecycleManager
from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.projection import AnswerProjection
from core_mem.v2.resampler import LightResampler
from core_mem.v2.schemas import BeliefState, Observation, SlotRecord
from core_mem.v2.vector_ops import dot_product

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "called",
    "current",
    "currently",
    "did",
    "do",
    "does",
    "exact",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "name",
    "of",
    "on",
    "only",
    "phrase",
    "please",
    "shortest",
    "the",
    "their",
    "they",
    "this",
    "to",
    "user",
    "was",
    "what",
    "where",
    "which",
    "who",
    "with",
}
_RECOMMENDATION_HINTS = {
    "creatively",
    "exploring",
    "idea",
    "ideas",
    "recommend",
    "recommendation",
    "suggest",
}
_REASON_HINTS = {
    "anymore",
    "because",
    "burden",
    "decided",
    "less",
    "longer",
    "pressure",
    "reason",
    "reasons",
    "stopped",
    "why",
}
_CURRENT_HINTS = {"current", "currently", "lately", "now", "recent", "recently", "these", "today"}
_TEMPORAL_HINTS = {"after", "before", "change", "changed", "evolution", "past", "used"}
_SOCIAL_HINTS = {"friend", "friends", "partner", "relationship", "social"}


@dataclass
class StructuredMemoryState:
    core_slots: list[SlotRecord] = field(default_factory=list)
    residual_slots: list[SlotRecord] = field(default_factory=list)

    def active_slots(self) -> list[SlotRecord]:
        return [slot for slot in [*self.core_slots, *self.residual_slots] if slot.active_flag]


@dataclass(frozen=True)
class QueryResult:
    selected_slots: list[SlotRecord]
    composed_memory: list[list[float]]
    belief_state: BeliefState
    evidence_block: str
    answer_text: str


@dataclass
class StructuredMemorySystem:
    parser: Stage2ObservationParser = field(default_factory=Stage2ObservationParser)
    slot_encoder: SlotEncoder = field(default_factory=SlotEncoder)
    query_encoder: QueryEncoder = field(default_factory=QueryEncoder)
    lifecycle: LifecycleManager = field(default_factory=LifecycleManager)
    consolidation: ConsolidationManager = field(default_factory=ConsolidationManager)
    resampler: LightResampler = field(default_factory=LightResampler)
    decoder: BeliefDecoder = field(default_factory=BeliefDecoder)
    projection: AnswerProjection = field(default_factory=AnswerProjection)
    top_k: int = 8
    state: StructuredMemoryState = field(default_factory=StructuredMemoryState)

    def observe_turn(
        self,
        text: str,
        *,
        source_dataset: str,
        source_dialogue_id: str,
        source_turn_id: str,
        session_id: str,
        timestamp: str,
        speaker: str = "user",
    ) -> StructuredMemoryState:
        observations = self.parser.parse_turn(
            text,
            source_dataset=source_dataset,
            source_dialogue_id=source_dialogue_id,
            source_turn_id=source_turn_id,
            session_id=session_id,
            speaker=speaker,
        )
        for observation in observations:
            self.observe_observation(observation, timestamp=timestamp)
        return self.state

    def observe_observation(self, observation: Observation, *, timestamp: str) -> StructuredMemoryState:
        slots = [*self.state.core_slots, *self.state.residual_slots]
        decision = self.lifecycle.decide(observation, slots)
        if decision.action == "ignore":
            return self.state

        next_core = list(self.state.core_slots)
        next_residual = list(self.state.residual_slots)
        matched_slot = self._find_slot(slots, decision.matched_slot_id)
        if decision.action == "merge" and matched_slot is not None:
            updated_slot = self.slot_encoder.encode(
                observation,
                timestamp=timestamp,
                existing_slot=matched_slot,
                bank=matched_slot.bank,
                revision_parent=matched_slot.revision_parent,
            )
            next_core = self._replace_slot(next_core, updated_slot)
            next_residual = self._replace_slot(next_residual, updated_slot)
        else:
            if decision.stale_old and matched_slot is not None:
                stale_slot = replace(matched_slot, active_flag=False)
                next_core = self._replace_slot(next_core, stale_slot)
                next_residual = self._replace_slot(next_residual, stale_slot)
            encoded = self.slot_encoder.encode(
                observation,
                timestamp=timestamp,
                existing_slot=matched_slot if decision.action == "overwrite" else None,
                bank="residual",
                revision_parent=matched_slot.slot_id if matched_slot is not None else None,
            )
            next_residual.append(encoded)

        consolidated_core, consolidated_residual = self.consolidation.apply(next_core, next_residual, decision)
        self.state = StructuredMemoryState(core_slots=consolidated_core, residual_slots=consolidated_residual)
        return self.state

    def query(self, query_id: str, query_text: str) -> QueryResult:
        query_vector = self.query_encoder.encode(query_text)
        query_terms = self._query_terms(query_text)
        role_weights = self._query_role_weights(query_text, query_terms)
        ranked = sorted(
            self.state.active_slots(),
            key=lambda slot: self._ranking_score(query_vector, query_terms, role_weights, slot),
            reverse=True,
        )
        selected = ranked[: self.top_k]
        composed = self.resampler.compose(query_vector, selected)
        belief = self.decoder.decode(query_id, query_text, selected, composed_memory=composed)
        return QueryResult(
            selected_slots=selected,
            composed_memory=composed,
            belief_state=belief,
            evidence_block=self.projection.render_evidence_block(belief),
            answer_text=self.projection.project_answer(query_text, belief),
        )

    @staticmethod
    def _find_slot(slots: list[SlotRecord], slot_id: str | None) -> SlotRecord | None:
        for slot in slots:
            if slot.slot_id == slot_id:
                return slot
        return None

    @staticmethod
    def _replace_slot(slots: list[SlotRecord], replacement: SlotRecord) -> list[SlotRecord]:
        return [replacement if slot.slot_id == replacement.slot_id else slot for slot in slots]

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        return {
            token
            for token in _TOKEN_RE.findall(text.lower())
            if len(token) >= 3 and token not in _QUERY_STOPWORDS
        }

    @staticmethod
    def _slot_terms(slot: SlotRecord) -> set[str]:
        relation_terms = slot.relation.replace("_", " ")
        return {
            token
            for token in _TOKEN_RE.findall(f"{relation_terms} {slot.canonical_gloss}".lower())
            if len(token) >= 3
        }

    @staticmethod
    def _query_role_weights(query_text: str, query_terms: set[str]) -> dict[str, float]:
        lowered = query_text.lower()
        weights = {
            "preference": 0.0,
            "constraint": 0.0,
            "goal": 0.0,
            "temporal": 0.0,
            "social": 0.0,
            "stable": 0.0,
            "generic_penalty": 0.0,
        }
        if query_terms & _RECOMMENDATION_HINTS or "what would you suggest" in lowered:
            weights["preference"] += 0.75
            weights["goal"] += 0.45
            weights["stable"] += 0.15
            weights["generic_penalty"] += 0.25
        if query_terms & _REASON_HINTS or "don't enjoy" in lowered or "do not enjoy" in lowered or "no longer" in lowered:
            weights["constraint"] += 0.8
            weights["temporal"] += 0.25
            weights["stable"] += 0.15
            weights["generic_penalty"] += 0.25
        if query_terms & _CURRENT_HINTS:
            weights["stable"] += 0.35
            weights["temporal"] += 0.1
        if query_terms & _TEMPORAL_HINTS:
            weights["temporal"] += 0.55
        if query_terms & _SOCIAL_HINTS:
            weights["social"] += 0.75
        if {"fact", "facts", "mentioned", "shared"} & query_terms:
            weights["stable"] += 0.2
            weights["generic_penalty"] += 0.1
        return weights

    @staticmethod
    def _role_bonus(role_weights: dict[str, float], slot: SlotRecord) -> float:
        scores = slot.soft_role_scores
        return (
            (role_weights["preference"] * scores.preference)
            + (role_weights["constraint"] * scores.constraint)
            + (role_weights["goal"] * scores.goal)
            + (role_weights["temporal"] * scores.temporal)
            + (role_weights["social"] * scores.social)
            + (role_weights["stable"] * scores.stable)
        ) * 0.45

    @staticmethod
    def _generic_penalty(role_weights: dict[str, float], slot: SlotRecord, lexical_overlap: float) -> float:
        if lexical_overlap > 0.0:
            return 0.0
        penalty = 0.0
        if slot.relation == "other_fact":
            penalty += 0.35 + role_weights["generic_penalty"]
        elif slot.relation == "hobby" and role_weights["constraint"] > 0.0:
            penalty += 0.15
        return penalty

    @staticmethod
    def _relation_specificity_bonus(slot: SlotRecord, lexical_overlap: float) -> float:
        if slot.relation == "other_fact":
            return -0.05 if lexical_overlap == 0.0 else 0.0
        return 0.12 if lexical_overlap > 0.0 else 0.03

    @staticmethod
    def _reason_alignment_adjustment(role_weights: dict[str, float], slot: SlotRecord) -> float:
        if role_weights["constraint"] <= 0.0:
            return 0.0
        return (slot.soft_role_scores.constraint * 0.55) - (slot.soft_role_scores.preference * 0.35)

    def _ranking_score(
        self,
        query_vector: list[float],
        query_terms: set[str],
        role_weights: dict[str, float],
        slot: SlotRecord,
    ) -> float:
        semantic = dot_product(query_vector, slot.retrieval_key)
        if not query_terms:
            return semantic + self._role_bonus(role_weights, slot)
        slot_terms = self._slot_terms(slot)
        if not slot_terms:
            return semantic + self._role_bonus(role_weights, slot)
        lexical_overlap = len(query_terms & slot_terms) / len(query_terms)
        current_bonus = 0.15 if role_weights["stable"] > 0.0 and slot.active_flag else 0.0
        confidence_bonus = slot.confidence * 0.08
        return (
            semantic
            + (1.5 * lexical_overlap)
            + self._role_bonus(role_weights, slot)
            + self._reason_alignment_adjustment(role_weights, slot)
            + self._relation_specificity_bonus(slot, lexical_overlap)
            + current_bonus
            + confidence_bonus
            - self._generic_penalty(role_weights, slot, lexical_overlap)
        )
