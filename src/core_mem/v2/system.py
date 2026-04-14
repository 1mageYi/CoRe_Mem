"""End-to-end stage-2 structured memory skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from core_mem.v2.consolidation import ConsolidationManager
from core_mem.v2.decoder import BeliefDecoder
from core_mem.v2.encoder import QueryEncoder, SlotEncoder
from core_mem.v2.lifecycle import LifecycleDecision, LifecycleManager
from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.projection import AnswerProjection
from core_mem.v2.resampler import LightResampler
from core_mem.v2.schemas import BeliefState, Observation, SlotRecord
from core_mem.v2.vector_ops import dot_product


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
        ranked = sorted(
            self.state.active_slots(),
            key=lambda slot: dot_product(query_vector, slot.retrieval_key),
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
