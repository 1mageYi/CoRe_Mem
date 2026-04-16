"""End-to-end stage-2 structured memory skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
from typing import Any, Callable

from core_mem.v2.consolidation import ConsolidationManager
from core_mem.v2.decoder import BeliefDecoder
from core_mem.v2.encoder import QueryEncoder, SlotEncoder
from core_mem.v2.lifecycle import LifecycleDecision, LifecycleManager
from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.projection import AnswerProjection
from core_mem.v2.resampler import LightResampler
from core_mem.v2.schemas import BeliefState, Observation, SlotRecord
from core_mem.v2.semantic_outputs import coerce_task_payload
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
    belief_source: str
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
    memory_mode: str = "symbolic"
    use_learned_memory: bool = False
    slot_assignment_mode: str = "symbolic"
    use_learned_slot_assignment: bool = False
    learned_memory_checkpoint_dir: str | None = None
    learned_memory_train_config_path: str | None = None
    learned_memory_device: str = "cpu"
    learned_belief_predictor: Callable[[str, str, list[SlotRecord]], Any] | None = None
    learned_slot_assignment_checkpoint_dir: str | None = None
    learned_slot_assignment_train_config_path: str | None = None
    learned_slot_assignment_device: str = "cpu"
    learned_slot_assignment_predictor: Callable[[Observation, list[SlotRecord]], Any] | None = None
    state: StructuredMemoryState = field(default_factory=StructuredMemoryState)
    _cached_learned_belief_predictor: Callable[[str, str, list[SlotRecord]], Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_learned_slot_assignment_predictor: Callable[[Observation, list[SlotRecord]], Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )

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
        decision = self._decide_slot_assignment(observation, slots)
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
        ranked = sorted(
            self.state.active_slots(),
            key=lambda slot: self._ranking_score(query_vector, query_terms, slot),
            reverse=True,
        )
        selected = ranked[: self.top_k]
        composed = self.resampler.compose(query_vector, selected)
        belief, belief_source = self._decode_belief(query_id, query_text, selected, composed)
        return QueryResult(
            selected_slots=selected,
            composed_memory=composed,
            belief_state=belief,
            belief_source=belief_source,
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

    def _ranking_score(self, query_vector: list[float], query_terms: set[str], slot: SlotRecord) -> float:
        semantic = dot_product(query_vector, slot.retrieval_key)
        if not query_terms:
            return semantic
        slot_terms = self._slot_terms(slot)
        if not slot_terms:
            return semantic
        lexical_overlap = len(query_terms & slot_terms) / len(query_terms)
        return semantic + (1.5 * lexical_overlap)

    def _decode_belief(
        self,
        query_id: str,
        query_text: str,
        selected: list[SlotRecord],
        composed: list[list[float]],
    ) -> tuple[BeliefState, str]:
        if not self._learned_memory_enabled():
            belief = self.decoder.decode(query_id, query_text, selected, composed_memory=composed)
            return belief, "symbolic"

        predictor = self._resolve_learned_belief_predictor()
        if predictor is None:
            return self._empty_learned_belief(query_id), "learned_memory_unavailable"

        try:
            payload = predictor(query_id, query_text, selected)
            learned_belief = self._coerce_learned_belief(payload, query_id=query_id, fallback_slots=selected)
        except Exception:
            return self._empty_learned_belief(query_id), "learned_memory_error"
        if not learned_belief.belief_items:
            return self._empty_learned_belief(query_id), "learned_memory_empty"
        return learned_belief, "learned_memory"

    def _slot_assignment_enabled(self) -> bool:
        return self.use_learned_slot_assignment or self.slot_assignment_mode == "learned"

    def _decide_slot_assignment(
        self,
        observation: Observation,
        slots: list[SlotRecord],
    ) -> LifecycleDecision:
        symbolic_decision = self.lifecycle.decide(observation, slots)
        if not self._slot_assignment_enabled():
            return symbolic_decision

        predictor = self._resolve_slot_assignment_predictor()
        if predictor is None:
            return LifecycleDecision(action="ignore")
        try:
            payload = predictor(observation, slots)
        except Exception:
            return LifecycleDecision(action="ignore")
        return self._coerce_slot_assignment_decision(
            payload,
            observation=observation,
            slots=slots,
            symbolic_decision=symbolic_decision,
        )

    def _learned_memory_enabled(self) -> bool:
        return self.use_learned_memory or self.memory_mode == "learned_memory"

    @staticmethod
    def _slot_assignment_candidates(observation: Observation, slots: list[SlotRecord]) -> list[SlotRecord]:
        return [
            slot
            for slot in slots
            if slot.active_flag and slot.entity == observation.entity and slot.relation == observation.relation
        ]

    def _select_slot_assignment_target(
        self,
        observation: Observation,
        candidates: list[SlotRecord],
        symbolic_decision: LifecycleDecision,
    ) -> SlotRecord | None:
        symbolic_target = self._find_slot(candidates, symbolic_decision.matched_slot_id)
        if symbolic_target is not None:
            return symbolic_target
        if not candidates:
            return None
        observation_terms = set(_TOKEN_RE.findall(observation.value.lower()))
        query_vector = self.query_encoder.encode(observation.value)
        return max(
            candidates,
            key=lambda slot: (
                len(observation_terms & self._slot_terms(slot)),
                dot_product(query_vector, slot.retrieval_key),
            ),
        )

    def _coerce_slot_assignment_decision(
        self,
        payload: Any,
        *,
        observation: Observation,
        slots: list[SlotRecord],
        symbolic_decision: LifecycleDecision,
    ) -> LifecycleDecision:
        if isinstance(payload, str):
            payload = coerce_task_payload("slot_assignment", payload)
        if not isinstance(payload, dict):
            return LifecycleDecision(action="ignore")

        action = str(payload.get("target_action", "")).strip().lower()
        if action not in {"merge", "overwrite", "new", "ignore"}:
            return LifecycleDecision(action="ignore")
        flags = payload.get("target_flags") or {}
        promote = bool(flags.get("promote", False))
        stale_old = bool(flags.get("stale_old", False))
        if action in {"merge", "overwrite"}:
            target = self._select_slot_assignment_target(
                observation,
                self._slot_assignment_candidates(observation, slots),
                symbolic_decision,
            )
            if target is None:
                action = "new"
            else:
                return LifecycleDecision(
                    action=action,
                    matched_slot_id=target.slot_id,
                    promote=promote,
                    stale_old=stale_old if action == "overwrite" else False,
                )
        return LifecycleDecision(action=action, promote=promote, stale_old=stale_old if action == "overwrite" else False)

    def _resolve_learned_belief_predictor(self) -> Callable[[str, str, list[SlotRecord]], Any] | None:
        if self.learned_belief_predictor is not None:
            return self.learned_belief_predictor
        if self._cached_learned_belief_predictor is not None:
            return self._cached_learned_belief_predictor
        checkpoint_dir = self.learned_memory_checkpoint_dir
        train_config_path = self.learned_memory_train_config_path
        if not checkpoint_dir or not train_config_path:
            return None

        import yaml

        from core_mem.v2.training import (
            TrainingExample,
            compact_slot_list,
            generate_prediction_text,
            load_runtime_components,
        )

        config = yaml.safe_load(Path(train_config_path).read_text(encoding="utf-8")) or {}
        model, tokenizer = load_runtime_components(
            config,
            Path(checkpoint_dir),
            device=self.learned_memory_device,
        )
        batching = config.get("training", {}).get("batching", {})
        max_source_length = int(batching.get("max_source_length", 256))
        max_target_length = int(batching.get("max_target_length", 192))

        def _predict(query_id: str, query_text: str, slots: list[SlotRecord]) -> str:
            example = TrainingExample(
                task_name="composition_to_belief",
                input_text=self._render_learned_belief_example(query_text, slots, compact_slot_list),
                target_text=json.dumps({"query_id": query_id, "belief_items": []}, ensure_ascii=False),
            )
            return generate_prediction_text(
                model,
                tokenizer,
                example,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                device=self.learned_memory_device,
            )

        self._cached_learned_belief_predictor = _predict
        return _predict

    def _resolve_slot_assignment_predictor(self) -> Callable[[Observation, list[SlotRecord]], Any] | None:
        if self.learned_slot_assignment_predictor is not None:
            return self.learned_slot_assignment_predictor
        if self._cached_learned_slot_assignment_predictor is not None:
            return self._cached_learned_slot_assignment_predictor

        checkpoint_dir = self.learned_slot_assignment_checkpoint_dir or self.learned_memory_checkpoint_dir
        train_config_path = self.learned_slot_assignment_train_config_path or self.learned_memory_train_config_path
        device = self.learned_slot_assignment_device or self.learned_memory_device
        if not checkpoint_dir or not train_config_path:
            return None

        import yaml

        from core_mem.v2.training import (
            TrainingExample,
            compact_observation_payload,
            compact_slot_list,
            generate_prediction_text,
            load_runtime_components,
        )

        config = yaml.safe_load(Path(train_config_path).read_text(encoding="utf-8")) or {}
        model, tokenizer = load_runtime_components(
            config,
            Path(checkpoint_dir),
            device=device,
        )
        batching = config.get("training", {}).get("batching", {})
        max_source_length = int(batching.get("max_source_length", 256))
        max_target_length = int(batching.get("max_target_length", 192))

        def _predict(observation: Observation, slots: list[SlotRecord]) -> str:
            example = TrainingExample(
                task_name="lifecycle_prediction",
                input_text=self._render_slot_assignment_example(
                    observation,
                    slots,
                    compact_observation_payload,
                    compact_slot_list,
                ),
                target_text=json.dumps(
                    {"target_action": "ignore", "target_flags": {"promote": False, "stale_old": False}},
                    ensure_ascii=False,
                ),
            )
            return generate_prediction_text(
                model,
                tokenizer,
                example,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                device=device,
            )

        self._cached_learned_slot_assignment_predictor = _predict
        return _predict

    @staticmethod
    def _render_learned_belief_example(
        query_text: str,
        slots: list[SlotRecord],
        compact_slot_list: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> str:
        payload = {
            "query": query_text,
            "memory_slots": compact_slot_list([slot.to_dict() for slot in slots]),
        }
        sections = [
            "task: composition_to_belief",
            "instruction: Recover the semantic fields and emit a compact structured object. Semantic correctness matters more than raw JSON surface matching.",
        ]
        for key, value in payload.items():
            sections.append(f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        return "\n".join(sections)

    @staticmethod
    def _render_slot_assignment_example(
        observation: Observation,
        slots: list[SlotRecord],
        compact_observation_payload: Callable[[dict[str, Any]], dict[str, Any]],
        compact_slot_list: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    ) -> str:
        payload = {
            "memory_context": compact_slot_list([slot.to_dict() for slot in slots]),
            "new_observation": compact_observation_payload(observation.to_dict()),
        }
        sections = [
            "task: lifecycle_prediction",
            "instruction: Predict slot_assignment action classification and hard-constraint flags. Recover the semantic fields and emit a compact structured object.",
        ]
        for key, value in payload.items():
            sections.append(f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        return "\n".join(sections)

    @staticmethod
    def _coerce_learned_belief(
        payload: Any,
        *,
        query_id: str,
        fallback_slots: list[SlotRecord],
    ) -> BeliefState:
        if isinstance(payload, BeliefState):
            return payload
        if isinstance(payload, str):
            payload = coerce_task_payload("composition_to_belief", payload)
        if not isinstance(payload, dict):
            raise TypeError("Learned belief payload must be a JSON object, string, or BeliefState.")

        raw_items = payload.get("belief_items", [])
        belief_items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            relation = str(item.get("relation", fallback_slots[index].relation if index < len(fallback_slots) else "other_fact"))
            value = str(item.get("value", ""))
            support_slot_ids = item.get("support_slot_ids")
            if not support_slot_ids:
                support_slot_ids = StructuredMemorySystem._infer_support_slot_ids(
                    fallback_slots,
                    relation=relation,
                    value=value,
                )
            belief_items.append(
                {
                    "relation": relation,
                    "value": value,
                    "status": str(item.get("status", "active")),
                    "time_scope": str(item.get("time_scope", "current")),
                    "confidence": float(item.get("confidence", 0.5)),
                    "support_slot_ids": [str(slot_id) for slot_id in support_slot_ids],
                }
            )

        return BeliefState.from_dict(
            {
                "query_id": str(payload.get("query_id", query_id)),
                "entity": str(payload.get("entity", "user")),
                "query_type": str(payload.get("query_type", "single_fact")),
                "belief_items": belief_items,
                "global_consistency": str(payload.get("global_consistency", "medium" if belief_items else "low")),
            }
        )

    @staticmethod
    def _empty_learned_belief(query_id: str) -> BeliefState:
        return BeliefState.from_dict(
            {
                "query_id": query_id,
                "entity": "user",
                "query_type": "single_fact",
                "belief_items": [],
                "global_consistency": "low",
            }
        )

    @staticmethod
    def _infer_support_slot_ids(slots: list[SlotRecord], *, relation: str, value: str) -> list[str]:
        normalized_value = value.lower().strip()
        for slot in slots:
            if slot.relation == relation:
                return [slot.slot_id]
        for slot in slots:
            if normalized_value and normalized_value in slot.canonical_gloss.lower():
                return [slot.slot_id]
        return [slots[0].slot_id] if slots else []
