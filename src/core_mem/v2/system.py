"""End-to-end stage-2 structured memory skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
from typing import Any, Callable

import torch

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
_SLOT_ASSIGNMENT_MAX_TARGET_LENGTH_CAP = 48
_SLOT_ASSIGNMENT_PROMPT_MAX_SLOTS = 12
_SLOT_ASSIGNMENT_PROMPT_MAX_CANDIDATES = 8
_SLOT_ASSIGNMENT_PROMPT_MAX_CONTEXT_SLOTS = 4
_DATE_VALUE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    re.IGNORECASE,
)
_NUMBER_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_RAW_VALUE_STRIP_RE = re.compile(r'^[\s\[\]\{\}",:]+|[\s\[\]\{\}",:]+$')
_STRUCTURAL_VALUE_NOISE_RE = re.compile(r'[\{\}\[\]]|":|",|"{2,}|"{3,}|,\s*"')
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "before",
    "called",
    "change",
    "changed",
    "current",
    "currently",
    "date",
    "did",
    "day",
    "do",
    "does",
    "exact",
    "former",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "long",
    "many",
    "me",
    "much",
    "my",
    "number",
    "of",
    "old",
    "on",
    "only",
    "phrase",
    "please",
    "previous",
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
    "daily",
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
    latent_retriever_checkpoint_dir: str | None = None
    latent_retriever_device: str = "cpu"
    latent_slot_ranker: Callable[[str, list[SlotRecord]], dict[str, float]] | None = None
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
    _cached_latent_slot_ranker: Callable[[str, list[SlotRecord]], dict[str, float]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _cached_learned_slot_assignment_predictor: Callable[[Observation, list[SlotRecord]], Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _recent_turns: list[str] = field(default_factory=list, init=False, repr=False)

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
        recent_context = "\n".join(self._recent_turns[-4:])
        observations = self.parser.parse_turn(
            text,
            source_dataset=source_dataset,
            source_dialogue_id=source_dialogue_id,
            source_turn_id=source_turn_id,
            session_id=session_id,
            speaker=speaker,
            context_text=recent_context,
        )
        self._remember_turn(text, speaker=speaker)
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
        active_slots = self.state.active_slots()
        latent_scores = self._latent_slot_scores(query_text, active_slots)
        scored_slots = [
            (
                slot,
                self._ranking_score(query_vector, query_terms, slot, query_text=query_text),
                self._lexical_overlap(query_terms, slot),
            )
            for slot in active_slots
        ]
        if latent_scores:
            if any(lexical_overlap > 0.0 for _, _, lexical_overlap in scored_slots):
                positive = [item for item in scored_slots if item[2] > 0.0]
                negative = [item for item in scored_slots if item[2] <= 0.0]
                positive.sort(
                    key=lambda item: (
                        item[1],
                        float(latent_scores.get(item[0].slot_id, float("-inf"))),
                        item[2],
                    ),
                    reverse=True,
                )
                negative.sort(
                    key=lambda item: (
                        float(latent_scores.get(item[0].slot_id, float("-inf"))),
                        item[1],
                    ),
                    reverse=True,
                )
                ranked = [slot for slot, _, _ in [*positive, *negative]]
            else:
                ranked = [
                    slot
                    for slot, _, _ in sorted(
                        scored_slots,
                        key=lambda item: (
                            float(latent_scores.get(item[0].slot_id, float("-inf"))),
                            item[1],
                        ),
                        reverse=True,
                    )
                ]
        else:
            ranked = [slot for slot, _, _ in sorted(scored_slots, key=lambda item: item[1], reverse=True)]
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

    def _remember_turn(self, text: str, *, speaker: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        self._recent_turns.append(f"{speaker}: {cleaned}")
        if len(self._recent_turns) > 6:
            self._recent_turns = self._recent_turns[-6:]

    @staticmethod
    def _replace_slot(slots: list[SlotRecord], replacement: SlotRecord) -> list[SlotRecord]:
        return [replacement if slot.slot_id == replacement.slot_id else slot for slot in slots]

    @staticmethod
    def _normalize_term(token: str) -> str:
        normalized = token.lower()
        for suffix in ("ing", "ed", "es", "s"):
            if len(normalized) > len(suffix) + 2 and normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized

    @classmethod
    def _query_terms(cls, text: str) -> set[str]:
        return {
            cls._normalize_term(token)
            for token in _TOKEN_RE.findall(text.lower())
            if len(token) >= 3 and token not in _QUERY_STOPWORDS
        }

    @classmethod
    def _slot_terms(cls, slot: SlotRecord) -> set[str]:
        relation_terms = slot.relation.replace("_", " ")
        return {
            cls._normalize_term(token)
            for token in _TOKEN_RE.findall(f"{relation_terms} {slot.canonical_gloss}".lower())
            if len(token) >= 3
        }

    @classmethod
    def _lexical_overlap(cls, query_terms: set[str], slot: SlotRecord) -> float:
        if not query_terms:
            return 0.0
        slot_terms = cls._slot_terms(slot)
        return (len(query_terms & slot_terms) / len(query_terms)) if slot_terms else 0.0

    @staticmethod
    def _query_prefers_historical_memory(query_text: str) -> bool:
        lowered = query_text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "previous",
                "before",
                "former",
                "old ",
                "old name",
                "changed",
                "earlier",
                "used to",
                "last name before",
            )
        )

    @staticmethod
    def _query_prefers_current_memory(query_text: str) -> bool:
        lowered = query_text.lower()
        return any(token in lowered for token in ("current", "currently", "now", "these days"))

    @staticmethod
    def _query_expects_date(query_text: str) -> bool:
        lowered = query_text.lower()
        return lowered.startswith("when ") or "what date" in lowered or "what day" in lowered

    @staticmethod
    def _query_expects_number(query_text: str) -> bool:
        lowered = query_text.lower()
        return lowered.startswith("how many") or lowered.startswith("how much") or "what number" in lowered

    @staticmethod
    def _slot_has_date_value(slot: SlotRecord) -> bool:
        return bool(_DATE_VALUE_RE.search(slot.canonical_gloss))

    @staticmethod
    def _slot_has_number_value(slot: SlotRecord) -> bool:
        return bool(_NUMBER_VALUE_RE.search(slot.canonical_gloss))

    def _ranking_score(
        self,
        query_vector: list[float],
        query_terms: set[str],
        slot: SlotRecord,
        *,
        query_text: str,
    ) -> float:
        semantic = dot_product(query_vector, slot.retrieval_key)
        lexical_overlap = self._lexical_overlap(query_terms, slot)

        lowered_query = query_text.lower()
        temporal_role = float(slot.soft_role_scores.temporal)
        stable_role = float(slot.soft_role_scores.stable)
        score = semantic + (2.4 * lexical_overlap)

        if self._query_prefers_historical_memory(lowered_query):
            score += (0.85 * temporal_role) - (0.15 * stable_role)
        elif self._query_prefers_current_memory(lowered_query):
            score += (0.45 * stable_role) - (0.1 * temporal_role)

        if lexical_overlap > 0.0 and self._query_expects_date(lowered_query) and self._slot_has_date_value(slot):
            score += 1.1
        if lexical_overlap > 0.0 and self._query_expects_number(lowered_query) and self._slot_has_number_value(slot):
            score += 1.25
        return score

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
            learned_belief = self._coerce_learned_belief(
                payload,
                query_id=query_id,
                query_text=query_text,
                fallback_slots=selected,
            )
        except Exception:
            return self._empty_learned_belief(query_id), "learned_memory_error"
        if not learned_belief.belief_items:
            return self._empty_learned_belief(query_id), "learned_memory_empty"
        return learned_belief, "learned_memory"

    def _slot_assignment_enabled(self) -> bool:
        return self.use_learned_slot_assignment or self.slot_assignment_mode == "learned"

    def _latent_slot_scores(self, query_text: str, slots: list[SlotRecord]) -> dict[str, float]:
        if not slots:
            return {}
        ranker = self._resolve_latent_slot_ranker()
        if ranker is None:
            return {}
        try:
            payload = ranker(query_text, slots)
        except Exception:
            return {}
        return {str(slot_id): float(score) for slot_id, score in payload.items()}

    def _decide_slot_assignment(
        self,
        observation: Observation,
        slots: list[SlotRecord],
    ) -> LifecycleDecision:
        symbolic_decision = self.lifecycle.decide(observation, slots)
        if not self._slot_assignment_enabled():
            return symbolic_decision
        candidates = self._slot_assignment_candidates(observation, slots)
        # When there is no eligible same-relation slot, the learned head cannot
        # legally merge/overwrite anyway; keep the parser/lifecycle cold-start
        # decision and avoid an expensive no-context generation call.
        if not candidates and symbolic_decision.action in {"new", "ignore"}:
            return symbolic_decision
        # `other_fact` is intentionally multi-valued and open-world; for these
        # symbolic `new` writes, learned arbitration adds cost without creating
        # a legal merge target.
        if observation.relation == "other_fact" and symbolic_decision.action == "new":
            return symbolic_decision
        # If the symbolic writer already resolved a unique merge/overwrite
        # target, there is no candidate-ranking ambiguity left for the learned
        # head to solve.
        if symbolic_decision.action in {"merge", "overwrite"} and len(candidates) <= 1:
            return symbolic_decision
        weak_other_fact_overwrite = self._fast_path_weak_other_fact_overwrite(
            observation,
            candidates=candidates,
            symbolic_decision=symbolic_decision,
        )
        if weak_other_fact_overwrite is not None:
            return weak_other_fact_overwrite

        predictor = self._resolve_slot_assignment_predictor()
        if predictor is None:
            return LifecycleDecision(action="ignore")
        prompt_slots = self._slot_assignment_prompt_slots(
            observation,
            slots,
            candidates=candidates,
            symbolic_decision=symbolic_decision,
        )
        try:
            payload = predictor(observation, prompt_slots)
        except Exception:
            return LifecycleDecision(action="ignore")
        return self._coerce_slot_assignment_decision(
            payload,
            observation=observation,
            slots=slots,
            candidates=candidates,
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

    def _slot_assignment_prompt_slots(
        self,
        observation: Observation,
        slots: list[SlotRecord],
        *,
        candidates: list[SlotRecord],
        symbolic_decision: LifecycleDecision,
    ) -> list[SlotRecord]:
        if not slots:
            return []

        prompt_slots: list[SlotRecord] = []
        seen_slot_ids: set[str] = set()

        def _append(slot: SlotRecord | None) -> None:
            if slot is None or slot.slot_id in seen_slot_ids:
                return
            prompt_slots.append(slot)
            seen_slot_ids.add(slot.slot_id)

        symbolic_target = self._find_slot(candidates, symbolic_decision.matched_slot_id)
        _append(symbolic_target)

        observation_terms = set(_TOKEN_RE.findall(observation.value.lower()))
        query_vector = self.query_encoder.encode(observation.value)
        ranked_candidates = sorted(
            candidates,
            key=lambda slot: self._slot_assignment_candidate_priority(
                slot,
                observation_terms=observation_terms,
                query_vector=query_vector,
            ),
            reverse=True,
        )
        for slot in ranked_candidates[:_SLOT_ASSIGNMENT_PROMPT_MAX_CANDIDATES]:
            _append(slot)

        for slot in reversed(slots):
            if len(prompt_slots) >= _SLOT_ASSIGNMENT_PROMPT_MAX_SLOTS:
                break
            if not slot.active_flag:
                continue
            _append(slot)
            if len(prompt_slots) >= len(ranked_candidates) + _SLOT_ASSIGNMENT_PROMPT_MAX_CONTEXT_SLOTS:
                break
        return prompt_slots[:_SLOT_ASSIGNMENT_PROMPT_MAX_SLOTS]

    def _slot_assignment_candidate_priority(
        self,
        slot: SlotRecord,
        *,
        observation_terms: set[str],
        query_vector: list[float],
    ) -> tuple[float, float, float]:
        lexical_overlap = len(observation_terms & self._slot_terms(slot))
        semantic_overlap = dot_product(query_vector, slot.retrieval_key)
        return lexical_overlap, semantic_overlap, float(slot.confidence)

    def _fast_path_weak_other_fact_overwrite(
        self,
        observation: Observation,
        *,
        candidates: list[SlotRecord],
        symbolic_decision: LifecycleDecision,
    ) -> LifecycleDecision | None:
        if observation.relation != "other_fact" or symbolic_decision.action != "overwrite":
            return None
        target = self._select_slot_assignment_target(observation, candidates, symbolic_decision)
        if target is None:
            return LifecycleDecision(action="new", promote=symbolic_decision.promote)
        observation_terms = set(_TOKEN_RE.findall(observation.value.lower()))
        lexical_overlap = len(observation_terms & self._slot_terms(target))
        normalized_value = observation.value.lower().strip()
        normalized_target = target.canonical_gloss.lower().strip()
        if lexical_overlap >= 3 or normalized_value in normalized_target or normalized_target in normalized_value:
            return None
        return LifecycleDecision(action="new", promote=symbolic_decision.promote)

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
        candidates: list[SlotRecord],
        symbolic_decision: LifecycleDecision,
    ) -> LifecycleDecision:
        from core_mem.v2.training import (
            _repair_lifecycle_payload_from_input_context,
            compact_observation_payload,
            compact_slot_list,
        )

        input_text = self._render_slot_assignment_example(
            observation,
            slots,
            compact_observation_payload,
            compact_slot_list,
        )
        raw_text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if isinstance(payload, str):
            payload = coerce_task_payload("slot_assignment", payload)
        payload = _repair_lifecycle_payload_from_input_context(
            input_text,
            raw_text,
            payload if isinstance(payload, dict) else None,
        )
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
                candidates,
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

    def _resolve_latent_slot_ranker(self) -> Callable[[str, list[SlotRecord]], dict[str, float]] | None:
        if self.latent_slot_ranker is not None:
            return self.latent_slot_ranker
        if self._cached_latent_slot_ranker is not None:
            return self._cached_latent_slot_ranker
        checkpoint_dir = self.latent_retriever_checkpoint_dir
        if not checkpoint_dir:
            return None
        checkpoint_path = Path(checkpoint_dir) / "latent_retriever.pt"
        if not checkpoint_path.exists():
            return None

        from core_mem.v2.encoder import _lexical_features
        from core_mem.v2.latent_training import TrainableLatentRetriever, _slot_features

        state_dict = torch.load(checkpoint_path, map_location=self.latent_retriever_device, weights_only=True)
        hidden_dim, feature_dim = state_dict["query_encoder.0.weight"].shape
        latent_dim, _ = state_dict["query_encoder.2.weight"].shape
        latent_queries = state_dict["latent_query_bank"].shape[0]
        model = TrainableLatentRetriever(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            latent_queries=latent_queries,
        ).to(self.latent_retriever_device)
        model.load_state_dict(state_dict)
        model.eval()

        def _rank(query_text: str, slots: list[SlotRecord]) -> dict[str, float]:
            if not slots:
                return {}
            query_features = torch.tensor(
                [_lexical_features(query_text)],
                dtype=torch.float32,
                device=self.latent_retriever_device,
            )
            candidate_features = torch.tensor(
                [
                    [
                        _slot_features(
                            {
                                "relation": slot.relation,
                                "canonical_gloss": slot.canonical_gloss,
                                "bank": slot.bank,
                                "entity": slot.entity,
                            }
                        )
                        for slot in slots
                    ]
                ],
                dtype=torch.float32,
                device=self.latent_retriever_device,
            )
            with torch.no_grad():
                logits = model(
                    query_features=query_features,
                    candidate_features=candidate_features,
                )[0].detach().cpu().tolist()
            return {slot.slot_id: float(logit) for slot, logit in zip(slots, logits)}

        self._cached_latent_slot_ranker = _rank
        return _rank

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
        max_target_length = min(
            int(batching.get("max_target_length", 192)),
            _SLOT_ASSIGNMENT_MAX_TARGET_LENGTH_CAP,
        )

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
            (
                "instruction: Recover the semantic fields and emit a compact structured object. "
                "Semantic correctness matters more than raw JSON surface matching. "
                "The memory_slots are already ordered from most query-relevant to least query-relevant. "
                "If one of the top-ranked slots directly answers the query, copy that slot's relation and value "
                "instead of switching to a generic relation such as other_fact."
            ),
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
            (
                "instruction: Predict slot_assignment action classification and hard-constraint flags. "
                "Choose exactly one target_action from [merge, overwrite, new, ignore]. "
                'Return JSON only with the schema {"target_action":"new","target_flags":{"promote":false,"stale_old":false}}. '
                "Recover the semantic fields and do not copy input fields into the output."
            ),
        ]
        for key, value in payload.items():
            sections.append(f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        return "\n".join(sections)

    @staticmethod
    def _coerce_learned_belief(
        payload: Any,
        *,
        query_id: str,
        query_text: str,
        fallback_slots: list[SlotRecord],
    ) -> BeliefState:
        from core_mem.v2.training import _repair_belief_payload_from_input_context, compact_slot_list

        if isinstance(payload, BeliefState):
            return payload
        input_text = StructuredMemorySystem._render_learned_belief_example(
            query_text,
            fallback_slots,
            compact_slot_list,
        )
        raw_text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if isinstance(payload, str):
            payload = coerce_task_payload("composition_to_belief", payload)
        payload = _repair_belief_payload_from_input_context(
            input_text,
            raw_text,
            payload if isinstance(payload, dict) else None,
        )
        if not isinstance(payload, dict):
            payload = StructuredMemorySystem._fallback_belief_payload_from_raw_text(
                raw_text,
                query_id=query_id,
                query_text=query_text,
                fallback_slots=fallback_slots,
            )
        if not isinstance(payload, dict):
            raise TypeError("Learned belief payload must be a JSON object, string, or BeliefState.")

        raw_items = payload.get("belief_items", [])
        fallback_payload = StructuredMemorySystem._fallback_belief_payload_from_raw_text(
            raw_text,
            query_id=query_id,
            query_text=query_text,
            fallback_slots=fallback_slots,
        )
        fallback_item = None
        if isinstance(fallback_payload, dict):
            candidate_items = fallback_payload.get("belief_items", [])
            if isinstance(candidate_items, list) and candidate_items and isinstance(candidate_items[0], dict):
                fallback_item = candidate_items[0]
        available_relations = {slot.relation for slot in fallback_slots}
        belief_items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            relation = str(item.get("relation", fallback_slots[index].relation if index < len(fallback_slots) else "other_fact"))
            value = str(item.get("value", ""))
            support_slot_ids = item.get("support_slot_ids")
            if relation not in available_relations and isinstance(fallback_item, dict):
                relation = str(fallback_item.get("relation", relation))
                fallback_support_ids = fallback_item.get("support_slot_ids")
                if fallback_support_ids:
                    support_slot_ids = fallback_support_ids
                fallback_value = str(fallback_item.get("value", ""))
                if fallback_value:
                    value = fallback_value
            if not support_slot_ids:
                support_slot_ids = StructuredMemorySystem._infer_support_slot_ids(
                    fallback_slots,
                    relation=relation,
                    value=value,
                )
            support_slot = StructuredMemorySystem._select_belief_support_slot(
                fallback_slots,
                relation=relation,
                support_slot_ids=[str(slot_id) for slot_id in support_slot_ids],
            )
            if StructuredMemorySystem._should_backfill_belief_value(
                value,
                relation=relation,
                support_slot=support_slot,
            ):
                value = StructuredMemorySystem._canonical_slot_value(support_slot)
            else:
                value = StructuredMemorySystem._sanitize_belief_value(value, relation=relation)
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

    @staticmethod
    def _canonical_slot_value(slot: SlotRecord | None) -> str:
        if slot is None:
            return ""
        canonical_gloss = slot.canonical_gloss.strip()
        if "=" in canonical_gloss:
            return canonical_gloss.split("=", 1)[1].strip()
        return canonical_gloss

    @classmethod
    def _select_belief_support_slot(
        cls,
        slots: list[SlotRecord],
        *,
        relation: str,
        support_slot_ids: list[str],
    ) -> SlotRecord | None:
        for slot_id in support_slot_ids:
            slot = cls._find_slot(slots, slot_id)
            if slot is not None:
                return slot
        for slot in slots:
            if slot.relation == relation:
                return slot
        return slots[0] if slots else None

    @classmethod
    def _sanitize_belief_value(cls, value: str, *, relation: str) -> str:
        cleaned = _RAW_VALUE_STRIP_RE.sub("", str(value or "")).strip()
        if not cleaned:
            return ""
        if "=" not in cleaned:
            return cleaned
        prefix, remainder = cleaned.split("=", 1)
        if prefix.strip().lower() == relation.strip().lower():
            normalized = _RAW_VALUE_STRIP_RE.sub("", remainder).strip()
            return normalized or cleaned
        return cleaned

    @classmethod
    def _belief_value_terms(cls, value: str) -> set[str]:
        return {
            cls._normalize_term(token)
            for token in _TOKEN_RE.findall(str(value or "").lower())
            if len(token) >= 3
        }

    @classmethod
    def _belief_value_grounded_in_support_slot(
        cls,
        value: str,
        *,
        support_slot: SlotRecord | None,
    ) -> bool:
        if support_slot is None:
            return False
        cleaned = cls._normalize_whitespace(value).lower()
        canonical_value = cls._normalize_whitespace(cls._canonical_slot_value(support_slot)).lower()
        canonical_gloss = cls._normalize_whitespace(support_slot.canonical_gloss).lower()
        if not cleaned:
            return False
        if cleaned == canonical_value:
            return True
        if cleaned in canonical_value or cleaned in canonical_gloss:
            return True
        if canonical_value and canonical_value in cleaned:
            return True
        cleaned_terms = cls._belief_value_terms(cleaned)
        canonical_terms = cls._belief_value_terms(canonical_value)
        if not cleaned_terms or not canonical_terms:
            return False
        overlap = cleaned_terms & canonical_terms
        if len(overlap) >= 3:
            return True
        return (len(overlap) / len(cleaned_terms)) >= 0.5

    @staticmethod
    def _belief_value_looks_abstractive(value: str) -> bool:
        cleaned = " ".join(str(value or "").split())
        if not cleaned:
            return False
        token_count = len(_TOKEN_RE.findall(cleaned.lower()))
        if token_count <= 1:
            return True
        if token_count >= 8:
            return True
        return any(marker in cleaned for marker in (",", ";", "."))

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join(str(text or "").split())

    @classmethod
    def _should_backfill_belief_value(
        cls,
        value: str,
        *,
        relation: str,
        support_slot: SlotRecord | None,
    ) -> bool:
        cleaned = cls._sanitize_belief_value(value, relation=relation)
        if not cleaned:
            return True
        if _STRUCTURAL_VALUE_NOISE_RE.search(cleaned):
            return True
        if cleaned.strip().lower() == relation.strip().lower():
            return True
        if "=" in cleaned:
            prefix = cleaned.split("=", 1)[0].strip().lower()
            if prefix and prefix != relation.strip().lower():
                return True
        if support_slot is None:
            return False
        canonical_value = cls._canonical_slot_value(support_slot)
        if not canonical_value:
            return False
        if cleaned.lower() == canonical_value.lower():
            return False
        grounded = cls._belief_value_grounded_in_support_slot(cleaned, support_slot=support_slot)
        same_relation = support_slot.relation.strip().lower() == relation.strip().lower()
        # Learned belief values can legitimately be a normalized or query-shaped
        # projection of the support slot content (for example extracting
        # "johnson" from an `other_fact` clause), and some same-relation heads
        # can emit concise normalized values that do not literally occur in the
        # support slot. Only force a same-relation backfill when the decoded
        # text both loses grounding and looks like an abstractive/noisy clause.
        if same_relation:
            return (not grounded) and cls._belief_value_looks_abstractive(cleaned)
        return not grounded

    @classmethod
    def _fallback_slot_rank(
        cls,
        slot: SlotRecord,
        *,
        query_text: str,
        raw_value: str,
        index: int,
    ) -> tuple[float, float, float, float, float, float]:
        query_terms = cls._query_terms(query_text)
        slot_terms = cls._slot_terms(slot)
        query_overlap = float(len(query_terms & slot_terms))
        raw_match = float(bool(raw_value) and raw_value.lower() in slot.canonical_gloss.lower())
        preference_bonus = float(slot.relation.endswith("_preference") or slot.relation in {"hobby", "goal"})
        non_other_fact = float(slot.relation != "other_fact")
        return (
            raw_match,
            query_overlap,
            preference_bonus,
            non_other_fact,
            float(slot.active_flag),
            float(slot.confidence) - (index * 0.001),
        )

    @staticmethod
    def _fallback_belief_payload_from_raw_text(
        raw_text: str,
        *,
        query_id: str,
        query_text: str,
        fallback_slots: list[SlotRecord],
    ) -> dict[str, Any] | None:
        cleaned = _RAW_VALUE_STRIP_RE.sub("", str(raw_text or "")).strip()
        if not fallback_slots:
            return None
        ranked_slots = sorted(
            enumerate(fallback_slots),
            key=lambda item: StructuredMemorySystem._fallback_slot_rank(
                item[1],
                query_text=query_text,
                raw_value=cleaned,
                index=item[0],
            ),
            reverse=True,
        )
        if not ranked_slots:
            return None
        _, best_slot = ranked_slots[0]
        canonical_value = StructuredMemorySystem._canonical_slot_value(best_slot)
        if cleaned and cleaned.lower() in best_slot.canonical_gloss.lower():
            value = cleaned
        else:
            value = canonical_value or cleaned
        if not value:
            return None
        return {
            "query_id": query_id,
            "entity": "user",
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": best_slot.relation,
                    "value": value,
                    "support_slot_ids": [best_slot.slot_id],
                }
            ],
            "global_consistency": "medium",
        }
