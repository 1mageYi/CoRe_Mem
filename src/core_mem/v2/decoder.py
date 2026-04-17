"""Belief decoding for the stage-2 latent-memory path."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch

from core_mem.v2.schemas import BeliefItem, BeliefState, SlotRecord

_DATE_VALUE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    re.IGNORECASE,
)
_NUMBER_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_QUERY_INTENT_STOPWORDS = {
    "before",
    "change",
    "changed",
    "current",
    "currently",
    "date",
    "day",
    "former",
    "how",
    "many",
    "much",
    "number",
    "old",
    "previous",
    "what",
    "when",
}


def _infer_query_type(query_text: str) -> str:
    lowered = query_text.lower()
    if " and " in lowered or "both" in lowered:
        return "multi_fact"
    if "current" in lowered or "now" in lowered:
        return "update_sensitive"
    if "when" in lowered or "before" in lowered or "after" in lowered:
        return "temporal"
    return "single_fact"


def _prefers_historical_memory(query_text: str) -> bool:
    lowered = query_text.lower()
    return any(
        phrase in lowered
        for phrase in ("previous", "before", "former", "old ", "old name", "changed", "earlier", "used to")
    )


def _expects_date(query_text: str) -> bool:
    lowered = query_text.lower()
    return lowered.startswith("when ") or "what date" in lowered or "what day" in lowered


def _expects_number(query_text: str) -> bool:
    lowered = query_text.lower()
    return lowered.startswith("how many") or lowered.startswith("how much") or "what number" in lowered


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token not in _QUERY_INTENT_STOPWORDS
    }


@dataclass(frozen=True)
class BeliefDecoder:
    max_items: int = 2

    def decode(
        self,
        query_id: str,
        query_text: str,
        slots: list[SlotRecord],
        composed_memory: list[list[float]] | None = None,
    ) -> BeliefState:
        memory_anchor = self._memory_anchor(composed_memory)
        seen_relations: set[str] = set()
        belief_items: list[BeliefItem] = []
        max_items = self.max_items if _infer_query_type(query_text) == "multi_fact" else 1
        for slot in slots:
            if not slot.active_flag or slot.relation in seen_relations:
                continue
            slot_score = self._slot_score(slot, query_text=query_text, memory_anchor=memory_anchor)
            seen_relations.add(slot.relation)
            belief_items.append(
                BeliefItem(
                    relation=slot.relation,
                    value=self._value_from_gloss(slot.canonical_gloss),
                    status="active" if slot.active_flag else "stale",
                    time_scope=self._time_scope_from_query(query_text, slot),
                    confidence=max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-slot_score)))),
                    support_slot_ids=[slot.slot_id],
                )
            )
            if len(belief_items) >= max_items:
                break
        return BeliefState(
            query_id=query_id,
            entity="user",
            query_type=_infer_query_type(query_text),
            belief_items=belief_items,
            global_consistency="high" if belief_items else "low",
        )

    @staticmethod
    def _value_from_gloss(gloss: str) -> str:
        if "=" in gloss:
            return gloss.split("=", 1)[1].strip()
        return gloss

    @staticmethod
    def _memory_anchor(composed_memory: list[list[float]] | None) -> torch.Tensor | None:
        if not composed_memory:
            return None
        memory_tensor = torch.tensor(composed_memory, dtype=torch.float32)
        if memory_tensor.numel() == 0:
            return None
        anchor = memory_tensor.mean(dim=0)
        norm = torch.linalg.vector_norm(anchor).item()
        if norm <= 0.0:
            return None
        return anchor / norm

    @staticmethod
    def _fit_width(vector: torch.Tensor, width: int) -> torch.Tensor:
        if vector.numel() == width:
            return vector
        if vector.numel() > width:
            return vector[:width]
        return torch.cat([vector, torch.zeros(width - vector.numel(), dtype=vector.dtype)])

    @staticmethod
    def _slot_score(slot: SlotRecord, *, query_text: str, memory_anchor: torch.Tensor | None) -> float:
        token_tensor = torch.tensor(slot.latent_tokens, dtype=torch.float32)
        slot_anchor = token_tensor.mean(dim=0)
        memory_score = 0.0
        if memory_anchor is not None:
            width = max(slot_anchor.numel(), memory_anchor.numel())
            slot_anchor = BeliefDecoder._fit_width(slot_anchor, width)
            aligned_memory = BeliefDecoder._fit_width(memory_anchor, width)
            slot_norm = torch.linalg.vector_norm(slot_anchor).item()
            memory_norm = torch.linalg.vector_norm(aligned_memory).item()
            if slot_norm > 0.0 and memory_norm > 0.0:
                memory_score = float(torch.dot(slot_anchor / slot_norm, aligned_memory / memory_norm))
        lowered_query = query_text.lower()
        relation_hint = float(any(part in lowered_query for part in slot.relation.split("_")))
        query_terms = _content_terms(query_text)
        slot_terms = _content_terms(slot.canonical_gloss.replace("_", " "))
        lexical_hint = (len(query_terms & slot_terms) / len(query_terms)) if query_terms and slot_terms else 0.0
        temporal_role = float(slot.soft_role_scores.temporal)
        stable_role = float(slot.soft_role_scores.stable)
        historical_query = _prefers_historical_memory(lowered_query)
        date_query = _expects_date(lowered_query)
        number_query = _expects_number(lowered_query)

        temporal_hint = 0.0
        if historical_query or date_query:
            temporal_hint += (0.8 * temporal_role) - (0.15 * stable_role)
        elif any(token in lowered_query for token in ("current", "now")):
            temporal_hint += (0.4 * stable_role) - (0.1 * temporal_role)

        value_text = slot.canonical_gloss
        date_hint = 0.9 if lexical_hint > 0.0 and date_query and _DATE_VALUE_RE.search(value_text) else 0.0
        number_hint = 1.0 if lexical_hint > 0.0 and number_query and _NUMBER_VALUE_RE.search(value_text) else 0.0

        if historical_query:
            activity_score = 0.05 if slot.active_flag else -0.05
        else:
            activity_score = 0.2 if slot.active_flag else -0.3
        return (
            (memory_score * 1.4)
            + relation_hint
            + (1.2 * lexical_hint)
            + temporal_hint
            + date_hint
            + number_hint
            + activity_score
            + (slot.confidence * 0.2)
        )

    @staticmethod
    def _time_scope_from_query(query_text: str, slot: SlotRecord) -> str:
        lowered = query_text.lower()
        if _prefers_historical_memory(lowered):
            return "past" if slot.soft_role_scores.temporal >= 0.5 else "current"
        if any(token in lowered for token in ("before", "after", "when")):
            return "past" if slot.soft_role_scores.temporal >= 0.5 else "current"
        return "current"
