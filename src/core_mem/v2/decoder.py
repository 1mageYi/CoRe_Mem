"""Belief decoding for the stage-2 latent-memory path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from core_mem.v2.schemas import BeliefItem, BeliefState, SlotRecord


def _infer_query_type(query_text: str) -> str:
    lowered = query_text.lower()
    if " and " in lowered or "both" in lowered:
        return "multi_fact"
    if "current" in lowered or "now" in lowered:
        return "update_sensitive"
    if "when" in lowered or "before" in lowered or "after" in lowered:
        return "temporal"
    return "single_fact"


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
        current_hint = 0.25 if any(token in lowered_query for token in ("current", "now")) and slot.active_flag else 0.0
        activity_score = 0.25 if slot.active_flag else -0.5
        return (memory_score * 1.4) + relation_hint + current_hint + activity_score + (slot.confidence * 0.2)

    @staticmethod
    def _time_scope_from_query(query_text: str, slot: SlotRecord) -> str:
        lowered = query_text.lower()
        if any(token in lowered for token in ("before", "after", "when")):
            return "past" if not slot.active_flag else "current"
        return "current"
