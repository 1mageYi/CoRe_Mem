"""Belief decoding skeleton for stage-2."""

from __future__ import annotations

from dataclasses import dataclass

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

    def decode(self, query_id: str, query_text: str, slots: list[SlotRecord]) -> BeliefState:
        seen_relations: set[str] = set()
        belief_items: list[BeliefItem] = []
        for slot in slots:
            if not slot.active_flag or slot.relation in seen_relations:
                continue
            seen_relations.add(slot.relation)
            belief_items.append(
                BeliefItem(
                    relation=slot.relation,
                    value=self._value_from_gloss(slot.canonical_gloss),
                    status="active" if slot.active_flag else "stale",
                    time_scope="current",
                    confidence=slot.confidence,
                    support_slot_ids=[slot.slot_id],
                )
            )
            if len(belief_items) >= self.max_items:
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
