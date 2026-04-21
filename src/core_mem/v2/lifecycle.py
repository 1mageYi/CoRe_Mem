"""Lifecycle rules for residual/core stage-2 memory updates."""

from __future__ import annotations

from dataclasses import dataclass

from core_mem.v2.schemas import Observation, SlotRecord

_FACET_RELATIONS = {
    "drink_preference",
    "food_preference",
    "music_preference",
    "hobby",
    "other_fact",
}


@dataclass(frozen=True)
class LifecycleDecision:
    action: str
    matched_slot_id: str | None = None
    promote: bool = False
    stale_old: bool = False


@dataclass(frozen=True)
class LifecycleManager:
    ignore_confidence_below: float = 0.4
    promote_confidence_at_least: float = 0.8

    def decide(self, observation: Observation, slots: list[SlotRecord]) -> LifecycleDecision:
        if observation.confidence < self.ignore_confidence_below:
            return LifecycleDecision(action="ignore")

        active_relation_slots = [
            slot for slot in slots if slot.active_flag and slot.relation == observation.relation and slot.entity == observation.entity
        ]
        for slot in active_relation_slots:
            if observation.value.lower() in slot.canonical_gloss.lower():
                return LifecycleDecision(
                    action="merge",
                    matched_slot_id=slot.slot_id,
                    promote=observation.confidence >= self.promote_confidence_at_least,
                )
            if (
                observation.relation in _FACET_RELATIONS
                and (
                    observation.time_scope == "current"
                    or (observation.relation == "other_fact" and observation.time_scope == "recent_change")
                )
                and slot.canonical_gloss.lower() not in observation.value.lower()
                and observation.value.lower() not in slot.canonical_gloss.lower()
            ):
                return LifecycleDecision(
                    action="new",
                    promote=observation.confidence >= self.promote_confidence_at_least and observation.time_scope == "current",
                )
        if active_relation_slots:
            return LifecycleDecision(
                action="overwrite",
                matched_slot_id=active_relation_slots[0].slot_id,
                stale_old=True,
                promote=observation.confidence >= self.promote_confidence_at_least,
            )
        return LifecycleDecision(
            action="new",
            promote=observation.confidence >= self.promote_confidence_at_least and observation.time_scope == "current",
        )
