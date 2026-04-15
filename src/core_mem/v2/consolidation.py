"""Explicit consolidation rules for stage-2 memory banks."""

from __future__ import annotations

from dataclasses import dataclass, replace

from core_mem.v2.lifecycle import LifecycleDecision
from core_mem.v2.schemas import SlotRecord

_MULTI_SLOT_RELATIONS = {
    "drink_preference",
    "food_preference",
    "music_preference",
    "hobby",
    "episodic_event",
}


@dataclass(frozen=True)
class ConsolidationManager:
    stable_threshold: float = 0.8

    def apply(
        self,
        core_slots: list[SlotRecord],
        residual_slots: list[SlotRecord],
        decision: LifecycleDecision,
    ) -> tuple[list[SlotRecord], list[SlotRecord]]:
        next_core = list(core_slots)
        next_residual: list[SlotRecord] = []
        for slot in residual_slots:
            should_promote = (
                slot.active_flag
                and slot.relation != "other_fact"
                and (decision.promote or slot.soft_role_scores.stable >= self.stable_threshold)
            )
            if should_promote:
                if slot.relation not in _MULTI_SLOT_RELATIONS:
                    next_core = [existing for existing in next_core if not (existing.active_flag and existing.relation == slot.relation)]
                next_core.append(replace(slot, bank="core"))
            else:
                next_residual.append(slot)
        return next_core, next_residual
