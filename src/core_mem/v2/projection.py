"""Belief-to-answer projection helpers for stage-2."""

from __future__ import annotations

from dataclasses import dataclass

from core_mem.v2.schemas import BeliefState


@dataclass(frozen=True)
class AnswerProjection:
    def render_evidence_block(self, belief: BeliefState) -> str:
        if not belief.belief_items:
            return "No supported belief items."
        return "\n".join(
            f"- {item.relation}: {item.value} ({item.status}, {item.time_scope})"
            for item in belief.belief_items
        )

    def project_answer(self, query_text: str, belief: BeliefState) -> str:
        if not belief.belief_items:
            return "unknown"
        if belief.query_type == "multi_fact":
            return "; ".join(f"{item.relation}={item.value}" for item in belief.belief_items)
        return belief.belief_items[0].value
