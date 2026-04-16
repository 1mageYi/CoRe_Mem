"""Belief-to-answer projection helpers for stage-2."""

from __future__ import annotations

from dataclasses import dataclass
import re

from core_mem.v2.schemas import BeliefState


_WHITESPACE_RE = re.compile(r"\s+")
_CLAUSE_SPLIT_RE = re.compile(r"\s+(?:because|but|so|although|though|while)\s+", re.IGNORECASE)
_LOCATION_PHRASE_RE = re.compile(
    r"\b(?:from|at|in|on|to)\s+([^,.]+?)(?=\s+\b(?:from|at|in|on|to)\b|[,.]|$)",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s+(?:minutes?|hours?|days?|weeks?|months?|years?)(?:\s+each\s+way)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b[^,.]*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


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
            return "; ".join(
                f"{item.relation}={self._canonicalize_value(query_text, item.value)}"
                for item in belief.belief_items
            )
        return self._canonicalize_value(query_text, belief.belief_items[0].value)

    def _canonicalize_value(self, query_text: str, value: str) -> str:
        text = self._normalize_whitespace(value)
        if not text:
            return "unknown"
        text = self._strip_explanatory_tail(text)
        lowered_query = query_text.strip().lower()
        if lowered_query.startswith("where "):
            text = self._extract_location_phrase(text)
        elif lowered_query.startswith("how long"):
            text = self._extract_pattern(text, _DURATION_RE)
        elif lowered_query.startswith("when ") or "what date" in lowered_query or "what day" in lowered_query:
            text = self._extract_pattern(text, _DATE_RE)
        elif lowered_query.startswith("how many") or "what number" in lowered_query or lowered_query.startswith("how much"):
            text = self._extract_pattern(text, _NUMBER_RE)
        return text or "unknown"

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text.strip())

    def _strip_explanatory_tail(self, text: str) -> str:
        candidate = text.split(" - ", 1)[0].strip(" ,.;")
        candidate = candidate.split(". ", 1)[0].strip(" ,.;")
        parts = _CLAUSE_SPLIT_RE.split(candidate, maxsplit=1)
        if parts:
            candidate = parts[0].strip(" ,.;")
        return self._normalize_whitespace(candidate)

    def _extract_location_phrase(self, text: str) -> str:
        matches = list(_LOCATION_PHRASE_RE.finditer(text))
        if matches:
            return self._normalize_whitespace(matches[-1].group(1).strip(" ,.;"))
        return text

    def _extract_pattern(self, text: str, pattern: re.Pattern[str]) -> str:
        match = pattern.search(text)
        if match:
            return self._normalize_whitespace(match.group(0).strip(" ,.;"))
        return text
