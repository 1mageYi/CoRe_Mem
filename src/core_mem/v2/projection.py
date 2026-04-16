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
_PAGE_NUMBER_RE = re.compile(r"\bpage\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_FREQUENCY_RE = re.compile(
    r"\b(?:once|twice|(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+times?\s+(?:a|per)\s+"
    r"(?:day|week|month|year)|daily|weekly|monthly|yearly|every\s+(?:day|week|month|year))\b",
    re.IGNORECASE,
)


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
        raw_text = self._normalize_whitespace(value)
        if not raw_text:
            return "unknown"
        lowered_query = query_text.strip().lower()
        if lowered_query.startswith("where "):
            text = self._extract_location_phrase(self._strip_explanatory_tail(raw_text))
            text = self._normalize_purchase_source_phrase(lowered_query, text)
        elif lowered_query.startswith("how long"):
            text = self._extract_pattern(self._strip_explanatory_tail(raw_text), _DURATION_RE)
        elif lowered_query.startswith("when ") or "what date" in lowered_query or "what day" in lowered_query:
            text = self._extract_pattern(self._strip_explanatory_tail(raw_text), _DATE_RE)
        elif lowered_query.startswith("how often") or "how frequently" in lowered_query:
            text = self._extract_frequency_phrase(raw_text)
        elif lowered_query.startswith("how many") or "what number" in lowered_query or lowered_query.startswith("how much"):
            text = self._extract_numeric_phrase(lowered_query, raw_text)
        else:
            text = self._strip_explanatory_tail(raw_text)
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

    def _extract_numeric_phrase(self, query_text: str, text: str) -> str:
        if "page" in query_text:
            match = _PAGE_NUMBER_RE.search(text)
            if match:
                return self._normalize_whitespace(match.group(1))
        stripped = self._strip_explanatory_tail(text)
        match = _NUMBER_RE.search(stripped)
        if match:
            return self._normalize_whitespace(match.group(0))
        match = _NUMBER_RE.search(text)
        if match:
            return self._normalize_whitespace(match.group(0))
        return stripped

    def _extract_frequency_phrase(self, text: str) -> str:
        stripped = self._strip_explanatory_tail(text)
        match = _FREQUENCY_RE.search(stripped)
        if match:
            return self._normalize_whitespace(match.group(0).strip(" ,.;"))
        match = _FREQUENCY_RE.search(text)
        if match:
            return self._normalize_whitespace(match.group(0).strip(" ,.;"))
        return stripped

    def _normalize_purchase_source_phrase(self, query_text: str, text: str) -> str:
        if not text:
            return text
        if not any(token in query_text for token in (" buy ", " bought ", " purchase ", " purchased ", " got ")):
            return text
        lowered = text.lower()
        if lowered.startswith("a "):
            remainder = text[2:]
        elif lowered.startswith("an "):
            remainder = text[3:]
        else:
            return text
        if not any(noun in lowered for noun in ("store", "shop", "market", "boutique", "deli")):
            return text
        return self._normalize_whitespace(f"the {remainder}")
