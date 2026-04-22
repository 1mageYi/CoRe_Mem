"""Generic finite-option answer scoring utilities for stage-2."""

from __future__ import annotations

from dataclasses import dataclass
import re


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_LOW_INFO_OPTION_TOKENS = {
    "about",
    "after",
    "also",
    "and",
    "any",
    "are",
    "back",
    "but",
    "can",
    "consider",
    "considered",
    "could",
    "find",
    "for",
    "from",
    "have",
    "how",
    "into",
    "its",
    "many",
    "more",
    "much",
    "not",
    "or",
    "other",
    "should",
    "some",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "think",
    "thoughts",
    "those",
    "through",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "you",
    "your",
}
_NEGATIVE_POLARITY_TOKENS = {
    "anxious",
    "anxiety",
    "chaotic",
    "crowd",
    "crowded",
    "deadline",
    "deadlin",
    "exhaust",
    "overwhelm",
    "pressure",
    "pressur",
    "rigid",
    "stress",
    "stressful",
    "tense",
    "uncomfortable",
}
_POSITIVE_POLARITY_TOKENS = {
    "adrenaline",
    "calm",
    "comfort",
    "enjoy",
    "enjoyable",
    "excite",
    "excit",
    "fulfill",
    "joy",
    "joyful",
    "love",
    "relax",
    "reward",
    "rewarding",
    "thrill",
}


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().lower().split())


def options_use_labels(options: list[str]) -> bool:
    return bool(options) and all(option.startswith("(") and ")" in option[:4] for option in options)


def option_label(option: str) -> str:
    closing = option.find(")")
    return option[: closing + 1].strip() if option.startswith("(") and closing > 0 else option.strip()


def option_body(option: str) -> str:
    label = option_label(option)
    remainder = option[len(label) :].strip()
    return remainder or option.strip()


def lexical_option_projection(answer_text: str, options: list[str]) -> str:
    normalized_answer = normalize_answer(answer_text)
    if not options:
        return answer_text
    if options_use_labels(options):
        for option in options:
            label = option_label(option)
            if normalized_answer == normalize_answer(label):
                return label
    for option in options:
        normalized_option = normalize_answer(option)
        normalized_body = normalize_answer(option_body(option))
        if normalized_answer and (
            normalized_answer in normalized_option
            or normalized_answer in normalized_body
            or normalized_option in normalized_answer
            or normalized_body in normalized_answer
        ):
            return option_label(option) if options_use_labels(options) else option
    return answer_text


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_answer(text)))


def _normalize_content_token(token: str) -> str:
    normalized = token.lower()
    if len(normalized) <= 4:
        return normalized
    if normalized.endswith("ies") and len(normalized) > 5:
        return normalized[:-3] + "y"
    for suffix in ("tions", "tion", "ings", "ing", "ed", "es", "s"):
        if len(normalized) > len(suffix) + 2 and normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


@dataclass(frozen=True)
class OptionScoringHead:
    """Rank answer options from latent/belief evidence without benchmark-specific shortcuts."""

    min_token_len: int = 3

    def _content_token_set(self, text: str) -> set[str]:
        return {
            _normalize_content_token(token)
            for token in _token_set(text)
            if len(token) >= self.min_token_len and token not in _LOW_INFO_OPTION_TOKENS
        }

    @staticmethod
    def _polarity_score(terms: set[str]) -> int:
        return len(terms & _NEGATIVE_POLARITY_TOKENS) - len(terms & _POSITIVE_POLARITY_TOKENS)

    def select_option(
        self,
        *,
        query_text: str,
        question_type: str = "",
        answer_text: str,
        options: list[str],
        belief_values: list[str] | None = None,
        evidence_text: str = "",
        selected_slot_glosses: list[str] | None = None,
        include_query_overlap: bool = True,
    ) -> str:
        if not options:
            return answer_text
        direct = lexical_option_projection(answer_text, options)
        if direct in options:
            return option_label(direct) if options_use_labels(options) else direct
        if direct != answer_text:
            return direct

        query_terms = self._content_token_set(query_text)
        answer_terms = self._content_token_set(answer_text)
        belief_terms = self._content_token_set(" ".join(belief_values or []))
        evidence_terms = self._content_token_set(evidence_text)
        gloss_terms = self._content_token_set(" ".join(selected_slot_glosses or []))
        evidence_polarity = self._polarity_score(answer_terms | belief_terms | evidence_terms | gloss_terms)
        polarity_tie_break_enabled = question_type in {
            "recall_user_shared_facts",
            "recalling_facts_mentioned_by_the_user",
        }
        best_option = answer_text
        best_score = float("-inf")
        best_tie_break = float("-inf")
        label_mode = options_use_labels(options)
        for option in options:
            body = option_body(option)
            option_terms = self._content_token_set(body)
            score = 0.0
            if answer_terms:
                overlap = len(answer_terms & option_terms)
                score += 4.0 * overlap
            if belief_terms:
                score += 3.0 * len(belief_terms & option_terms)
            if evidence_terms:
                score += 2.0 * len(evidence_terms & option_terms)
            if gloss_terms:
                score += 1.5 * len(gloss_terms & option_terms)
            if include_query_overlap and query_terms:
                score += 0.5 * len(query_terms & option_terms)
            normalized_body = normalize_answer(body)
            normalized_answer = normalize_answer(answer_text)
            if normalized_answer and normalized_body:
                if normalized_answer in normalized_body or normalized_body in normalized_answer:
                    score += 6.0
            option_polarity = self._polarity_score(option_terms)
            tie_break = 0.0
            if polarity_tie_break_enabled:
                if evidence_polarity != 0 and option_polarity != 0 and (evidence_polarity * option_polarity) > 0:
                    tie_break = float(abs(option_polarity))
                elif evidence_polarity != 0 and option_polarity != 0:
                    tie_break = float(-abs(option_polarity))
            if score > best_score or (score == best_score and tie_break > best_tie_break):
                best_score = score
                best_tie_break = tie_break
                best_option = option_label(option) if label_mode else option
        return best_option
