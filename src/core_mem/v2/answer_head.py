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
    "should",
    "some",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
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


@dataclass(frozen=True)
class OptionScoringHead:
    """Rank answer options from latent/belief evidence without benchmark-specific shortcuts."""

    min_token_len: int = 3
    unsupported_detail_penalty: float = 0.5
    support_density_bonus: float = 4.0
    option_length_penalty: float = 0.3

    def _content_token_set(self, text: str) -> set[str]:
        return {
            token
            for token in _token_set(text)
            if len(token) >= self.min_token_len and token not in _LOW_INFO_OPTION_TOKENS
        }

    def select_option(
        self,
        *,
        query_text: str,
        answer_text: str,
        options: list[str],
        belief_values: list[str] | None = None,
        evidence_text: str = "",
        selected_slot_glosses: list[str] | None = None,
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
        support_terms = query_terms | answer_terms | belief_terms | evidence_terms | gloss_terms
        best_option = answer_text
        best_score = float("-inf")
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
            if query_terms:
                score += 0.5 * len(query_terms & option_terms)
            normalized_body = normalize_answer(body)
            normalized_answer = normalize_answer(answer_text)
            if normalized_answer and normalized_body:
                if normalized_answer in normalized_body or normalized_body in normalized_answer:
                    score += 6.0
            if option_terms and support_terms:
                supported = len(option_terms & support_terms)
                unsupported = len(option_terms - support_terms)
                score += self.support_density_bonus * (supported / len(option_terms))
                score -= self.unsupported_detail_penalty * unsupported
                score -= self.option_length_penalty * len(option_terms)
            if score > best_score:
                best_score = score
                best_option = option_label(option) if label_mode else option
        return best_option
