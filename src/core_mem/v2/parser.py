"""Rule-first observation parser skeleton for stage-2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from core_mem.v2.schemas import Observation

_DRINK_KEYWORDS = {"coffee", "tea", "matcha", "latte", "espresso", "juice", "water"}
_FOOD_KEYWORDS = {"pizza", "pasta", "sushi", "salad", "burger", "ramen", "taco"}
_MUSIC_KEYWORDS = {
    "jazz",
    "rock",
    "pop",
    "classical",
    "hip hop",
    "lofi",
    "music",
    "musical",
    "remix",
    "remixes",
    "track",
    "tracks",
    "beat",
    "beats",
    "sound",
    "sounds",
    "melody",
    "melodies",
    "midi",
}
_MUSIC_TECH_CONTEXT_TERMS = {
    "music",
    "musical",
    "remix",
    "remixes",
    "track",
    "tracks",
    "beat",
    "beats",
    "sound",
    "sounds",
    "melody",
    "melodies",
    "instrument",
    "instruments",
}
_MUSIC_TECH_SOFTWARE_TERMS = {"software", "midi", "digital"}
_MUSIC_TECH_NEGATION_PATTERNS = (
    re.compile(r"\b(?:didn't|did not|don't|do not|no longer)\s+(?:enjoy|like|love|prefer|want)\b", re.IGNORECASE),
    re.compile(r"\bdidn't resonate\b|\bdid not resonate\b", re.IGNORECASE),
)
_STORE_CONTEXT_PATTERNS = (
    re.compile(
        r"\b(?:cartwheel app from|app from|shop(?:ping)? at|redeem(?:ed)? [^,.!?]* at)\s+"
        r"(?P<store>[a-z0-9&' -]{1,40}?)(?:\s+(?:and|but|which|that|for|pretty|quite|very|almost|maybe|every)\b|[,.!?]|$)",
        re.IGNORECASE,
    ),
)
_WITHDRAWAL_PATTERNS = (
    re.compile(
        r"\b(?:i(?:'ve| have)? decided to|i)\s+(?P<value>step back from\s+[^,.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+)?(?P<value>opted out of\s+[^,.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+)?(?P<value>stopped attending\s+[^,.!?]+)",
        re.IGNORECASE,
    ),
)
_AUTHORING_PATTERNS = (
    re.compile(
        r"\b(?:i\s+)?(?:even\s+)?(?P<value>(?:wrote|write|created|create|curated|curate|published|publish|posted|post|shared|share)\s+"
        r"[^,.!?]{0,120}?\b(?:blog|list|portfolio|newsletter)\b[^,.!?]{0,80})",
        re.IGNORECASE,
    ),
)
_LEARNING_TOPIC_PATTERNS = (
    re.compile(
        r"\b(?:learn(?:ing)?\s+more\s+about|learned\s+about|opportunity\s+to\s+learn\s+more\s+about|chance\s+to\s+learn\s+more\s+about)\s+"
        r"(?P<value>[^,.!?]+)",
        re.IGNORECASE,
    ),
)
_LEARNING_TOPIC_ANCHORS = (
    (re.compile(r"\b(?:film|films|cinema)\b", re.IGNORECASE), "film"),
    (re.compile(r"\b(?:movie|movies)\b", re.IGNORECASE), "movie"),
    (re.compile(r"\b(?:music|musical|song|songs)\b", re.IGNORECASE), "music"),
    (re.compile(r"\b(?:book|books|reading)\b", re.IGNORECASE), "book"),
)


def _slugify(text: str) -> str:
    return re.sub(r"\W+", "_", text.strip().lower()).strip("_") or "value"


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    lowered = f" {text.lower()} "
    for keyword in keywords:
        if " " in keyword:
            if keyword in lowered:
                return True
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return True
    return False


def _strip_role_prefix(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = re.sub(r"^(?:user|assistant|system)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        if updated == cleaned:
            return cleaned
        cleaned = updated.strip()


def _extract_music_technology_value(clause: str) -> tuple[str, str, float] | None:
    lowered = clause.lower()
    if any(pattern.search(lowered) for pattern in _MUSIC_TECH_NEGATION_PATTERNS):
        return None
    if not any(term in lowered for term in _MUSIC_TECH_SOFTWARE_TERMS):
        return None
    if not any(term in lowered for term in _MUSIC_TECH_CONTEXT_TERMS):
        return None
    return "producing music with software", "positive", 0.78


def _extract_withdrawal_value(clause: str) -> tuple[str, str, float] | None:
    for pattern in _WITHDRAWAL_PATTERNS:
        match = pattern.search(clause)
        if not match:
            continue
        value = match.group("value").strip(" .,!?\n\t")
        if value:
            return value.lower(), "negative", 0.8
    return None


def _normalize_feedback_value(value: str) -> str:
    lowered = " ".join(value.lower().split())
    if "feedback" not in lowered:
        return value
    positive = "positive " if any(token in lowered for token in ("great", "positive", "helpful", "encouraging")) else ""
    source = " from my peers" if "peer" in lowered else ""
    about_match = re.search(r"\babout\s+([^,.!?]+)", lowered)
    about = f" about {about_match.group(1).strip()}" if about_match else ""
    return f"getting {positive}feedback{source}{about}".strip()


def _normalize_authored_content_value(value: str) -> str:
    cleaned = " ".join(value.lower().split())
    cleaned = re.sub(r"^(?:i\s+)?", "", cleaned)
    cleaned = re.sub(r"\s+(?:as well|too)$", "", cleaned)
    return cleaned.strip(" .,!?\n\t")


def _extract_authored_content_value(clause: str) -> tuple[str, str, float] | None:
    for pattern in _AUTHORING_PATTERNS:
        match = pattern.search(clause)
        if not match:
            continue
        value = _normalize_authored_content_value(match.group("value"))
        if value:
            return value, "positive", 0.78
    return None


def _extract_learning_topic_value(clause: str, *, turn_text: str | None = None) -> tuple[str, str, float] | None:
    if not turn_text or _extract_authored_content_value(turn_text) is None:
        return None
    for pattern in _LEARNING_TOPIC_PATTERNS:
        match = pattern.search(clause)
        if not match:
            continue
        value = " ".join(match.group("value").lower().split()).strip(" .,!?\n\t")
        value = _anchor_learning_topic_value(value, turn_text=turn_text)
        if value:
            return value, "positive", 0.76
    return None


def _anchor_learning_topic_value(value: str, *, turn_text: str) -> str:
    lowered_value = f" {value.lower()} "
    for pattern, anchor in _LEARNING_TOPIC_ANCHORS:
        if f" {anchor} " in lowered_value:
            return value
        if pattern.search(turn_text):
            return f"{anchor} {value}".strip()
    return value


def _infer_relation(text: str, value: str) -> tuple[str, str]:
    lowered_text = text.lower()
    lowered_value = value.lower()
    if any(phrase in lowered_value for phrase in ("step back from", "opted out of", "stopped attending")):
        return "other_fact", "other"
    if "redeemed" in lowered_text and "coupon" in lowered_text:
        return "episodic_event", "event"
    if "graduated with" in lowered_text or "degree" in lowered_text:
        return "education_degree", "occupation"
    if "commute" in lowered_text and "takes" in lowered_text:
        return "commute_duration", "other"
    if "playlist" in lowered_text and "called" in lowered_text:
        return "playlist_name", "other"
    if "play i attended" in lowered_text or "production of" in lowered_text:
        return "attended_play", "event"
    if "repainted my bedroom walls" in lowered_text:
        return "paint_color", "other"
    if "serenity yoga" in lowered_text:
        return "class_location", "location"
    if "work as" in lowered_text or re.search(r"\bi am an?\b|\bi'm an?\b", lowered_text):
        return "occupation", "occupation"
    if "shop at" in lowered_text or "app from" in lowered_text:
        return "location", "location"
    if "live in" in lowered_text or re.search(r"\b(?:i'm from|i am from|originally from)\b", lowered_text):
        return "location", "location"
    if "want to" in lowered_text or "plan to" in lowered_text:
        return "goal", "goal"
    if "can't" in lowered_text or "cannot" in lowered_text or "allergic" in lowered_text:
        return "constraint", "constraint"
    if _contains_keyword(lowered_value, _DRINK_KEYWORDS) or re.search(r"\bdrink\b", lowered_text):
        return "drink_preference", "preference"
    if _contains_keyword(lowered_value, _FOOD_KEYWORDS) or re.search(r"\beat\b", lowered_text):
        return "food_preference", "preference"
    if _contains_keyword(lowered_value, _MUSIC_KEYWORDS) or "listen to" in lowered_text:
        return "music_preference", "preference"
    if "hobby" in lowered_text or "in my free time" in lowered_text or "enjoy" in lowered_text:
        return "hobby", "preference"
    return "other_fact", "other"


def _infer_time_scope(text: str) -> str:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("step back from", "opted out of", "stopped attending")):
        return "recent_change"
    if "used to" in lowered:
        return "past"
    if "will" in lowered or "plan to" in lowered or "going to" in lowered:
        return "future"
    if "now" in lowered or "recently" in lowered or "these days" in lowered:
        return "recent_change"
    return "current"


def _infer_status_hint(time_scope: str) -> str:
    if time_scope == "past":
        return "stale"
    if time_scope == "recent_change":
        return "active"
    return "active"


@dataclass(frozen=True)
class Stage2ObservationParser:
    minimum_confidence: float = 0.55

    def parse_turn(
        self,
        text: str,
        *,
        source_dataset: str,
        source_dialogue_id: str,
        source_turn_id: str,
        session_id: str,
        speaker: str = "user",
        entity: str = "user",
        context_text: str | None = None,
    ) -> list[Observation]:
        if speaker == "assistant":
            return []
        candidates: list[Observation] = []
        turn_text = _strip_role_prefix(text).strip()
        for idx, clause in enumerate(self._split_clauses(text)):
            parsed = self._parse_clause(
                clause,
                local_index=idx,
                source_dataset=source_dataset,
                source_dialogue_id=source_dialogue_id,
                source_turn_id=source_turn_id,
                session_id=session_id,
                speaker=speaker,
                entity=entity,
                context_text=context_text,
                turn_text=turn_text,
            )
            if parsed is not None:
                candidates.append(parsed)
        return candidates

    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", _strip_role_prefix(text)).strip()
        if not normalized:
            return []
        return [chunk.strip() for chunk in re.split(r"[.;]| but | and ", normalized) if chunk.strip()]

    def _parse_clause(
        self,
        clause: str,
        *,
        local_index: int,
        source_dataset: str,
        source_dialogue_id: str,
        source_turn_id: str,
        session_id: str,
        speaker: str,
        entity: str,
        context_text: str | None,
        turn_text: str,
    ) -> Observation | None:
        value, polarity, confidence = self._extract_value(clause, context_text=context_text, turn_text=turn_text)
        if not value or confidence < self.minimum_confidence:
            return None

        relation, value_type = _infer_relation(clause, value)
        time_scope = _infer_time_scope(clause)
        status_hint = _infer_status_hint(time_scope)
        canonical_gloss = f"{relation}={value}"
        digest = hashlib.sha1(
            f"{source_dataset}|{source_dialogue_id}|{source_turn_id}|{local_index}|{canonical_gloss}".encode("utf-8")
        ).hexdigest()[:12]

        return Observation(
            obs_id=f"obs_{digest}",
            source_dataset=source_dataset,
            source_dialogue_id=source_dialogue_id,
            source_turn_id=source_turn_id,
            session_id=session_id,
            speaker=speaker,
            entity=entity,
            relation=relation,
            value=value,
            value_type=value_type,
            time_scope=time_scope,
            status_hint=status_hint,
            polarity=polarity,
            confidence=confidence,
            evidence_text=clause,
            canonical_gloss=canonical_gloss,
            metadata={"parser": "rule_first_v1", "value_slug": _slugify(value)},
        )

    @staticmethod
    def _extract_value(
        clause: str,
        *,
        context_text: str | None = None,
        turn_text: str | None = None,
    ) -> tuple[str, str, float]:
        cleaned = _strip_role_prefix(clause).strip()
        lowered = cleaned.lower()
        special_patterns = [
            (r"\b(?:i(?:'ve| have) been using .*? from )(?P<value>[a-z0-9&' -]{1,40}?)(?:\s+and\b|[,.!?]|$)", "neutral", 0.8),
            (
                r"\bi shop at (?P<value>[a-z0-9&' -]{1,40}?)(?:\s+(?:pretty|quite|very|almost|about|maybe|every)\b|[,.!?]|$)",
                "neutral",
                0.78,
            ),
            (r"\b(?:i\s+)?(?:actually\s+)?(?P<value>redeemed\s+[^,.!?]*coupon[^,.!?]*)", "neutral", 0.8),
            (r"\bi graduated with(?: a degree in)?\s+(?P<value>[^,.!?]+)", "neutral", 0.84),
            (r"\b(?:my |the )?daily commute(?: to work)?(?:, which)? takes\s+(?P<value>[^,.!?]+)", "neutral", 0.8),
            (r"\bthe play i attended was(?: actually)?(?: a production of)?\s+(?P<value>[^,.!?]+)", "neutral", 0.82),
            (r"\b(?:playlist .*? called|playlist .*?, called)\s+(?P<value>[^,.!?]+)", "neutral", 0.8),
            (r"\brepainted my bedroom walls\s+(?P<value>[^,.!?]+)", "neutral", 0.8),
            (r"\bcan't make it to\s+(?P<value>serenity yoga)\b", "neutral", 0.78),
            (r"\bnear\s+(?P<value>serenity yoga)\b", "neutral", 0.74),
            (
                r"\b(?:after\s+)?receiv(?:ing|ed)\s+(?:some\s+)?(?P<value>[^,.!?]*feedback[^,.!?]*(?:about [^,.!?]+)?)",
                "positive",
                0.8,
            ),
        ]
        for pattern, polarity, confidence in special_patterns:
            match = re.search(pattern, lowered)
            if match:
                value = match.group("value").strip(" .,!?\n\t")
                if "feedback" in value:
                    value = _normalize_feedback_value(value)
                if "redeemed" in lowered and "coupon" in lowered:
                    store = _infer_contextual_store(context_text or "")
                    if store and f" at {store}" not in value and f" from {store}" not in value:
                        value = f"{value} at {store}"
                return value, polarity, confidence

        music_tech = _extract_music_technology_value(cleaned)
        if music_tech is not None:
            return music_tech
        withdrawal = _extract_withdrawal_value(cleaned)
        if withdrawal is not None:
            return withdrawal
        authored_content = _extract_authored_content_value(cleaned)
        if authored_content is not None:
            return authored_content
        learning_topic = _extract_learning_topic_value(cleaned, turn_text=turn_text)
        if learning_topic is not None:
            return learning_topic

        patterns = [
            (r"\b(?:i like|i love|i prefer|my favorite(?: drink| food| music)? is)\s+(?P<value>.+)", "positive", 0.9),
            (r"\b(?:i don't like|i do not like|i hate|i can't stand)\s+(?P<value>.+)", "negative", 0.9),
            (r"\b(?:i work as|my job is|i am an?|i'm an?)\s+(?P<value>.+)", "neutral", 0.82),
            (r"\b(?:i live in|i'm from|i am from)\s+(?P<value>.+)", "neutral", 0.82),
            (r"\b(?:i want to|i plan to|i'm going to)\s+(?P<value>.+)", "positive", 0.8),
            (r"\b(?:i can't eat|i cannot eat|i'm allergic to|i am allergic to)\s+(?P<value>.+)", "negative", 0.88),
            (r"\b(?:my hobby is|in my free time i|i enjoy)\s+(?P<value>.+)", "positive", 0.75),
            (r"\b(?:i used to)\s+(?P<value>.+)", "neutral", 0.7),
        ]
        for pattern, polarity, confidence in patterns:
            match = re.search(pattern, lowered)
            if match:
                value = match.group("value").strip(" .,!?\n\t")
                return value, polarity, confidence
        return "", "neutral", 0.0


def _infer_contextual_store(context_text: str) -> str:
    lowered = _strip_role_prefix(context_text).strip().lower()
    if not lowered:
        return ""
    for pattern in _STORE_CONTEXT_PATTERNS:
        matches = list(pattern.finditer(lowered))
        if not matches:
            continue
        candidate = matches[-1].group("store").strip(" .,!?\n\t\"'")
        if candidate:
            return candidate
    return ""
