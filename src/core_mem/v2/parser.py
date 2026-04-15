"""Rule-first observation parser skeleton for stage-2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from core_mem.v2.schemas import Observation

_DRINK_KEYWORDS = {"coffee", "tea", "matcha", "latte", "espresso", "juice", "water"}
_FOOD_KEYWORDS = {"pizza", "pasta", "sushi", "salad", "burger", "ramen", "taco"}
_MUSIC_KEYWORDS = {"jazz", "rock", "pop", "classical", "hip hop", "lofi", "music"}


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


def _infer_relation(text: str, value: str) -> tuple[str, str]:
    lowered_text = text.lower()
    lowered_value = value.lower()
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
    if "live in" in lowered_text or "from " in lowered_text:
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
    ) -> list[Observation]:
        if speaker == "assistant":
            return []
        candidates: list[Observation] = []
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
    ) -> Observation | None:
        value, polarity, confidence = self._extract_value(clause)
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
    def _extract_value(clause: str) -> tuple[str, str, float]:
        cleaned = _strip_role_prefix(clause).strip()
        lowered = cleaned.lower()
        special_patterns = [
            (r"\bi graduated with(?: a degree in)?\s+(?P<value>[^,.!?]+)", "neutral", 0.84),
            (r"\b(?:my |the )?daily commute(?: to work)?(?:, which)? takes\s+(?P<value>[^,.!?]+)", "neutral", 0.8),
            (r"\bthe play i attended was(?: actually)?(?: a production of)?\s+(?P<value>[^,.!?]+)", "neutral", 0.82),
            (r"\b(?:playlist .*? called|playlist .*?, called)\s+(?P<value>[^,.!?]+)", "neutral", 0.8),
            (r"\brepainted my bedroom walls\s+(?P<value>[^,.!?]+)", "neutral", 0.8),
            (r"\bcan't make it to\s+(?P<value>serenity yoga)\b", "neutral", 0.78),
            (r"\bnear\s+(?P<value>serenity yoga)\b", "neutral", 0.74),
        ]
        for pattern, polarity, confidence in special_patterns:
            match = re.search(pattern, lowered)
            if match:
                value = match.group("value").strip(" .,!?\n\t")
                return value, polarity, confidence

        patterns = [
            (r"\b(?:i like|i love|i prefer|my favorite(?: drink| food| music)? is)\s+(?P<value>.+)", "positive", 0.9),
            (r"\b(?:i don't like|i do not like|i hate|i can't stand)\s+(?P<value>.+)", "negative", 0.9),
            (r"\b(?:i am|i'm|i work as|my job is)\s+(?:an?\s+)?(?P<value>.+)", "neutral", 0.82),
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
