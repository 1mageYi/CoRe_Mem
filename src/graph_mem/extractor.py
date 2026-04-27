from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import StructuredRecord


@dataclass(slots=True)
class RuleExtractor:
    """
    Non-LLM extractor (MVP).

    This starter version is intentionally conservative: it produces a stable
    event-frame output using lightweight lexical rules and can be replaced later
    by a richer extractor without changing downstream graph code.
    """

    def extract_from_turn(
        self,
        text: str,
        *,
        time_index: int,
        source_turn_id: int,
    ) -> StructuredRecord:
        lowered = text.lower()

        topic = self._detect_topic(lowered)
        update_type = self._detect_update_type(lowered)
        polarity = self._detect_polarity(lowered)
        event_type = self._detect_event_type(update_type)
        value = self._extract_value(text)
        slot = self._detect_slot(topic)

        return StructuredRecord(
            text=text.strip(),
            event_type=event_type,
            topic=topic,
            preference_slot=slot,
            value=value,
            update_type=update_type,
            polarity=polarity,
            time_index=time_index,
            condition_tag="",
            evidence_span=text.strip(),
            source_turn_ids=[source_turn_id],
        )

    @staticmethod
    def _detect_topic(text: str) -> str:
        if any(k in text for k in ("food", "restaurant", "diet", "cuisine")):
            return "food"
        if any(k in text for k in ("movie", "film", "cinema")):
            return "movies"
        if any(k in text for k in ("music", "song", "playlist")):
            return "music"
        if any(k in text for k in ("travel", "trip", "flight", "hotel")):
            return "travel"
        return "general"

    @staticmethod
    def _detect_slot(topic: str) -> str:
        mapping = {
            "food": "food_preference",
            "movies": "movie_preference",
            "music": "music_preference",
            "travel": "travel_preference",
            "general": "other_fact",
        }
        return mapping.get(topic, "other_fact")

    @staticmethod
    def _detect_update_type(text: str) -> str:
        if any(k in text for k in ("no longer", "not anymore", "used to")):
            return "revision"
        if any(k in text for k in ("prefer now", "changed", "shifted", "instead")):
            return "revision"
        if any(k in text for k in ("also", "another", "additionally", "besides")):
            return "update"
        if any(k in text for k in ("conflict", "contradict", "opposite")):
            return "conflict"
        if any(k in text for k in ("like", "love", "prefer", "want")):
            return "new"
        return "unknown"

    @staticmethod
    def _detect_polarity(text: str) -> str:
        if re.search(r"\b(no|not|don't|dislike|hate)\b", text):
            return "negative"
        if re.search(r"\b(love|like|prefer|enjoy)\b", text):
            return "positive"
        return "uncertain"

    @staticmethod
    def _detect_event_type(update_type: str) -> str:
        if update_type == "new":
            return "emergence"
        if update_type in ("update", "revision", "conflict"):
            return "supplement"
        return "unknown"

    @staticmethod
    def _extract_value(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= 180:
            return cleaned
        return cleaned[:180]
