from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .schemas import StructuredRecord

# --- spaCy NER: lazy-loaded singleton -------------------------------------------
# Kept module-level so the model is loaded once per process and shared.
# If spaCy or the en_core_web_sm model is unavailable, entity extraction silently
# degrades to returning an empty list (entity edges simply won't be built).
_NLP: object = None   # None = not yet attempted; False = failed/unavailable
_NER_LABELS = {"PERSON", "GPE", "LOC", "ORG", "EVENT", "WORK_OF_ART", "NORP"}


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        except Exception:  # model not downloaded or spacy missing
            _NLP = False
    return _NLP if _NLP is not False else None


def extract_entities(text: str) -> list[str]:
    """Return unique lowercase named entity strings from *text* using spaCy NER.

    Returns an empty list when spaCy is unavailable so downstream code works
    regardless of whether the model is installed.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)  # type: ignore[operator]
    seen: set[str] = set()
    out: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in _NER_LABELS:
            continue
        norm = ent.text.strip().lower()
        if len(norm) < 3 or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


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
            entity_mentions=extract_entities(text),
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
