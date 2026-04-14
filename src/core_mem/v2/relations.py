"""Relation normalization helpers for stage-2."""

from __future__ import annotations

DEFAULT_RELATION_SET = {
    "drink_preference",
    "food_preference",
    "music_preference",
    "tool_preference",
    "hobby",
    "location",
    "occupation",
    "education",
    "goal",
    "constraint",
    "relationship",
    "episodic_event",
    "recent_change",
    "temporal_fact",
    "profile_trait",
    "other_fact",
}

RELATION_ALIASES = {
    "drink": "drink_preference",
    "beverage": "drink_preference",
    "food": "food_preference",
    "meal": "food_preference",
    "music": "music_preference",
    "song": "music_preference",
    "artist": "music_preference",
    "tool": "tool_preference",
    "city": "location",
    "home": "location",
    "job": "occupation",
    "work": "occupation",
    "profession": "occupation",
    "school": "education",
    "study": "education",
    "plan": "goal",
    "aim": "goal",
    "limit": "constraint",
    "restriction": "constraint",
    "partner": "relationship",
    "friend": "relationship",
    "event": "episodic_event",
    "change": "recent_change",
    "time": "temporal_fact",
    "trait": "profile_trait",
}


def normalize_relation(relation: str) -> str:
    normalized = relation.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in DEFAULT_RELATION_SET:
        return normalized
    return RELATION_ALIASES.get(normalized, "other_fact")
