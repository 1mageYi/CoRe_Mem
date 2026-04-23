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
    "education_degree",
    "goal",
    "constraint",
    "relationship",
    "commute_duration",
    "playlist_name",
    "attended_play",
    "paint_color",
    "class_location",
    "episodic_event",
    "recent_change",
    "temporal_fact",
    "reason_fact",
    "environment_fact",
    "social_fact",
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
    "degree": "education_degree",
    "commute": "commute_duration",
    "playlist": "playlist_name",
    "paint": "paint_color",
    "yoga": "class_location",
    "plan": "goal",
    "aim": "goal",
    "limit": "constraint",
    "restriction": "constraint",
    "partner": "relationship",
    "friend": "relationship",
    "event": "episodic_event",
    "change": "recent_change",
    "time": "temporal_fact",
    "reason": "reason_fact",
    "environment": "environment_fact",
    "social": "social_fact",
    "trait": "profile_trait",
}


def normalize_relation(relation: str) -> str:
    normalized = relation.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in DEFAULT_RELATION_SET:
        return normalized
    return RELATION_ALIASES.get(normalized, "other_fact")
