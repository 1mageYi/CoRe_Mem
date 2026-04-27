from __future__ import annotations

from .schemas import StructuredRecord


def format_structured_text(record: StructuredRecord) -> str:
    """
    Stable serializer for embedding input.

    Keeping this format deterministic helps reduce noisy embedding variance
    caused by stylistic differences in raw conversation turns.
    """
    return (
        f"[topic={record.topic}]"
        f"[slot={record.preference_slot}]"
        f"[event={record.event_type}]"
        f"[update={record.update_type}]"
        f"[polarity={record.polarity}]"
        f"[time={record.time_index}]"
        f"[cond={record.condition_tag or 'none'}]"
        f" {record.value or record.text}"
    )
