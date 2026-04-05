"""Shared prompt-ready benchmark records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    sample_id: str
    benchmark: str
    prompt: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
