"""Routing decisions and ingest outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RouteDecision:
    """Router v1 output: either open a new slot or merge into exactly one index."""

    is_new_slot: bool
    """If True, ``slot_index`` is None and ``e_new`` should be stored as a new memory."""

    slot_index: int | None
    """When ``is_new_slot`` is False, the winning candidate index in ``0 .. K-1``."""

    logits: "torch.Tensor | None" = None
    """Optional (K+1,) logits for debugging / calibration (last index = new_slot)."""


@dataclass(frozen=True)
class IngestResult:
    """Result after ``ingest_embedding`` (route + optional compact)."""

    decision: RouteDecision
    memory_updated: bool
    """True if an existing slot was overwritten (merge path)."""

    new_slot_appended: bool
    """True if a new slot was appended."""
