"""Writer: converts text deltas into memory entries via the write path.

Flow: text delta -> encode (MiniLM) -> residual merge -> optional core promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core_mem.config import MemorySettings
from core_mem.core_updater import CoreUpdater
from core_mem.embedding import encode
from core_mem.residual_manager import ResidualManager
from core_mem.slot import MemorySlot

logger = logging.getLogger(__name__)


@dataclass
class WriteResult:
    """Outcome of a single write operation."""

    slot: MemorySlot
    merged: bool
    promoted: bool


class MemoryWriter:
    """Entry point for the write path.

    Accepts text deltas, encodes them, writes to residual, and promotes to
    core when the promotion criteria are met.
    """

    def __init__(
        self,
        settings: MemorySettings | None = None,
        residual: ResidualManager | None = None,
        core: CoreUpdater | None = None,
    ) -> None:
        self.cfg = settings or MemorySettings()
        self.residual = residual or ResidualManager(self.cfg)
        self.core = core or CoreUpdater(self.cfg)

    def write(self, text: str, *, provenance: str = "") -> WriteResult:
        """Write a single text delta through the full pipeline.

        Parameters
        ----------
        text : str
            The memory-worthy text (a fact, preference, event, etc.).
        provenance : str, optional
            Session or turn identifier for traceability.

        Returns
        -------
        WriteResult
            Contains the target slot, whether it was merged (vs. new), and
            whether promotion to core was triggered.
        """
        embedding = encode(text)
        return self.write_embedding(embedding, provenance=provenance)

    def write_embedding(
        self,
        embedding: np.ndarray,
        *,
        provenance: str = "",
    ) -> WriteResult:
        """Write a pre-computed embedding through the pipeline.

        Useful when the caller already has the embedding (e.g. during tests
        or when batching encode calls).
        """
        idx, sim = self.residual.find_most_similar(embedding)
        merged = idx >= 0 and sim >= self.cfg.merge_threshold

        slot, promoted = self.residual.write(embedding, provenance=provenance)

        if promoted:
            self.residual.remove_slot(slot)
            self.core.promote(slot)
            logger.info(
                "Promoted slot to core (merge_count=%d, provenance=%s)",
                slot.merge_count, slot.provenance,
            )

        return WriteResult(slot=slot, merged=merged, promoted=promoted)

    def write_batch(
        self,
        texts: list[str],
        *,
        provenance: str = "",
    ) -> list[WriteResult]:
        """Write multiple text deltas. Encodes in a single batch for speed."""
        embeddings = encode(texts)
        return [
            self.write_embedding(emb, provenance=provenance)
            for emb in embeddings
        ]

    # ------------------------------------------------------------------
    # Persistence (delegates to residual + core)
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save both residual and core state."""
        directory = Path(directory)
        self.residual.save(directory / "residual")
        self.core.save(directory / "core")
        logger.info("Memory state saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        """Load both residual and core state."""
        directory = Path(directory)
        self.residual.load(directory / "residual")
        self.core.load(directory / "core")
        logger.info(
            "Memory state loaded: %d residual, %d core slots",
            self.residual.count, self.core.count,
        )
