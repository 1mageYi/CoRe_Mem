"""Residual Manager: manages bounded residual memory slots, handles overflow and eviction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core_mem.config import MemorySettings
from core_mem.slot import MemorySlot, cosine_similarity, online_centroid_merge

logger = logging.getLogger(__name__)


class ResidualManager:
    """Bounded residual memory store with merge, eviction, and promotion."""

    def __init__(self, settings: MemorySettings | None = None) -> None:
        self.cfg = settings or MemorySettings()
        self.slots: list[MemorySlot] = []

    @property
    def capacity(self) -> int:
        return self.cfg.residual_slots

    @property
    def count(self) -> int:
        return len(self.slots)

    @property
    def is_full(self) -> bool:
        return self.count >= self.capacity

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def find_most_similar(self, embedding: np.ndarray) -> tuple[int, float]:
        """Return (index, similarity) of the most similar existing slot.

        Returns (-1, -1.0) when the store is empty.
        """
        if not self.slots:
            return -1, -1.0
        sims = [cosine_similarity(embedding, s.embedding) for s in self.slots]
        idx = int(np.argmax(sims))
        return idx, sims[idx]

    def write(
        self,
        embedding: np.ndarray,
        provenance: str = "",
    ) -> tuple[MemorySlot, bool]:
        """Write a new memory delta into the residual store.

        Returns
        -------
        slot : MemorySlot
            The slot that was written to (merged or newly created).
        promoted : bool
            True if the slot now meets promotion criteria.
        """
        idx, sim = self.find_most_similar(embedding)

        if idx >= 0 and sim >= self.cfg.merge_threshold:
            slot = self.slots[idx]
            online_centroid_merge(slot, embedding, self.cfg.recency_weight)
            logger.debug(
                "Merged into residual slot %d (sim=%.3f, merge_count=%d)",
                idx, sim, slot.merge_count,
            )
        else:
            slot = self._create_slot(embedding, provenance)
            logger.debug("Created new residual slot (count=%d)", self.count)

        promoted = self._check_promotion(slot)
        return slot, promoted

    def evict_lowest(self) -> Optional[MemorySlot]:
        """Remove and return the slot with the lowest merge_count."""
        if not self.slots:
            return None
        idx = int(np.argmin([s.merge_count for s in self.slots]))
        evicted = self.slots.pop(idx)
        logger.debug(
            "Evicted residual slot (merge_count=%d)", evicted.merge_count,
        )
        return evicted

    def remove_slot(self, slot: MemorySlot) -> None:
        """Remove a specific slot instance from the store."""
        try:
            self.slots.remove(slot)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_slot(
        self,
        embedding: np.ndarray,
        provenance: str,
    ) -> MemorySlot:
        if self.is_full:
            self.evict_lowest()
        slot = MemorySlot(embedding=embedding, provenance=provenance)
        self.slots.append(slot)
        return slot

    def _check_promotion(self, slot: MemorySlot) -> bool:
        if slot.merge_count < self.cfg.promotion_merge_count:
            return False
        return slot.is_stable(
            window=self.cfg.stability_window,
            epsilon=self.cfg.stability_epsilon,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist residual slots to *directory* (safetensors + JSON)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not self.slots:
            np.save(directory / "residual_embeddings.npy", np.empty((0, 0)))
            (directory / "residual_meta.json").write_text("[]")
            return

        from safetensors.numpy import save_file
        embeddings = np.stack([s.embedding for s in self.slots])
        save_file({"embeddings": embeddings}, str(directory / "residual.safetensors"))

        meta = [s.to_dict() for s in self.slots]
        (directory / "residual_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
        )

    def load(self, directory: str | Path) -> None:
        """Restore residual slots from *directory*."""
        directory = Path(directory)
        meta_path = directory / "residual_meta.json"
        st_path = directory / "residual.safetensors"

        if not meta_path.exists() or not st_path.exists():
            logger.warning("No residual state found at %s", directory)
            return

        from safetensors.numpy import load_file
        tensors = load_file(str(st_path))
        embeddings = tensors["embeddings"]
        meta = json.loads(meta_path.read_text())

        self.slots = [
            MemorySlot.from_dict(m, embeddings[i])
            for i, m in enumerate(meta)
        ]
        logger.info("Loaded %d residual slots from %s", len(self.slots), directory)
