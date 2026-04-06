"""Core Updater: decides what enters/updates core memory and performs consolidation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from core_mem.config import MemorySettings
from core_mem.slot import MemorySlot, cosine_similarity, online_centroid_merge

logger = logging.getLogger(__name__)


class CoreUpdater:
    """Bounded core memory store with merge and forced-merge eviction."""

    def __init__(self, settings: MemorySettings | None = None) -> None:
        self.cfg = settings or MemorySettings()
        self.slots: list[MemorySlot] = []

    @property
    def capacity(self) -> int:
        return self.cfg.core_slots

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
        """Return (index, similarity) of the most similar core slot.

        Returns (-1, -1.0) when the store is empty.
        """
        if not self.slots:
            return -1, -1.0
        sims = [cosine_similarity(embedding, s.embedding) for s in self.slots]
        idx = int(np.argmax(sims))
        return idx, sims[idx]

    def promote(self, slot: MemorySlot) -> None:
        """Accept a promoted residual slot into core memory.

        If a sufficiently similar core slot exists, merge into it.
        Otherwise create a new core slot (evicting if necessary).
        """
        idx, sim = self.find_most_similar(slot.embedding)

        if idx >= 0 and sim >= self.cfg.core_threshold:
            target = self.slots[idx]
            online_centroid_merge(target, slot.embedding, self.cfg.recency_weight)
            target.salience = max(target.salience, slot.salience)
            logger.debug(
                "Merged promoted slot into core slot %d (sim=%.3f, merge_count=%d)",
                idx, sim, target.merge_count,
            )
        else:
            self._create_slot(slot)

    # ------------------------------------------------------------------
    # Eviction: forced merge into nearest neighbour
    # ------------------------------------------------------------------

    def _create_slot(self, slot: MemorySlot) -> None:
        if self.is_full:
            self._evict_lowest()
        self.slots.append(slot)
        logger.debug("Created new core slot (count=%d)", self.count)

    def _evict_lowest(self) -> None:
        """Evict the core slot with the lowest merge_count by force-merging
        it into its nearest neighbour among the remaining core slots.
        """
        if len(self.slots) < 2:
            if self.slots:
                self.slots.pop(0)
            return

        victim_idx = int(np.argmin([s.merge_count for s in self.slots]))
        victim = self.slots.pop(victim_idx)

        nearest_idx, _ = self.find_most_similar(victim.embedding)
        if nearest_idx >= 0:
            online_centroid_merge(
                self.slots[nearest_idx], victim.embedding, self.cfg.recency_weight,
            )
            logger.debug(
                "Force-merged evicted core slot into core slot %d", nearest_idx,
            )
        else:
            logger.warning("No neighbour to force-merge; evicted core slot dropped")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist core slots to *directory* (safetensors + JSON)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not self.slots:
            np.save(directory / "core_embeddings.npy", np.empty((0, 0)))
            (directory / "core_meta.json").write_text("[]")
            return

        from safetensors.numpy import save_file
        embeddings = np.stack([s.embedding for s in self.slots])
        save_file({"embeddings": embeddings}, str(directory / "core.safetensors"))

        meta = [s.to_dict() for s in self.slots]
        (directory / "core_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
        )

    def load(self, directory: str | Path) -> None:
        """Restore core slots from *directory*."""
        directory = Path(directory)
        meta_path = directory / "core_meta.json"
        st_path = directory / "core.safetensors"

        if not meta_path.exists() or not st_path.exists():
            logger.warning("No core state found at %s", directory)
            return

        from safetensors.numpy import load_file
        tensors = load_file(str(st_path))
        embeddings = tensors["embeddings"]
        meta = json.loads(meta_path.read_text())

        self.slots = [
            MemorySlot.from_dict(m, embeddings[i])
            for i, m in enumerate(meta)
        ]
        logger.info("Loaded %d core slots from %s", len(self.slots), directory)
