"""Unit tests for CoreUpdater."""

import numpy as np
import pytest

from core_mem.config import MemorySettings
from core_mem.core_updater import CoreUpdater
from core_mem.slot import MemorySlot


def _rand_emb(dim: int = 384) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def small_cfg() -> MemorySettings:
    return MemorySettings(
        residual_slots=4,
        core_slots=3,
        core_threshold=0.8,
        recency_weight=1.5,
    )


class TestCoreUpdater:
    def test_promote_creates_slot(self, small_cfg):
        cu = CoreUpdater(small_cfg)
        slot = MemorySlot(embedding=_rand_emb(), provenance="test")
        cu.promote(slot)
        assert cu.count == 1

    def test_promote_merges_similar(self, small_cfg):
        cu = CoreUpdater(small_cfg)
        emb = _rand_emb()
        cu.promote(MemorySlot(embedding=emb.copy()))

        noise = emb + np.random.randn(384).astype(np.float32) * 0.01
        noise = noise / np.linalg.norm(noise)
        cu.promote(MemorySlot(embedding=noise))

        assert cu.count == 1
        assert cu.slots[0].merge_count == 2

    def test_promote_creates_new_when_dissimilar(self, small_cfg):
        cu = CoreUpdater(small_cfg)
        cu.promote(MemorySlot(embedding=_rand_emb()))
        cu.promote(MemorySlot(embedding=_rand_emb()))
        assert cu.count == 2

    def test_eviction_force_merges(self, small_cfg):
        cu = CoreUpdater(small_cfg)
        for _ in range(small_cfg.core_slots):
            cu.promote(MemorySlot(embedding=_rand_emb()))
        assert cu.count == small_cfg.core_slots

        cu.promote(MemorySlot(embedding=_rand_emb()))
        assert cu.count == small_cfg.core_slots

    def test_persistence_roundtrip(self, small_cfg, tmp_path):
        cu = CoreUpdater(small_cfg)
        cu.promote(MemorySlot(embedding=_rand_emb(), provenance="s1"))
        cu.promote(MemorySlot(embedding=_rand_emb(), provenance="s2"))

        cu.save(tmp_path)

        cu2 = CoreUpdater(small_cfg)
        cu2.load(tmp_path)

        assert cu2.count == cu.count
        for a, b in zip(cu.slots, cu2.slots):
            assert np.allclose(a.embedding, b.embedding, atol=1e-5)
            assert a.merge_count == b.merge_count
