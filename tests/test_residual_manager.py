"""Unit tests for ResidualManager."""

import numpy as np
import pytest

from core_mem.config import MemorySettings
from core_mem.residual_manager import ResidualManager
from core_mem.slot import cosine_similarity


def _rand_emb(dim: int = 384) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def small_cfg() -> MemorySettings:
    return MemorySettings(
        residual_slots=4,
        core_slots=2,
        merge_threshold=0.8,
        promotion_merge_count=3,
        stability_window=2,
        stability_epsilon=0.1,
        recency_weight=1.5,
    )


class TestResidualManager:
    def test_write_creates_new_slot(self, small_cfg):
        rm = ResidualManager(small_cfg)
        emb = _rand_emb()
        slot, promoted = rm.write(emb)
        assert rm.count == 1
        assert not promoted

    def test_write_merges_similar(self, small_cfg):
        rm = ResidualManager(small_cfg)
        emb = _rand_emb()
        rm.write(emb)

        noise = emb + np.random.randn(384).astype(np.float32) * 0.01
        noise = noise / np.linalg.norm(noise)
        slot2, _ = rm.write(noise)

        assert rm.count == 1
        assert slot2.merge_count == 2

    def test_write_creates_new_when_dissimilar(self, small_cfg):
        rm = ResidualManager(small_cfg)
        rm.write(_rand_emb())
        rm.write(_rand_emb())
        assert rm.count == 2

    def test_eviction_when_full(self, small_cfg):
        rm = ResidualManager(small_cfg)
        for _ in range(5):
            rm.write(_rand_emb())
        assert rm.count == small_cfg.residual_slots

    def test_evict_lowest_removes_min_merge_count(self, small_cfg):
        rm = ResidualManager(small_cfg)
        e1 = _rand_emb()
        rm.write(e1)
        rm.slots[0].merge_count = 10

        e2 = _rand_emb()
        rm.write(e2)
        rm.slots[1].merge_count = 1

        evicted = rm.evict_lowest()
        assert evicted.merge_count == 1
        assert rm.count == 1

    def test_find_most_similar_empty(self, small_cfg):
        rm = ResidualManager(small_cfg)
        idx, sim = rm.find_most_similar(_rand_emb())
        assert idx == -1
        assert sim == -1.0

    def test_persistence_roundtrip(self, small_cfg, tmp_path):
        rm = ResidualManager(small_cfg)
        rm.write(_rand_emb(), provenance="s1")
        rm.write(_rand_emb(), provenance="s2")

        rm.save(tmp_path)

        rm2 = ResidualManager(small_cfg)
        rm2.load(tmp_path)

        assert rm2.count == rm.count
        for a, b in zip(rm.slots, rm2.slots):
            assert np.allclose(a.embedding, b.embedding, atol=1e-5)
            assert a.merge_count == b.merge_count
