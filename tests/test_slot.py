"""Unit tests for MemorySlot and vector utilities."""

import numpy as np
import pytest

from core_mem.slot import MemorySlot, cosine_similarity, online_centroid_merge, _l2_normalize


def _rand_emb(dim: int = 384) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestMemorySlot:
    def test_embedding_is_normalized(self):
        raw = np.random.randn(384).astype(np.float32)
        slot = MemorySlot(embedding=raw)
        assert abs(np.linalg.norm(slot.embedding) - 1.0) < 1e-5

    def test_default_merge_count(self):
        slot = MemorySlot(embedding=_rand_emb())
        assert slot.merge_count == 1

    def test_is_stable_needs_enough_history(self):
        slot = MemorySlot(embedding=_rand_emb())
        assert not slot.is_stable(window=3, epsilon=0.05)

    def test_is_stable_returns_true_when_converged(self):
        slot = MemorySlot(embedding=_rand_emb())
        for _ in range(5):
            slot.record_change(0.01)
        assert slot.is_stable(window=3, epsilon=0.05)

    def test_is_stable_returns_false_when_volatile(self):
        slot = MemorySlot(embedding=_rand_emb())
        slot.record_change(0.01)
        slot.record_change(0.01)
        slot.record_change(0.10)
        assert not slot.is_stable(window=3, epsilon=0.05)

    def test_to_dict_and_from_dict_roundtrip(self):
        emb = _rand_emb()
        slot = MemorySlot(embedding=emb, provenance="s1t2", salience=0.8)
        slot.record_change(0.02)
        slot.record_change(0.03)

        meta = slot.to_dict()
        restored = MemorySlot.from_dict(meta, emb)

        assert restored.merge_count == slot.merge_count
        assert restored.salience == slot.salience
        assert restored.provenance == slot.provenance
        assert len(restored._recent_deltas) == 2


class TestVectorUtils:
    def test_cosine_similarity_identical(self):
        v = _rand_emb()
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        a = np.zeros(384, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(384, dtype=np.float32)
        b[1] = 1.0
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_online_centroid_merge_moves_toward_incoming(self):
        a = _rand_emb()
        b = _rand_emb()
        slot = MemorySlot(embedding=a.copy())

        before_sim = cosine_similarity(slot.embedding, b)
        online_centroid_merge(slot, b, recency_weight=1.5)
        after_sim = cosine_similarity(slot.embedding, b)

        assert after_sim > before_sim

    def test_online_centroid_merge_increments_count(self):
        slot = MemorySlot(embedding=_rand_emb())
        assert slot.merge_count == 1
        online_centroid_merge(slot, _rand_emb(), recency_weight=1.5)
        assert slot.merge_count == 2

    def test_online_centroid_merge_records_change(self):
        slot = MemorySlot(embedding=_rand_emb())
        magnitude = online_centroid_merge(slot, _rand_emb(), recency_weight=1.5)
        assert magnitude > 0
        assert len(slot._recent_deltas) == 1
