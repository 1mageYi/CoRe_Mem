"""Unit tests for MemoryWriter (uses pre-computed embeddings to avoid model load)."""

import numpy as np
import pytest

from core_mem.config import MemorySettings
from core_mem.writer import MemoryWriter


def _rand_emb(dim: int = 384) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def cfg() -> MemorySettings:
    return MemorySettings(
        residual_slots=4,
        core_slots=2,
        merge_threshold=0.8,
        core_threshold=0.8,
        promotion_merge_count=3,
        stability_window=2,
        stability_epsilon=0.15,
        recency_weight=1.5,
    )


class TestMemoryWriter:
    def test_write_embedding_creates_residual(self, cfg):
        writer = MemoryWriter(cfg)
        result = writer.write_embedding(_rand_emb(), provenance="s1t1")
        assert writer.residual.count == 1
        assert not result.promoted

    def test_repeated_similar_writes_trigger_promotion(self, cfg):
        writer = MemoryWriter(cfg)
        base = _rand_emb()
        promoted = False
        for i in range(10):
            noise = base + np.random.randn(384).astype(np.float32) * 0.005
            noise = noise / np.linalg.norm(noise)
            result = writer.write_embedding(noise, provenance=f"s1t{i}")
            if result.promoted:
                promoted = True
                break

        assert promoted
        assert writer.core.count >= 1

    def test_dissimilar_writes_fill_residual(self, cfg):
        writer = MemoryWriter(cfg)
        for _ in range(4):
            writer.write_embedding(_rand_emb())
        assert writer.residual.count == 4

    def test_save_and_load_roundtrip(self, cfg, tmp_path):
        writer = MemoryWriter(cfg)
        for _ in range(3):
            writer.write_embedding(_rand_emb())

        writer.save(tmp_path)

        writer2 = MemoryWriter(cfg)
        writer2.load(tmp_path)

        assert writer2.residual.count == writer.residual.count
        assert writer2.core.count == writer.core.count
