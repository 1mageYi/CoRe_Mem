from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_mem import AddPipeline, MemoryGraphStore, RuleExtractor, SearchPipeline


class FakeEmbedder:
    def encode(self, text: str) -> list[float]:
        # Deterministic tiny embedding for tests.
        x = float(len(text) % 13) / 13.0
        y = float(sum(ord(c) for c in text) % 17) / 17.0
        z = float(text.count("food") + text.count("prefer")) / 10.0
        return [x, y, z]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


def test_graph_pipeline_smoke() -> None:
    extractor = RuleExtractor()
    embedder = FakeEmbedder()
    store = MemoryGraphStore()
    add = AddPipeline(graph_store=store, embedder=embedder)
    search = SearchPipeline(graph_store=store, embedder=embedder)

    rows = [
        "I like Italian food.",
        "I prefer Mediterranean food now.",
        "I do not like crowded events anymore.",
    ]
    for idx, text in enumerate(rows, start=1):
        rec = extractor.extract_from_turn(text, time_index=idx, source_turn_id=idx)
        add.add_record(rec, now_ts=idx)

    evidence = search.search("What food should I try now?", now_ts=10)
    assert len(evidence) > 0
    assert len(store.nodes) >= 2
