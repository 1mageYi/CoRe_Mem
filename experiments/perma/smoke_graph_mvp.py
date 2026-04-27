from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_mem import AddPipeline, BgeM3Embedder, MemoryGraphStore, RuleExtractor, SearchPipeline


class SimpleEmbedder:
    def encode(self, text: str) -> list[float]:
        x = float(len(text) % 23) / 23.0
        y = float(sum(ord(c) for c in text) % 29) / 29.0
        z = float(text.count("prefer") + text.count("like")) / 10.0
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


def main() -> None:
    extractor = RuleExtractor()
    use_bge = os.getenv("USE_BGE_M3", "").strip() == "1"
    embedder = BgeM3Embedder() if use_bge else SimpleEmbedder()
    store = MemoryGraphStore()
    add = AddPipeline(graph_store=store, embedder=embedder)
    search = SearchPipeline(graph_store=store, embedder=embedder)

    turns = [
        "I like Italian food and usually order pasta.",
        "Recently I prefer Mediterranean seafood instead.",
        "I do not enjoy crowded places anymore.",
        "I prefer small and quiet events now.",
        "I am looking for travel plans with calm schedules.",
    ]

    for idx, text in enumerate(turns, start=1):
        rec = extractor.extract_from_turn(text, time_index=idx, source_turn_id=idx)
        add.add_record(rec, now_ts=idx)

    query = "Recommend food options that match my latest preference."
    evidence = search.search(query, now_ts=len(turns) + 1)

    print("=== Graph MVP Smoke ===")
    print(f"nodes={len(store.nodes)} edges={store.graph.number_of_edges()}")
    print(f"query={query}")
    print("top evidence:")
    for i, ev in enumerate(evidence, start=1):
        print(f"{i}. {ev}")


if __name__ == "__main__":
    main()
