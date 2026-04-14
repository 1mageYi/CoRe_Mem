from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.embeddings.sentence_transformer import SentenceTransformerEmbedder
from core_mem.memory import VanillaCoreMemory


class DummyEmbedder:
    def __init__(self) -> None:
        self.mapping = {
            "alice likes chess": [1.0, 0.0],
            "alice studies math": [0.0, 1.0],
            "what does alice like": [1.0, 0.0],
        }

    def embed(self, text: str) -> list[float]:
        return list(self.mapping[text])


def test_sentence_transformer_embedder_uses_expected_default_model():
    embedder = SentenceTransformerEmbedder()
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.normalize is True


def test_vanilla_core_memory_supports_text_observe_and_query():
    memory = VanillaCoreMemory(embedder=DummyEmbedder())
    memory.observe_text(
        "alice likes chess",
        timestamp="2026-04-05T00:00:00Z",
        salience=0.8,
        support=["pref"],
    )
    memory.observe_text(
        "alice studies math",
        timestamp="2026-04-05T00:01:00Z",
        salience=0.3,
        support=["study"],
    )

    result = memory.query_text("what does alice like")
    assert result.supporting_slots[0].support == ["pref"]
