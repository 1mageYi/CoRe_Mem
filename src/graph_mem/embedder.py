from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class BgeM3Embedder:
    model_name: str = "BAAI/bge-m3"
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Lazy import keeps module import fast for unit tests.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32).tolist()

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        av = np.asarray(a, dtype=np.float32)
        bv = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom == 0.0:
            return 0.0
        return float(np.dot(av, bv) / denom)
