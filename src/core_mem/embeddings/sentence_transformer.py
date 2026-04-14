"""Sentence-transformers embedding adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    device: str = "cpu"
    _model: Any | None = field(default=None, init=False, repr=False)

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing at runtime
                raise RuntimeError(
                    "sentence-transformers is not installed. Add it to the environment before live embedding."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=self.normalize)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

