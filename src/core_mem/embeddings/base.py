"""Embedding interfaces for stage-1 CoRe Memory."""

from __future__ import annotations

from typing import Protocol


class TextEmbedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Encode one text input into a dense latent vector."""

