"""Embedding adapters for stage-1 CoRe Memory."""

from core_mem.embeddings.base import TextEmbedder
from core_mem.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "SentenceTransformerEmbedder",
    "TextEmbedder",
]
