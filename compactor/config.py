"""Defaults aligned with DESIGN.md (gtr-t5-base + vec2text stack)."""

from __future__ import annotations

# SentenceTransformer id used for memory / compactor training & inference.
EMBED_MODEL_NAME = "sentence-transformers/gtr-t5-base"

# Populated on first encoder load (typically 768 for gtr-t5-base).
_embed_dim: int | None = None


def set_embed_dim(dim: int) -> None:
    global _embed_dim
    _embed_dim = dim


def embed_dim() -> int:
    if _embed_dim is None:
        raise RuntimeError("embed_dim unset; call set_embed_dim or load_encoder first")
    return _embed_dim
