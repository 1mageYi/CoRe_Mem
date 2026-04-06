"""Embedding encoder backed by sentence-transformers (global singleton)."""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(texts: str | list[str], *, normalize: bool = True) -> np.ndarray:
    """Encode one or more texts into L2-normalized embeddings.

    Parameters
    ----------
    texts : str or list[str]
        Input text(s).
    normalize : bool
        If True (default), L2-normalize the output vectors.

    Returns
    -------
    np.ndarray
        Shape ``(n, 384)`` for a list input, ``(384,)`` for a single string.
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if single:
        return embeddings[0]
    return embeddings


def embedding_dim() -> int:
    """Return the dimensionality of the embedding model."""
    return _EMBEDDING_DIM
