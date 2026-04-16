"""Lazy SentenceTransformer encoder (gtr-t5-base) with L2-normalized outputs."""

from __future__ import annotations

import numpy as np
import torch

from compactor.config import EMBED_MODEL_NAME, set_embed_dim

_model = None


def load_encoder(model_name: str = EMBED_MODEL_NAME):
    """Return a SentenceTransformer singleton and set global embedding dim."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        get_dim = getattr(_model, "get_embedding_dimension", None)
        dim = get_dim() if callable(get_dim) else _model.get_sentence_embedding_dimension()
        set_embed_dim(dim)
    return _model


def encode_texts(
    texts: list[str],
    *,
    model_name: str = EMBED_MODEL_NAME,
    normalize: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode strings to a (n, d) float32 tensor on CPU or ``device``."""
    model = load_encoder(model_name)
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    t = torch.from_numpy(np.asarray(emb, dtype=np.float32))
    if device is not None:
        t = t.to(device)
    return t


def encode_texts_numpy(
    texts: list[str],
    *,
    model_name: str = EMBED_MODEL_NAME,
    normalize: bool = True,
) -> np.ndarray:
    """Encode to (n, d) float32 numpy array (CPU)."""
    model = load_encoder(model_name)
    emb = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)
