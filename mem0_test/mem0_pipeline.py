"""Mem0 (OSS) ingestion + retrieval aligned with PersonaMem §4.4.

Paper:
  - Iteratively build a memory DB with LLM-generated facts over each turn.
  - At inference, retrieve the top 5 relevant facts per question.

Implementation choices (paper silent on details):
  - Open-source ``Memory.from_config`` (not Mem0 Cloud ``MemoryClient``) so we
    can use the same OpenAI-compatible ``LLM_BASE_URL`` / key as RAG_test.
  - One isolated ``user_id`` per benchmark row = ``pm_{question_id}``.
  - FAISS on-disk under ``mem0_test/.faiss_personamem_idx`` (no external DB).
  - Default embedder: local HuggingFace ``all-MiniLM-L6-v2`` (proxies often lack
    ``text-embedding-*``). Override with ``MEM0_EMBED_PROVIDER=openai`` if available.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from llm_utils import parse_search_results

_HERE = Path(__file__).resolve().parent
# Separate path from any stale index built under a different embedding dim.
_FAISS_DIR = _HERE / ".faiss_personamem_idx"


def build_memory_config(
    *,
    api_key: str,
    base_url: str,
    llm_model: str,
    embed_model: str | None = None,
) -> dict[str, Any]:
    """Return a mem0 ``Memory.from_config`` dict (OpenAI-compatible endpoint)."""
    # Paper does not specify Mem0's embedding backend. Many OpenAI-compatible
    # proxies do not expose ``text-embedding-*``; default to local HF models.
    embed_provider = os.getenv("MEM0_EMBED_PROVIDER", "huggingface").lower()
    embed_model = embed_model or os.getenv(
        "MEM0_HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Help mem0 sub-clients pick up routing if they read env only
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    if embed_provider == "openai":
        emb_dims = int(os.getenv("MEM0_EMBEDDING_DIMS", "1536"))
        embedder: dict[str, Any] = {
            "provider": "openai",
            "config": {
                "model": os.getenv(
                    "MEM0_EMBEDDING_MODEL", "text-embedding-3-small"
                ),
                "api_key": api_key,
                "openai_base_url": base_url,
                "embedding_dims": emb_dims,
            },
        }
    else:
        hf_cfg: dict[str, Any] = {"model": embed_model}
        if os.getenv("MEM0_EMBEDDING_DIMS"):
            hf_cfg["embedding_dims"] = int(os.environ["MEM0_EMBEDDING_DIMS"])
        embedder = {"provider": "huggingface", "config": hf_cfg}
        emb_dims = hf_cfg.get("embedding_dims")
        if emb_dims is None:
            # Must match FAISS index width (mem0 defaults FAISS to 1536 otherwise).
            from sentence_transformers import SentenceTransformer

            emb_dims = SentenceTransformer(embed_model).get_sentence_embedding_dimension()

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": api_key,
                "openai_base_url": base_url,
                "temperature": 0.0,
                "max_tokens": 4096,
            },
        },
        "embedder": embedder,
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": "personamem_mem0_eval",
                "path": str(_FAISS_DIR),
                "distance_strategy": "cosine",
                "normalize_L2": True,
                "embedding_model_dims": emb_dims,
            },
        },
    }


def _clean_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _add_chunk(memory: Any, chunk: list[dict], user_id: str) -> None:
    """Call ``memory.add`` (mem0ai 2.x has no ``async_mode`` flag)."""
    memory.add(chunk, user_id=user_id, infer=True)


def ingest_history_turnwise(
    memory: Any,
    messages: list[dict],
    user_id: str,
) -> float:
    """Feed conversation to Mem0 in **pairs** (user, assistant) like dialog turns.

    Returns wall-clock seconds spent in ``add`` calls.
    """
    msgs = _clean_messages(messages)
    t0 = time.perf_counter()
    i = 0
    while i < len(msgs):
        pair = msgs[i : i + 2]
        _add_chunk(memory, pair, user_id=user_id)
        i += len(pair)
    return time.perf_counter() - t0


def search_top_memories(
    memory: Any,
    query: str,
    user_id: str,
    top_k: int = 5,
) -> tuple[list[str], float]:
    """Semantic search over Mem0 store; returns (memory_texts, elapsed_s)."""
    t0 = time.perf_counter()
    # mem0ai 2.x: entity id lives in ``filters`` (``user_id`` kwarg is not on the signature).
    raw = memory.search(
        query,
        filters={"user_id": user_id},
        top_k=top_k,
        threshold=0.0,
    )
    elapsed = time.perf_counter() - t0
    texts = parse_search_results(raw)
    return texts[:top_k], elapsed
