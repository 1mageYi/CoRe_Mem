"""RAG pipeline for PersonaMem evaluation.

Steps per question:
  1. Embed every message in the sliced conversation history with BGE-M3.
  2. Retrieve the top-k messages by cosine similarity to the user question.
  3. Restore chronological order of retrieved messages.
  4. Query GPT-4o-mini with retrieved context + question + options.
  5. Parse the answer letter from the model output.

Faithfully mirrors §4.4 of the PersonaMem paper:
  "top five most relevant messages … using dense BGE-M3 embeddings"
"""

from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Embedding  (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_embed_model = None
_EMBED_MODEL_NAME = "BAAI/bge-m3"


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading {_EMBED_MODEL_NAME} (first call only) …")
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        print("[embedder] Model ready.")
    return _embed_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode texts with L2 normalisation. Returns shape (n, dim)."""
    model = _get_embed_model()
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return np.asarray(embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    messages: list[dict],
    query: str,
    k: int = 5,
) -> list[dict]:
    """Return the top-k messages most relevant to query.

    Retrieved messages are returned in their original chronological order
    so the LLM sees a coherent conversation snippet.

    Parameters
    ----------
    messages : list of OpenAI-style message dicts ({role, content}).
    query    : the user's current question.
    k        : number of messages to retrieve.
    """
    if not messages:
        return []

    # Filter messages with non-empty content
    valid = [(i, m) for i, m in enumerate(messages) if m.get("content", "").strip()]
    if not valid:
        return []

    indices, valid_msgs = zip(*valid)
    texts = [f"{m['role']}: {m['content']}" for m in valid_msgs]

    query_emb = embed_texts([query])[0]   # (dim,)
    msg_embs = embed_texts(list(texts))   # (n, dim)

    scores = msg_embs @ query_emb         # cosine similarities (already L2-normed)
    top_k = min(k, len(valid_msgs))
    top_local = np.argsort(scores)[::-1][:top_k]

    # Map back to original indices and sort chronologically
    top_original = sorted(indices[i] for i in top_local)
    return [messages[i] for i in top_original]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_INSTRUCTIONS = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token."
)


def query_llm(
    client,
    model: str,
    question: str,
    all_options: str,
    retrieved_messages: list[dict],
    max_retries: int = 3,
) -> tuple[str, float]:
    """Send retrieved context + question + options to the LLM.

    Matches the original inference.py prompt construction exactly:
        messages = context + [{"role": "user", "content": question + instructions + options}]

    Returns
    -------
    (raw_response_text, elapsed_seconds_for_the_api_call)
    """
    messages = retrieved_messages + [
        {
            "role": "user",
            "content": f"{question}\n\n{_INSTRUCTIONS}\n\n{all_options}",
        }
    ]

    delay = 1.0
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(model=model, messages=messages)
            elapsed = time.perf_counter() - t0
            return resp.choices[0].message.content, elapsed
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            print(f"  [LLM retry {attempt + 1}/{max_retries}] {exc} — retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Answer extraction  (mirrors inference.py extract_answer exactly)
# ---------------------------------------------------------------------------

def extract_answer(predicted_answer: str, correct_answer: str) -> tuple[bool, str]:
    """Return (is_correct, parsed_letter).

    Implements the same heuristic as the original PersonaMem codebase so that
    accuracy numbers are directly comparable.
    """
    def _options(text: str) -> set[str]:
        text = text.lower()
        in_parens = re.findall(r"\(([a-d])\)", text)
        return set(in_parens) if in_parens else set(re.findall(r"\b([a-d])\b", text))

    correct = correct_answer.lower().strip("() ")

    full = predicted_answer
    pred = predicted_answer.strip()
    if " " in pred:
        pred = pred.split(" ")[-1].strip()
    pred = pred.rstrip()

    if _options(pred) == {correct}:
        return True, pred
    if _options(full) == {correct}:
        return True, pred
    return False, pred
