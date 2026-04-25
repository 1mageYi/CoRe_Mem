"""Shared LLM helpers for PersonaMem MCQ (same contract as RAG_test / official inference)."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

_INSTRUCTIONS = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token."
)


def query_llm(
    client: "OpenAI",
    model: str,
    question: str,
    all_options: str,
    context_prefix: str,
    *,
    max_retries: int = 3,
) -> tuple[str, float]:
    """Single user turn: optional memory prefix + question + options."""
    body = context_prefix.strip()
    if body:
        content = (
            f"{body}\n\n---\n\n"
            f"{question}\n\n{_INSTRUCTIONS}\n\n{all_options}"
        )
    else:
        content = f"{question}\n\n{_INSTRUCTIONS}\n\n{all_options}"

    messages = [{"role": "user", "content": content}]

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


def extract_answer(predicted_answer: str, correct_answer: str) -> tuple[bool, str]:
    """Same heuristic as PersonaMem inference.py extract_answer."""

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


def format_retrieved_memories(memories: list[str]) -> str:
    """Turn top-k memory strings into a single prefix block."""
    if not memories:
        return ""
    lines = [f"- {m.strip()}" for m in memories if m and str(m).strip()]
    return (
        "The following bullet points are distilled memories about the user "
        "(extracted from prior conversation turns):\n"
        + "\n".join(lines)
    )


def parse_search_results(raw: object) -> list[str]:
    """Normalize mem0 search() return shape across SDK versions."""
    if raw is None:
        return []
    rows: list = []
    if isinstance(raw, dict):
        rows = raw.get("results") or raw.get("memories") or raw.get("data") or []
    elif isinstance(raw, list):
        rows = raw
    else:
        return []

    out: list[str] = []
    for r in rows:
        if isinstance(r, str):
            out.append(r)
        elif isinstance(r, dict):
            text = r.get("memory") or r.get("text") or r.get("content")
            if text:
                out.append(str(text))
    return out
