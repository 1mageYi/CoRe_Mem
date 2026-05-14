from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(slots=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str


def load_llm_config(env_file: str | None = None) -> LLMConfig:
    if env_file:
        load_dotenv(env_file)
    model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    return LLMConfig(model=model, base_url=base_url, api_key=api_key)


def build_eval_prompt(question: str, options_text: str, evidence_texts: list[str]) -> str:
    evidence = "\n".join(f"- {x}" for x in evidence_texts)
    return (
        "You are evaluating a personalized memory assistant.\n"
        "Use ONLY the evidence snippets below to choose the best option.\n"
        "Return only one capital letter from the options.\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer:"
    )


def query_option_letter(client: OpenAI, cfg: LLMConfig, prompt: str, max_retries: int = 3) -> tuple[str, float]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            text = resp.choices[0].message.content or ""
            letter = extract_option_letter(text)
            return letter, elapsed
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(30.0, delay * 2.0)
    return "?", 0.0


def extract_option_letter(text: str) -> str:
    found = re.findall(r"\b([A-H])\b", text.upper())
    if found:
        return found[-1]
    paren = re.findall(r"\(([A-H])\)", text.upper())
    if paren:
        return paren[-1]
    return "?"
