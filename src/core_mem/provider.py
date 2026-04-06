"""OpenAI-compatible LLM provider adapter (Alibaba Cloud DashScope)."""

from __future__ import annotations

from openai import OpenAI

from core_mem.config import LLMSettings


def build_client(settings: LLMSettings | None = None) -> OpenAI:
    settings = settings or LLMSettings()
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)
