"""Minimal OpenAI-compatible provider adapter for stage-1 scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration used by the stage-1 API provider adapter."""

    api_key_env: str = "ALIYUN_API_KEY"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-235b-a22b-instruct-2507"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class OpenAICompatibleProvider:
    """Formats requests for an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, config: OpenAICompatibleConfig | None = None) -> None:
        self.config = config or OpenAICompatibleConfig()

    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def build_chat_payload(self, prompt: str, temperature: float = 0.0) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

