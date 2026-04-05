"""OpenAI-compatible provider adapter for stage-1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import request


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration used by the stage-1 API provider adapter."""

    api_key_env: str = "ALIYUN_API_KEY"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-235b-a22b-instruct-2507"
    timeout_seconds: int = 120

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

    @property
    def endpoint_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> "ProviderResponse":
        if not self.is_configured():
            raise RuntimeError(
                f"Missing API key. Set the environment variable {self.config.api_key_env} before live inference."
            )

        payload = self.build_chat_payload(prompt=prompt, temperature=temperature)
        if system_prompt:
            payload["messages"].insert(0, {"role": "system", "content": system_prompt})
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ProviderResponse(content=content, raw=parsed)


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    raw: dict[str, Any]
