"""OpenAI-compatible provider adapter for stage-1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration used by the stage-1 API provider adapter."""

    api_key_env: str = "ALIYUN_API_KEY"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-235b-a22b-instruct-2507"
    timeout_seconds: int = 120
    max_retries: int = 0
    retry_backoff_seconds: float = 2.0
    min_request_interval_seconds: float = 0.0
    max_retry_delay_seconds: float = 60.0

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class OpenAICompatibleProvider:
    """Formats requests for an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, config: OpenAICompatibleConfig | None = None) -> None:
        self.config = config or OpenAICompatibleConfig()
        self._last_request_started_at: float | None = None

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

    def _retry_delay_seconds(self, exc: error.HTTPError, attempt: int) -> float:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), self.config.max_retry_delay_seconds)
            except ValueError:
                pass
        exponential = self.config.retry_backoff_seconds * (2 ** max(attempt - 1, 0))
        return min(exponential, self.config.max_retry_delay_seconds)

    def _maybe_wait_for_request_interval(self) -> None:
        minimum_interval = self.config.min_request_interval_seconds
        if minimum_interval <= 0:
            return
        now = time.monotonic()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            if elapsed < minimum_interval:
                time.sleep(minimum_interval - elapsed)
                now = time.monotonic()
        self._last_request_started_at = now

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
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                self._maybe_wait_for_request_interval()
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                should_retry = exc.code == 429 or 500 <= exc.code < 600
                if not should_retry or attempt >= attempts:
                    raise
                time.sleep(self._retry_delay_seconds(exc, attempt))
        content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ProviderResponse(content=content, raw=parsed)


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    raw: dict[str, Any]
