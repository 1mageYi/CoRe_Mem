from io import BytesIO
from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import core_mem.providers.openai_compatible as provider_module


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._buffer = BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buffer.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeHeaders:
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = mapping or {}

    def get(self, key: str, default=None):
        return self._mapping.get(key, default)


def test_provider_chat_uses_openai_compatible_contract(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr(provider_module.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ALIYUN_API_KEY", "test-key")
    provider = provider_module.OpenAICompatibleProvider()
    response = provider.chat("Say hello", temperature=0.2, max_tokens=64)

    assert response.content == "hello"
    assert captured["url"].endswith("/chat/completions")
    assert captured["timeout"] == 120
    assert captured["body"]["messages"][0]["content"] == "Say hello"
    assert captured["body"]["max_tokens"] == 64
    assert "Bearer test-key" in captured["headers"]["Authorization"]


def test_provider_retries_rate_limits(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(req, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise provider_module.error.HTTPError(
                req.full_url,
                429,
                "Too Many Requests",
                FakeHeaders({"Retry-After": "0"}),
                None,
            )
        return FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(provider_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(provider_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setenv("ALIYUN_API_KEY", "test-key")
    provider = provider_module.OpenAICompatibleProvider(
        provider_module.OpenAICompatibleConfig(max_retries=2, retry_backoff_seconds=0.1)
    )

    response = provider.chat("Retry once", temperature=0.0, max_tokens=8)

    assert response.content == "ok"
    assert calls["count"] == 2
    assert sleeps == [0.0]


def test_provider_waits_for_minimum_request_interval(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []
    timeline = iter([100.0, 101.0, 101.0, 105.0])

    def fake_monotonic():
        return next(timeline)

    def fake_urlopen(req, timeout):
        calls["count"] += 1
        return FakeHTTPResponse({"choices": [{"message": {"content": f"ok-{calls['count']}"}}]})

    monkeypatch.setattr(provider_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(provider_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(provider_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setenv("ALIYUN_API_KEY", "test-key")
    provider = provider_module.OpenAICompatibleProvider(
        provider_module.OpenAICompatibleConfig(min_request_interval_seconds=5.0)
    )

    first = provider.chat("first", temperature=0.0, max_tokens=8)
    second = provider.chat("second", temperature=0.0, max_tokens=8)

    assert first.content == "ok-1"
    assert second.content == "ok-2"
    assert calls["count"] == 2
    assert sleeps == [4.0]


def test_provider_caps_retry_delay(monkeypatch):
    captured: list[float] = []

    monkeypatch.setenv("ALIYUN_API_KEY", "test-key")
    provider = provider_module.OpenAICompatibleProvider(
        provider_module.OpenAICompatibleConfig(
            retry_backoff_seconds=5.0,
            max_retry_delay_seconds=30.0,
        )
    )

    uncapped = provider._retry_delay_seconds(
        provider_module.error.HTTPError(
            "https://example.com",
            429,
            "Too Many Requests",
            FakeHeaders(),
            None,
        ),
        attempt=5,
    )
    captured.append(uncapped)

    capped_retry_after = provider._retry_delay_seconds(
        provider_module.error.HTTPError(
            "https://example.com",
            429,
            "Too Many Requests",
            FakeHeaders({"Retry-After": "120"}),
            None,
        ),
        attempt=1,
    )
    captured.append(capped_retry_after)

    assert captured == [30.0, 30.0]
