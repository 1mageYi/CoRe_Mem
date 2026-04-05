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
