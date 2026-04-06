"""Project-wide configuration loaded from environment and config files."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-235b-a22b-instruct-2507"

    model_config = {"env_prefix": "LLM_"}


class MemorySettings(BaseSettings):
    core_slots: int = 32
    residual_slots: int = 64

    model_config = {"env_prefix": "MEMORY_"}
