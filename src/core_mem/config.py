"""Project-wide configuration loaded from environment and config files."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-235b-a22b-instruct-2507"

    model_config = {"env_prefix": "LLM_"}


class EmbeddingSettings(BaseSettings):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384

    model_config = {"env_prefix": "EMBEDDING_"}


class MemorySettings(BaseSettings):
    core_slots: int = 32
    residual_slots: int = 64
    merge_threshold: float = 0.5
    core_threshold: float = 0.5
    promotion_merge_count: int = 5
    stability_window: int = 3
    stability_epsilon: float = 0.05
    recency_weight: float = 1.5

    model_config = {"env_prefix": "MEMORY_"}
