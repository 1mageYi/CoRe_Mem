"""Configuration loading for the stage-1 CoRe Memory platform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key_env: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 120
    max_retries: int = 0
    retry_backoff_seconds: float = 2.0
    min_request_interval_seconds: float = 0.0
    max_retry_delay_seconds: float = 60.0


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "sentence_transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    device: str = "cpu"


@dataclass(frozen=True)
class PersonaMemSettings:
    data_root: str = "data/personamem"
    variant: str = "32k"
    task: str = "multiple_choice"


@dataclass(frozen=True)
class LongMemEvalSettings:
    data_root: str = "data/longmemeval"
    subset: str = "s"


@dataclass(frozen=True)
class BenchmarksConfig:
    primary: str
    secondary: str
    personamem: PersonaMemSettings
    longmemeval: LongMemEvalSettings


@dataclass(frozen=True)
class OutputsConfig:
    root: str = "outputs"


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    stage: str
    llm: LLMConfig
    embedding: EmbeddingConfig
    benchmarks: BenchmarksConfig
    outputs: OutputsConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level config must be a mapping.")
    return loaded


def load_project_config(path: str | Path) -> ProjectConfig:
    payload = _load_yaml(path)
    project = payload.get("project", {})
    llm = payload.get("llm", {})
    embedding = payload.get("embedding", {})
    benchmarks = payload.get("benchmarks", {})
    outputs = payload.get("outputs", {})

    return ProjectConfig(
        name=project.get("name", "core_mem"),
        stage=project.get("stage", "stage1"),
        llm=LLMConfig(
            provider=llm.get("provider", "aliyun_openai_compatible"),
            api_key_env=llm.get("api_key_env", "ALIYUN_API_KEY"),
            base_url=llm.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=llm.get("model", "qwen3-235b-a22b-instruct-2507"),
            temperature=float(llm.get("temperature", 0.0)),
            max_tokens=int(llm.get("max_tokens", 512)),
            timeout_seconds=int(llm.get("timeout_seconds", 120)),
            max_retries=int(llm.get("max_retries", 0)),
            retry_backoff_seconds=float(llm.get("retry_backoff_seconds", 2.0)),
            min_request_interval_seconds=float(llm.get("min_request_interval_seconds", 0.0)),
            max_retry_delay_seconds=float(llm.get("max_retry_delay_seconds", 60.0)),
        ),
        embedding=EmbeddingConfig(
            provider=embedding.get("provider", "sentence_transformers"),
            model_name=embedding.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            normalize=bool(embedding.get("normalize", True)),
            device=embedding.get("device", "cpu"),
        ),
        benchmarks=BenchmarksConfig(
            primary=benchmarks.get("primary", "personamem"),
            secondary=benchmarks.get("secondary", "longmemeval_s"),
            personamem=PersonaMemSettings(**(benchmarks.get("personamem", {}) or {})),
            longmemeval=LongMemEvalSettings(**(benchmarks.get("longmemeval", {}) or {})),
        ),
        outputs=OutputsConfig(root=outputs.get("root", "outputs")),
    )
