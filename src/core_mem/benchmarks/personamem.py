"""PersonaMem adapter for official stage-1 benchmark files."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from core_mem.benchmarks.common import PromptRecord


@dataclass(frozen=True)
class PersonaMemConfig:
    variant: str = "32k"
    task: str = "multiple_choice"


@dataclass(frozen=True)
class PersonaMemQuestion:
    persona_id: str
    question_id: str
    question_type: str
    topic: str
    user_question_or_message: str
    correct_answer: str
    all_options: list[str]
    shared_context_id: str
    end_index_in_shared_context: int


class PersonaMemAdapter:
    def __init__(self, data_root: str | Path = "data/personamem", config: PersonaMemConfig | None = None) -> None:
        self.data_root = Path(data_root)
        self.config = config or PersonaMemConfig()

    def planned_variant_path(self, variant: str = "32k") -> Path:
        return self.data_root / variant

    def questions_path(self, variant: str | None = None) -> Path:
        selected = variant or self.config.variant
        return self.data_root / f"questions_{selected}.csv"

    def contexts_path(self, variant: str | None = None) -> Path:
        selected = variant or self.config.variant
        return self.data_root / f"shared_contexts_{selected}.jsonl"

    def load_questions(self, path: str | Path | None = None, limit: int | None = None) -> list[PersonaMemQuestion]:
        rows: list[PersonaMemQuestion] = []
        target = Path(path) if path else self.questions_path()
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                rows.append(
                    PersonaMemQuestion(
                        persona_id=str(row.get("persona_id", "")),
                        question_id=str(row.get("question_id", "")),
                        question_type=str(row.get("question_type", "")),
                        topic=str(row.get("topic", "")),
                        user_question_or_message=str(row.get("user_question_or_message", "")),
                        correct_answer=str(row.get("correct_answer", "")),
                        all_options=self._parse_options(row.get("all_options", "[]")),
                        shared_context_id=str(row.get("shared_context_id", "")),
                        end_index_in_shared_context=int(row.get("end_index_in_shared_context", 0) or 0),
                    )
                )
                if limit is not None and idx + 1 >= limit:
                    break
        return rows

    def load_shared_contexts(self, path: str | Path | None = None) -> dict[str, Any]:
        contexts: dict[str, Any] = {}
        target = Path(path) if path else self.contexts_path()
        with target.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                payload = json.loads(line)
                context_id = self._extract_context_id(payload, idx)
                contexts[context_id] = self._extract_context_payload(payload)
        return contexts

    def build_records(
        self,
        *,
        questions_path: str | Path | None = None,
        contexts_path: str | Path | None = None,
        limit: int | None = None,
    ) -> list[PromptRecord]:
        questions = self.load_questions(path=questions_path, limit=limit)
        contexts = self.load_shared_contexts(path=contexts_path)
        records: list[PromptRecord] = []
        for question in questions:
            rendered_context = self.render_context_for_question(question, contexts)
            options_block = "\n".join(
                f"{idx + 1}. {option}" for idx, option in enumerate(question.all_options)
            )
            answer_instruction = "Return the exact best option text."
            if self._options_use_labels(question.all_options) and question.correct_answer.startswith("("):
                answer_instruction = "Return only the best option label, for example (a)."
            prompt = (
                "You are answering a PersonaMem multiple-choice personalization question.\n\n"
                f"Context:\n{rendered_context}\n\n"
                f"User query:\n{question.user_question_or_message}\n\n"
                f"Options:\n{options_block}\n\n"
                f"{answer_instruction}"
            )
            records.append(
                PromptRecord(
                    sample_id=question.question_id,
                    benchmark="personamem",
                    prompt=prompt,
                    expected_answer=question.correct_answer,
                    metadata={
                        "persona_id": question.persona_id,
                        "question_type": question.question_type,
                        "topic": question.topic,
                        "shared_context_id": question.shared_context_id,
                        "prompt_version": "personamem_mc_v1",
                    },
                )
            )
        return records

    def render_context_for_question(
        self,
        question: PersonaMemQuestion,
        contexts: dict[str, Any],
    ) -> str:
        payload = contexts.get(question.shared_context_id)
        if payload is None:
            raise KeyError(f"Missing shared context id: {question.shared_context_id}")
        sliced = self._slice_context(payload, question.end_index_in_shared_context)
        return self._render_context(sliced)

    @staticmethod
    def _parse_options(raw_value: str) -> list[str]:
        parsed = ast.literal_eval(raw_value) if raw_value else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        raise ValueError("PersonaMem all_options must parse to a list.")

    @staticmethod
    def _options_use_labels(options: list[str]) -> bool:
        return bool(options) and all(option.startswith("(") and ")" in option[:4] for option in options)

    @staticmethod
    def _extract_context_id(payload: dict[str, Any], line_idx: int) -> str:
        for key in ("shared_context_id", "context_id", "id"):
            value = payload.get(key)
            if value:
                return str(value)
        if len(payload) == 1:
            return str(next(iter(payload.keys())))
        return f"line_{line_idx}"

    @staticmethod
    def _extract_context_payload(payload: dict[str, Any]) -> Any:
        for key in ("shared_context", "context", "messages", "conversation"):
            if key in payload:
                return payload[key]
        if len(payload) == 1:
            return next(iter(payload.values()))
        return payload

    @staticmethod
    def _slice_context(payload: Any, end_index: int) -> Any:
        if end_index <= 0:
            return payload
        if isinstance(payload, (list, str)):
            return payload[:end_index]
        if isinstance(payload, dict) and "messages" in payload and isinstance(payload["messages"], list):
            sliced = dict(payload)
            sliced["messages"] = payload["messages"][:end_index]
            return sliced
        return payload

    @classmethod
    def _render_context(cls, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                return "\n".join(
                    f"{item.get('role', 'unknown')}: {item.get('content', '')}".strip()
                    for item in payload
                )
            return "\n".join(str(item) for item in payload)
        if isinstance(payload, dict):
            if "messages" in payload and isinstance(payload["messages"], list):
                return cls._render_context(payload["messages"])
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return str(payload)
