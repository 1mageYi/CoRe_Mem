"""LongMemEval adapter for official cleaned benchmark files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from core_mem.benchmarks.common import PromptRecord


@dataclass(frozen=True)
class LongMemEvalConfig:
    subset: str = "s"


@dataclass(frozen=True)
class LongMemEvalQuestion:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_sessions: list[list[dict[str, Any]]]
    answer_session_ids: list[str]


class LongMemEvalAdapter:
    def __init__(self, data_root: str | Path = "data/longmemeval", config: LongMemEvalConfig | None = None) -> None:
        self.data_root = Path(data_root)
        self.config = config or LongMemEvalConfig()

    def planned_subset_path(self, subset: str = "s") -> Path:
        return self.data_root / subset

    def dataset_path(self, subset: str | None = None) -> Path:
        selected = (subset or self.config.subset).lower()
        return self.data_root / f"longmemeval_{selected}_cleaned.json"

    def load_questions(
        self,
        path: str | Path | None = None,
        limit: int | None = None,
    ) -> list[LongMemEvalQuestion]:
        target = Path(path) if path else self.dataset_path()
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        questions: list[LongMemEvalQuestion] = []
        for idx, item in enumerate(data):
            questions.append(
                LongMemEvalQuestion(
                    question_id=str(item.get("question_id", "")),
                    question_type=str(item.get("question_type", "")),
                    question=str(item.get("question", "")),
                    answer=str(item.get("answer", "")),
                    question_date=str(item.get("question_date", "")),
                    haystack_sessions=item.get("haystack_sessions", []),
                    answer_session_ids=[str(value) for value in item.get("answer_session_ids", [])],
                )
            )
            if limit is not None and idx + 1 >= limit:
                break
        return questions

    def build_records(
        self,
        *,
        dataset_path: str | Path | None = None,
        limit: int | None = None,
    ) -> list[PromptRecord]:
        questions = self.load_questions(path=dataset_path, limit=limit)
        records: list[PromptRecord] = []
        for question in questions:
            history = self.render_history(question.haystack_sessions)
            prompt = (
                "You are answering a LongMemEval long-term memory question.\n\n"
                f"Timestamped history:\n{history}\n\n"
                f"Question date: {question.question_date}\n"
                f"Question: {question.question}\n\n"
                "Return only the exact answer phrase from the history, with no leading article, verb, or extra words."
            )
            records.append(
                PromptRecord(
                    sample_id=question.question_id,
                    benchmark="longmemeval",
                    prompt=prompt,
                    expected_answer=question.answer,
                    metadata={
                        "question_type": question.question_type,
                        "answer_session_ids": question.answer_session_ids,
                        "prompt_version": "longmemeval_s_v1",
                    },
                )
            )
        return records

    @classmethod
    def render_history(cls, sessions: list[list[dict[str, Any]]]) -> str:
        rendered_sessions = []
        for session_idx, session in enumerate(sessions, start=1):
            turns = "\n".join(
                f"{turn.get('role', 'unknown')}: {turn.get('content', '')}".strip()
                for turn in session
            )
            rendered_sessions.append(f"[Session {session_idx}]\n{turns}")
        return "\n\n".join(rendered_sessions)
