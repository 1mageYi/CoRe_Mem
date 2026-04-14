"""Generate stage-2 benchmark canary manifests without launching the benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable, Iterable, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.benchmarks.longmemeval import LongMemEvalAdapter
from core_mem.benchmarks.personamem import PersonaMemAdapter

T = TypeVar("T")


def _stratified_take(items: Iterable[T], total: int, key_fn: Callable[[T], str]) -> list[T]:
    grouped: dict[str, deque[T]] = defaultdict(deque)
    for item in items:
        grouped[key_fn(item)].append(item)
    if not grouped:
        return []

    selected: list[T] = []
    keys = sorted(grouped)
    while len(selected) < total and any(grouped.values()):
        for key in keys:
            if grouped[key]:
                selected.append(grouped[key].popleft())
                if len(selected) >= total:
                    break
    return selected


def build_canary_manifests(
    output_root: Path,
    *,
    personamem_size: int,
    longmemeval_size: int,
    personamem_questions: Path,
    personamem_contexts: Path,
    longmemeval_dataset: Path,
) -> dict[str, object]:
    output_dir = output_root / "evals_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    personamem = PersonaMemAdapter(data_root=personamem_questions.parent)
    longmemeval = LongMemEvalAdapter(data_root=longmemeval_dataset.parent)
    pm_questions = personamem.load_questions(path=personamem_questions)
    lm_questions = longmemeval.load_questions(path=longmemeval_dataset)

    pm_sample = _stratified_take(pm_questions, personamem_size, lambda item: item.question_type or "unknown")
    lm_sample = _stratified_take(lm_questions, longmemeval_size, lambda item: item.question_type or "unknown")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pm_path = output_dir / f"{timestamp}_personamem_canary.json"
    lm_path = output_dir / f"{timestamp}_longmemeval_canary.json"
    summary_path = output_dir / f"{timestamp}_stage2_canary_plan.json"

    pm_payload = {
        "benchmark": "personamem",
        "planned_size": personamem_size,
        "source_questions": str(personamem_questions),
        "source_contexts": str(personamem_contexts),
        "sample_ids": [item.question_id for item in pm_sample],
        "question_types": sorted({item.question_type for item in pm_sample}),
    }
    lm_payload = {
        "benchmark": "longmemeval_s",
        "planned_size": longmemeval_size,
        "source_dataset": str(longmemeval_dataset),
        "sample_ids": [item.question_id for item in lm_sample],
        "question_types": sorted({item.question_type for item in lm_sample}),
    }
    pm_path.write_text(json.dumps(pm_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lm_path.write_text(json.dumps(lm_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "created_at": timestamp,
                "personamem_manifest": str(pm_path),
                "longmemeval_manifest": str(lm_path),
                "personamem_count": len(pm_sample),
                "longmemeval_count": len(lm_sample),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "summary_path": str(summary_path),
        "personamem_manifest": str(pm_path),
        "longmemeval_manifest": str(lm_path),
        "personamem_count": len(pm_sample),
        "longmemeval_count": len(lm_sample),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--personamem-size", type=int, default=64)
    parser.add_argument("--longmemeval-size", type=int, default=64)
    parser.add_argument("--personamem-questions", default="data/personamem/questions_32k.csv")
    parser.add_argument("--personamem-contexts", default="data/personamem/shared_contexts_32k.jsonl")
    parser.add_argument("--longmemeval-dataset", default="data/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_canary_manifests(
        Path(args.output_root),
        personamem_size=args.personamem_size,
        longmemeval_size=args.longmemeval_size,
        personamem_questions=Path(args.personamem_questions),
        personamem_contexts=Path(args.personamem_contexts),
        longmemeval_dataset=Path(args.longmemeval_dataset),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"summary_path={payload['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
