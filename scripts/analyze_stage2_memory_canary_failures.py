"""Analyze stage-2 memory canary predictions and write a failure breakdown."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def _label_prefix_match(expected: str, prediction: str | None) -> bool:
    normalized_expected = _normalize(expected)
    normalized_prediction = _normalize(prediction)
    return bool(normalized_expected and normalized_prediction.startswith(normalized_expected))


def _latest_summary(root: Path, benchmark: str) -> Path | None:
    candidates = sorted((root / "outputs_v2" / "evals_benchmark").glob("*stage2_memory_canary.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        if payload.get("benchmark") == benchmark and payload.get("status") == "completed":
            return path
    return None


def _summarize_group(
    rows: list[dict[str, Any]],
    *,
    key_name: str,
) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}
    for key, group_rows in sorted(
        defaultdict(list, ((row.get(key_name) or "unknown", []) for row in rows)).items()
    ):
        groups[key] = group_rows
    groups = {}
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row.get(key_name) or "unknown")].append(row)
    for key in sorted(grouped_rows):
        group_rows = grouped_rows[key]
        groups[key] = {
            "count": len(group_rows),
            "provider_exact_match": sum(
                1
                for row in group_rows
                if _normalize(row.get("provider_prediction")) == _normalize(row.get("expected_answer"))
            ),
            "provider_label_prefix_match": sum(
                1
                for row in group_rows
                if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction"))
            ),
            "local_exact_match": sum(
                1
                for row in group_rows
                if _normalize(row.get("memory_answer_local")) == _normalize(row.get("expected_answer"))
            ),
        }
    return groups


def build_analysis(
    *,
    root: Path,
    benchmark: str,
    summary_path: Path | None = None,
    example_limit: int = 12,
) -> dict[str, Any]:
    effective_summary_path = summary_path or _latest_summary(root, benchmark)
    if effective_summary_path is None:
        raise FileNotFoundError(f"No completed stage2 memory canary summary found for benchmark={benchmark}")
    summary = _read_json(effective_summary_path)
    predictions_path = Path(summary["predictions_path"])
    if not predictions_path.is_absolute():
        predictions_path = (root / predictions_path).resolve()
    rows = _read_jsonl(predictions_path)

    provider_exact = sum(
        1 for row in rows if _normalize(row.get("provider_prediction")) == _normalize(row.get("expected_answer"))
    )
    provider_prefix = sum(
        1 for row in rows if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction"))
    )
    local_exact = sum(
        1 for row in rows if _normalize(row.get("memory_answer_local")) == _normalize(row.get("expected_answer"))
    )
    provider_completed = sum(1 for row in rows if row.get("provider_status") == "completed")

    failure_relation_counts: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    for row in rows:
        if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction")):
            continue
        relations = [
            str(item.get("relation", "unknown"))
            for item in row.get("belief_state", {}).get("belief_items", [])
            if isinstance(item, dict)
        ]
        failure_relation_counts.update(relations or ["<empty_belief>"])
        if len(failure_examples) < example_limit:
            failure_examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "question_type": row.get("question_type", "unknown"),
                    "topic": row.get("topic", "unknown"),
                    "expected_answer": row.get("expected_answer"),
                    "provider_prediction_preview": str(row.get("provider_prediction") or "")[:240],
                    "memory_answer_local_preview": str(row.get("memory_answer_local") or "")[:240],
                    "belief_relations": relations,
                    "selected_slot_count": len(row.get("selected_slot_ids", [])),
                }
            )

    total = len(rows)
    return {
        "benchmark": benchmark,
        "summary_path": str(effective_summary_path),
        "predictions_path": str(predictions_path),
        "sample_count": total,
        "provider_completed": provider_completed,
        "provider_exact_match": provider_exact,
        "provider_exact_match_rate": (provider_exact / total) if total else 0.0,
        "provider_label_prefix_match": provider_prefix,
        "provider_label_prefix_match_rate": (provider_prefix / total) if total else 0.0,
        "local_exact_match": local_exact,
        "local_exact_match_rate": (local_exact / total) if total else 0.0,
        "by_question_type": _summarize_group(rows, key_name="question_type"),
        "by_topic": _summarize_group(rows, key_name="topic"),
        "top_failure_belief_relations": dict(failure_relation_counts.most_common(12)),
        "failure_examples": failure_examples,
    }


def write_analysis(root: Path, benchmark: str, payload: dict[str, Any]) -> dict[str, str]:
    artifacts_dir = root / "outputs_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    stamped_path = artifacts_dir / f"{stamp}_{benchmark}_stage2_canary_analysis.json"
    latest_path = artifacts_dir / f"latest_{benchmark}_stage2_canary_analysis.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    stamped_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return {
        "stamped_path": str(stamped_path),
        "latest_path": str(latest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--benchmark", default="personamem")
    parser.add_argument("--summary-path")
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_analysis(
        root=root,
        benchmark=args.benchmark,
        summary_path=Path(args.summary_path).resolve() if args.summary_path else None,
        example_limit=args.example_limit,
    )
    if not args.no_write:
        payload["artifact_paths"] = write_analysis(root, args.benchmark, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"provider_exact_match={payload['provider_exact_match']}")
        print(f"provider_label_prefix_match={payload['provider_label_prefix_match']}")
        if "artifact_paths" in payload:
            print(f"latest_path={payload['artifact_paths']['latest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
