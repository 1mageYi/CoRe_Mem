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


def _exact_match(expected: str, prediction: str | None) -> bool:
    return _normalize(expected) == _normalize(prediction)


def _canonical_benchmark_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in {"longmemeval", "longmemeval_s"}:
        return "longmemeval"
    return normalized


def _latest_summary(root: Path, benchmark: str) -> Path | None:
    expected = _canonical_benchmark_name(benchmark)
    candidates = sorted((root / "outputs_v2" / "evals_benchmark").glob("*stage2_memory_canary.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        if _canonical_benchmark_name(str(payload.get("benchmark", ""))) == expected and payload.get("status") == "completed":
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
        "commit_hash": summary.get("commit_hash"),
        "memory_mode": summary.get("memory_mode"),
        "slot_assignment_mode": summary.get("slot_assignment_mode"),
        "benchmark": _canonical_benchmark_name(benchmark),
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
        "error_clusters": dict(failure_relation_counts.most_common(12)),
        "failure_buckets": dict(failure_relation_counts.most_common(12)),
        "failure_examples": failure_examples,
    }


def _classify_failure_layer(row: dict[str, Any]) -> str:
    belief_items = row.get("belief_state", {}).get("belief_items", [])
    selected_slot_ids = row.get("selected_slot_ids", [])
    expected_answer = str(row.get("expected_answer", ""))
    local_answer = row.get("memory_answer_local")
    provider_prediction = row.get("provider_prediction")

    if not belief_items:
        return "parser"
    if not selected_slot_ids:
        return "retrieval"
    if _exact_match(expected_answer, local_answer) and not _label_prefix_match(expected_answer, provider_prediction):
        return "provider"
    if _normalize(local_answer):
        return "projection"
    return "belief"


def build_layered_analysis(
    *,
    root: Path,
    benchmark: str,
    summary_path: Path | None = None,
    example_limit: int = 8,
) -> dict[str, Any]:
    effective_summary_path = summary_path or _latest_summary(root, benchmark)
    if effective_summary_path is None:
        raise FileNotFoundError(f"No completed stage2 memory canary summary found for benchmark={benchmark}")
    summary = _read_json(effective_summary_path)
    predictions_path = Path(summary["predictions_path"])
    if not predictions_path.is_absolute():
        predictions_path = (root / predictions_path).resolve()
    rows = _read_jsonl(predictions_path)

    layered_rows: dict[str, list[dict[str, Any]]] = {
        "parser": [],
        "retrieval": [],
        "belief": [],
        "projection": [],
        "provider": [],
    }
    for row in rows:
        if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction")):
            continue
        layer = _classify_failure_layer(row)
        layered_rows[layer].append(row)

    layers: dict[str, Any] = {}
    for layer_name, layer_rows in layered_rows.items():
        layers[layer_name] = {
            "count": len(layer_rows),
            "sample_ids": [str(row.get("sample_id", "")) for row in layer_rows[:example_limit]],
            "question_types": dict(
                Counter(str(row.get("question_type", "unknown")) for row in layer_rows).most_common(6)
            ),
        }

    return {
        "commit_hash": summary.get("commit_hash"),
        "memory_mode": summary.get("memory_mode"),
        "slot_assignment_mode": summary.get("slot_assignment_mode"),
        "benchmark": _canonical_benchmark_name(benchmark),
        "summary_path": str(effective_summary_path),
        "predictions_path": str(predictions_path),
        "sample_count": len(rows),
        "layers": layers,
    }


def write_analysis(root: Path, benchmark: str, payload: dict[str, Any]) -> dict[str, str]:
    benchmark_key = _canonical_benchmark_name(benchmark)
    artifacts_dir = root / "outputs_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    stamped_path = artifacts_dir / f"{stamp}_{benchmark_key}_stage2_canary_analysis.json"
    latest_path = artifacts_dir / f"latest_{benchmark_key}_stage2_canary_analysis.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    stamped_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    semantic_latest_path = None
    if benchmark_key == "longmemeval" and payload.get("memory_mode") == "learned_memory":
        semantic_latest_path = artifacts_dir / "latest_longmemeval_stage2_semantic_analysis.json"
        semantic_latest_path.write_text(text, encoding="utf-8")
    v24_latest_path = None
    if benchmark_key == "longmemeval" and payload.get("slot_assignment_mode") == "learned":
        v24_latest_path = artifacts_dir / "latest_longmemeval_stage2_v24_analysis.json"
        v24_latest_path.write_text(text, encoding="utf-8")
    return {
        "stamped_path": str(stamped_path),
        "latest_path": str(latest_path),
        "semantic_latest_path": str(semantic_latest_path) if semantic_latest_path else "",
        "v24_latest_path": str(v24_latest_path) if v24_latest_path else "",
    }


def write_layered_analysis(root: Path, benchmark: str, payload: dict[str, Any]) -> dict[str, str]:
    benchmark_key = _canonical_benchmark_name(benchmark)
    artifacts_dir = root / "outputs_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    stamped_path = artifacts_dir / f"{stamp}_{benchmark_key}_stage2_layered_analysis.json"
    latest_path = artifacts_dir / f"latest_{benchmark_key}_stage2_layered_analysis.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    stamped_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return {
        "stamped_path": str(stamped_path),
        "latest_path": str(latest_path),
    }


def _summary_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "sample_count": int(summary.get("sample_count", len(rows))),
        "provider_exact_match": sum(
            1 for row in rows if _normalize(row.get("provider_prediction")) == _normalize(row.get("expected_answer"))
        ),
        "provider_label_prefix_match": sum(
            1 for row in rows if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction"))
        ),
        "local_exact_match": sum(
            1 for row in rows if _normalize(row.get("memory_answer_local")) == _normalize(row.get("expected_answer"))
        ),
    }


def build_online_gain(
    *,
    root: Path,
    baseline_summary_path: Path,
    improved_summary_path: Path,
) -> dict[str, Any]:
    baseline_summary = _read_json(baseline_summary_path)
    improved_summary = _read_json(improved_summary_path)

    baseline_predictions_path = Path(baseline_summary["predictions_path"])
    if not baseline_predictions_path.is_absolute():
        baseline_predictions_path = (root / baseline_predictions_path).resolve()
    improved_predictions_path = Path(improved_summary["predictions_path"])
    if not improved_predictions_path.is_absolute():
        improved_predictions_path = (root / improved_predictions_path).resolve()

    baseline_rows = _read_jsonl(baseline_predictions_path)
    improved_rows = _read_jsonl(improved_predictions_path)
    baseline_metrics = _summary_metrics(baseline_summary, baseline_rows)
    improved_metrics = _summary_metrics(improved_summary, improved_rows)

    delta_provider_exact = improved_metrics["provider_exact_match"] - baseline_metrics["provider_exact_match"]
    delta_provider_prefix = (
        improved_metrics["provider_label_prefix_match"] - baseline_metrics["provider_label_prefix_match"]
    )
    delta_local_exact = improved_metrics["local_exact_match"] - baseline_metrics["local_exact_match"]

    return {
        "artifact_type": "stage2_semantic_online_gain",
        "commit_hash": improved_summary.get("commit_hash"),
        "benchmark": _canonical_benchmark_name(str(improved_summary.get("benchmark", ""))),
        "memory_mode": improved_summary.get("memory_mode"),
        "slot_assignment_mode": improved_summary.get("slot_assignment_mode"),
        "comparison_scope": f"live_provider_canary_{improved_metrics['sample_count']}",
        "baseline_summary_path": str(baseline_summary_path),
        "improved_summary_path": str(improved_summary_path),
        "baseline_commit_hash": baseline_summary.get("commit_hash"),
        "improved_commit_hash": improved_summary.get("commit_hash"),
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "delta_provider_exact_match": delta_provider_exact,
        "delta_provider_label_prefix_match": delta_provider_prefix,
        "delta_local_exact_match": delta_local_exact,
        "positive_gain": any(delta > 0 for delta in (delta_provider_exact, delta_provider_prefix, delta_local_exact)),
    }


def write_semantic_online_gain(root: Path, payload: dict[str, Any]) -> dict[str, str]:
    artifacts_dir = root / "outputs_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    stamped_path = artifacts_dir / f"{stamp}_stage2_semantic_online_gain.json"
    latest_path = artifacts_dir / "latest_stage2_semantic_online_gain.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    stamped_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return {
        "stamped_path": str(stamped_path),
        "latest_path": str(latest_path),
    }


def write_v24_online_gain(root: Path, payload: dict[str, Any]) -> dict[str, str]:
    artifacts_dir = root / "outputs_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    stamped_path = artifacts_dir / f"{stamp}_stage2_v24_online_gain.json"
    latest_path = artifacts_dir / "latest_stage2_v24_online_gain.json"
    v24_payload = dict(payload)
    v24_payload["artifact_type"] = "stage2_v24_online_gain"
    text = json.dumps(v24_payload, ensure_ascii=False, indent=2)
    stamped_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return {
        "stamped_path": str(stamped_path),
        "latest_path": str(latest_path),
    }


def write_slot_assignment_gain(root: Path, payload: dict[str, Any]) -> dict[str, str]:
    artifacts_dir = root / "outputs_v2" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamped_path = artifacts_dir / f"{_timestamp()}_stage2_slot_assignment_gain.json"
    latest_path = artifacts_dir / "latest_stage2_slot_assignment_gain.json"
    slot_assignment_payload = dict(payload)
    slot_assignment_payload["artifact_type"] = "stage2_slot_assignment_gain"
    text = json.dumps(slot_assignment_payload, ensure_ascii=False, indent=2)
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
    parser.add_argument("--baseline-summary-path")
    parser.add_argument("--improved-summary-path")
    parser.add_argument("--write-semantic-online-gain", action="store_true")
    parser.add_argument("--write-v24-online-gain", action="store_true")
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
    layered_payload = build_layered_analysis(
        root=root,
        benchmark=args.benchmark,
        summary_path=Path(args.summary_path).resolve() if args.summary_path else None,
        example_limit=args.example_limit,
    )
    if not args.no_write:
        payload["artifact_paths"] = write_analysis(root, args.benchmark, payload)
        layered_payload["artifact_paths"] = write_layered_analysis(root, args.benchmark, layered_payload)
        payload["layered_artifact_paths"] = layered_payload["artifact_paths"]
    gain_payload = None
    if args.write_semantic_online_gain:
        if not args.baseline_summary_path or not args.improved_summary_path:
            raise ValueError("--write-semantic-online-gain requires --baseline-summary-path and --improved-summary-path")
        gain_payload = build_online_gain(
            root=root,
            baseline_summary_path=Path(args.baseline_summary_path).resolve(),
            improved_summary_path=Path(args.improved_summary_path).resolve(),
        )
        if not args.no_write:
            gain_payload["artifact_paths"] = write_semantic_online_gain(root, gain_payload)
    if args.write_v24_online_gain:
        if not args.baseline_summary_path or not args.improved_summary_path:
            raise ValueError("--write-v24-online-gain requires --baseline-summary-path and --improved-summary-path")
        gain_payload = build_online_gain(
            root=root,
            baseline_summary_path=Path(args.baseline_summary_path).resolve(),
            improved_summary_path=Path(args.improved_summary_path).resolve(),
        )
        if not args.no_write:
            gain_payload["artifact_paths"] = write_v24_online_gain(root, gain_payload)
    if args.json:
        response: dict[str, Any] = {"analysis": payload, "layered_analysis": layered_payload}
        if gain_payload is not None:
            response["semantic_online_gain"] = gain_payload
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(f"provider_exact_match={payload['provider_exact_match']}")
        print(f"provider_label_prefix_match={payload['provider_label_prefix_match']}")
        if "artifact_paths" in payload:
            print(f"latest_path={payload['artifact_paths']['latest_path']}")
        if "layered_artifact_paths" in payload:
            print(f"latest_layered_path={payload['layered_artifact_paths']['latest_path']}")
        if gain_payload is not None and "artifact_paths" in gain_payload:
            print(f"semantic_online_gain_path={gain_payload['artifact_paths']['latest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
