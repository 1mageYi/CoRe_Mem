"""Mechanical verifier for stage-2 memory canary quality improvement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _contains(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8")


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


def compute_quality_status(
    *,
    root: Path,
    benchmark: str = "personamem",
    summary_path: Path | None = None,
) -> dict[str, Any]:
    effective_summary_path = summary_path or _latest_summary(root, benchmark)
    summary = _read_json(effective_summary_path) if effective_summary_path else {}
    predictions_path = Path(summary["predictions_path"]).resolve() if summary.get("predictions_path") else None
    if predictions_path and not predictions_path.exists():
        candidate = (root / summary["predictions_path"]).resolve()
        predictions_path = candidate if candidate.exists() else predictions_path
    rows = _read_jsonl(predictions_path) if predictions_path and predictions_path.exists() else []

    provider_exact = sum(
        1 for row in rows if _normalize(row.get("provider_prediction")) == _normalize(row.get("expected_answer"))
    )
    provider_prefix = sum(
        1 for row in rows if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction"))
    )
    local_exact = sum(
        1 for row in rows if _normalize(row.get("memory_answer_local")) == _normalize(row.get("expected_answer"))
    )

    current_status_path = root / "docs" / "current_status.md"
    project_index_path = root / ".agent-os" / "project-index.md"
    analysis_path = root / "outputs_v2" / "artifacts" / f"latest_{benchmark}_stage2_canary_analysis.json"

    checks = {
        "current_status_tracks_failure_analysis_priority": _contains(
            current_status_path,
            "对 live canary 做 failure analysis，并迭代修复 online memory / belief / answer 链路",
        ),
        "project_index_tracks_td024": _contains(
            project_index_path,
            "`TD-024` `[doing]`: 基于 `PersonaMem 64` live stage-2 canary 做 failure analysis",
        ),
        "analysis_artifact_exists": analysis_path.exists(),
        "live_personamem64_canary_completed": bool(summary)
        and summary.get("benchmark") == benchmark
        and summary.get("status") == "completed"
        and int(summary.get("sample_count", 0)) >= 64
        and int(summary.get("live_predictions_completed", 0)) >= 64,
        "provider_exact_ge_2": provider_exact >= 2,
        "provider_exact_ge_4": provider_exact >= 4,
        "provider_exact_ge_8": provider_exact >= 8,
        "provider_label_prefix_ge_24": provider_prefix >= 24,
        "provider_label_prefix_ge_32": provider_prefix >= 32,
        "local_exact_ge_1": local_exact >= 1,
    }
    return {
        "score": sum(1 for value in checks.values() if value),
        "total": len(checks),
        "checks": checks,
        "benchmark": benchmark,
        "summary_path": str(effective_summary_path) if effective_summary_path else None,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "metrics": {
            "provider_exact_match": provider_exact,
            "provider_label_prefix_match": provider_prefix,
            "local_exact_match": local_exact,
            "sample_count": len(rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--benchmark", default="personamem")
    parser.add_argument("--summary-path")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_quality_status(
        root=Path(args.root).resolve(),
        benchmark=args.benchmark,
        summary_path=Path(args.summary_path).resolve() if args.summary_path else None,
    )
    if args.score_only:
        print(payload["score"])
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
