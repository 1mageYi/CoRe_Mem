"""Mechanical verifier for stage-2 latent-core quality improvements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _contains(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8")


def _contains_any(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return any(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _latest_eval(root: Path) -> Path | None:
    candidates = sorted((root / "outputs_v2" / "evals_local").glob("*stage2_local_eval.json"))
    return candidates[-1] if candidates else None


def _latest_canary(root: Path) -> Path | None:
    candidates = sorted((root / "outputs_v2" / "evals_benchmark").glob("*stage2_memory_canary.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        if payload.get("status") == "completed" and int(payload.get("sample_count", 0)) >= 64:
            return path
    return None


def _metric(payload: dict[str, Any], name: str) -> float:
    return float(payload.get("metrics", {}).get(name, 0.0))


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def _label_prefix_match(expected: str, prediction: str | None) -> bool:
    normalized_expected = _normalize(expected)
    normalized_prediction = _normalize(prediction)
    return bool(normalized_expected and normalized_prediction.startswith(normalized_expected))


def _latent_readiness_score(root: Path) -> int:
    script = root / "scripts" / "verify_stage2_latent_status.py"
    if not script.exists():
        return 0
    result = subprocess.run(
        ["python3", str(script), "--score-only"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def _canary_quality_guard(root: Path) -> bool:
    summary_path = _latest_canary(root)
    if summary_path is None:
        return False
    summary = _read_json(summary_path)
    predictions_ref = summary.get("predictions_path")
    if not predictions_ref:
        return False
    predictions_path = Path(str(predictions_ref))
    if not predictions_path.is_absolute():
        predictions_path = (root / predictions_path).resolve()
    if not predictions_path.exists():
        return False
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
    return provider_exact >= 8 and provider_prefix >= 24 and local_exact >= 1 and len(rows) >= 64


def compute_latent_core_quality(root: Path, eval_path: Path | None = None) -> dict[str, Any]:
    effective_eval_path = eval_path or _latest_eval(root)
    eval_payload = _read_json(effective_eval_path) if effective_eval_path else {}

    current_status_path = root / "docs" / "current_status.md"
    project_index_path = root / ".agent-os" / "project-index.md"

    checks = {
        "current_status_tracks_latent_core_priority": _contains(
            current_status_path,
            "把主指标重新锚定到 `stage-2 local intrinsic quality`",
        ),
        "project_index_tracks_td026": _contains_any(
            project_index_path,
            [
                "`TD-026` `[doing]`: 以系统/模型/latent 本体更强、更稳健为锚点",
                "`WS-012` `[done]`: Stage-2 latent-core robustness 目标已机械达成",
                "历史主线兼容记录：`TD-026` `[doing]`: 以系统/模型/latent 本体更强、更稳健为锚点",
            ],
        ),
        "latent_readiness_9_of_9": _latent_readiness_score(root) >= 9,
        "local_eval_artifact_exists": effective_eval_path is not None and Path(effective_eval_path).exists(),
        "joint_belief_accuracy_ge_0_88": _metric(eval_payload, "joint_belief_accuracy") >= 0.88,
        "slot_value_f1_ge_0_94": _metric(eval_payload, "slot_value_f1") >= 0.94,
        "support_slot_recall_ge_0_97": _metric(eval_payload, "support_slot_recall") >= 0.97,
        "answer_exact_match_ge_0_75": _metric(eval_payload, "answer_exact_match") >= 0.75,
        "compression_fidelity_ge_0_87": _metric(eval_payload, "compression_fidelity") >= 0.87,
        "personamem_canary_quality_ge_9": _canary_quality_guard(root),
    }

    metrics = {
        "joint_belief_accuracy": _metric(eval_payload, "joint_belief_accuracy"),
        "slot_value_f1": _metric(eval_payload, "slot_value_f1"),
        "support_slot_recall": _metric(eval_payload, "support_slot_recall"),
        "answer_exact_match": _metric(eval_payload, "answer_exact_match"),
        "compression_fidelity": _metric(eval_payload, "compression_fidelity"),
    }
    return {
        "score": sum(1 for passed in checks.values() if passed),
        "total": len(checks),
        "checks": checks,
        "eval_path": str(effective_eval_path) if effective_eval_path else None,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--eval-path")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_latent_core_quality(
        Path(args.root).resolve(),
        Path(args.eval_path).resolve() if args.eval_path else None,
    )
    if args.score_only:
        print(payload["score"])
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
