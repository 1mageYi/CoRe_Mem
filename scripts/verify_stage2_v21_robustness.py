"""Mechanical verifier for stage-2 robust v2.1 milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _contains_all(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(pattern in text for pattern in patterns)


def _contains_any(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return any(pattern in text for pattern in patterns)


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


def _run_score(script: Path) -> int:
    if not script.exists():
        return 0
    result = subprocess.run(
        ["python3", str(script), "--score-only"],
        cwd=script.parent.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def _current_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else None


def _benchmark_matches(payload_benchmark: str | None, expected: str) -> bool:
    normalized = str(payload_benchmark or "").strip().lower()
    aliases = {
        "personamem": {"personamem"},
        "longmemeval": {"longmemeval", "longmemeval_s"},
    }
    return normalized in aliases.get(expected, {expected})


def _latest_canary(
    root: Path,
    *,
    benchmark: str,
    min_samples: int,
    current_head_only: bool,
) -> Path | None:
    current_head = _current_head(root) if current_head_only else None
    candidates = sorted((root / "outputs_v2" / "evals_benchmark").glob("*stage2_memory_canary.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        if payload.get("status") != "completed":
            continue
        if not _benchmark_matches(payload.get("benchmark"), benchmark):
            continue
        if int(payload.get("sample_count", 0)) < min_samples:
            continue
        if int(payload.get("live_predictions_completed", 0)) < min_samples:
            continue
        if not bool(payload.get("provider_configured", False)):
            continue
        if current_head is not None and str(payload.get("commit_hash", "")) != current_head:
            continue
        return path
    return None


def _summary_metrics(root: Path, summary_path: Path | None) -> dict[str, float]:
    if summary_path is None or not summary_path.exists():
        return {
            "sample_count": 0.0,
            "provider_exact_match": 0.0,
            "provider_label_prefix_match": 0.0,
            "local_exact_match": 0.0,
            "provider_exact_match_rate": 0.0,
            "provider_label_prefix_match_rate": 0.0,
            "local_exact_match_rate": 0.0,
        }
    summary = _read_json(summary_path)
    predictions_ref = summary.get("predictions_path")
    if not predictions_ref:
        return {
            "sample_count": float(summary.get("sample_count", 0)),
            "provider_exact_match": 0.0,
            "provider_label_prefix_match": 0.0,
            "local_exact_match": 0.0,
            "provider_exact_match_rate": 0.0,
            "provider_label_prefix_match_rate": 0.0,
            "local_exact_match_rate": 0.0,
        }
    predictions_path = Path(str(predictions_ref))
    if not predictions_path.is_absolute():
        predictions_path = (root / predictions_path).resolve()
    rows = _read_jsonl(predictions_path)
    total = len(rows)
    provider_exact = sum(
        1 for row in rows if _normalize(row.get("provider_prediction")) == _normalize(row.get("expected_answer"))
    )
    provider_prefix = sum(
        1 for row in rows if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction"))
    )
    local_exact = sum(
        1 for row in rows if _normalize(row.get("memory_answer_local")) == _normalize(row.get("expected_answer"))
    )
    return {
        "sample_count": float(total),
        "provider_exact_match": float(provider_exact),
        "provider_label_prefix_match": float(provider_prefix),
        "local_exact_match": float(local_exact),
        "provider_exact_match_rate": (provider_exact / total) if total else 0.0,
        "provider_label_prefix_match_rate": (provider_prefix / total) if total else 0.0,
        "local_exact_match_rate": (local_exact / total) if total else 0.0,
    }


def _runner_avoids_shortcuts(root: Path) -> bool:
    runner = root / "scripts" / "run_stage2_memory_canary.py"
    if not runner.exists():
        return False
    text = runner.read_text(encoding="utf-8")
    banned = [
        "Latent matcher candidate:",
        "_finalize_personamem_provider_prediction",
        "_best_personamem_option_label(",
        "blank-provider fallback",
    ]
    return not any(pattern in text for pattern in banned)


def _layered_analysis_ok(path: Path) -> bool:
    if not path.exists():
        return False
    payload = _read_json(path)
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        return False
    required = {"parser", "retrieval", "belief", "projection", "provider"}
    return required.issubset(layers.keys())


def _learned_gain_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    deltas = [
        float(payload.get("delta_provider_exact_match", 0.0)),
        float(payload.get("delta_provider_label_prefix_match", 0.0)),
        float(payload.get("delta_local_exact_match", 0.0)),
    ]
    positive = bool(payload.get("positive_gain")) or any(delta > 0.0 for delta in deltas)
    if not positive:
        return None
    return payload


def _latest_eval(root: Path) -> dict[str, Any]:
    candidates = sorted((root / "outputs_v2" / "evals_local").glob("*stage2_local_eval.json"))
    best_payload: dict[str, Any] = {}
    best_token_f1 = -1.0
    for path in candidates:
        payload = _read_json(path)
        trained_eval_metrics = ((payload.get("trained_eval") or {}).get("metrics") or {})
        token_f1 = float(trained_eval_metrics.get("token_f1", 0.0))
        if token_f1 >= best_token_f1:
            best_payload = payload
            best_token_f1 = token_f1
    return best_payload


def compute_v21_robustness(root: Path) -> dict[str, Any]:
    current_status_path = root / "docs" / "current_status.md"
    implementation_plan_path = root / "docs" / "implementation_plan.md"
    project_index_path = root / ".agent-os" / "project-index.md"
    todo_path = root / ".agent-os" / "todo.md"

    latent_core_score = _run_score(root / "scripts" / "verify_stage2_latent_core_quality.py")

    personamem_64 = _latest_canary(root, benchmark="personamem", min_samples=64, current_head_only=True)
    longmemeval_64 = _latest_canary(root, benchmark="longmemeval", min_samples=64, current_head_only=True)
    personamem_128 = _latest_canary(root, benchmark="personamem", min_samples=128, current_head_only=True)

    personamem_128_metrics = _summary_metrics(root, personamem_128)
    longmemeval_64_metrics = _summary_metrics(root, longmemeval_64)

    layered_analysis_path = root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_layered_analysis.json"
    learned_gain_path = root / "outputs_v2" / "artifacts" / "latest_stage2_learned_online_gain.json"
    learned_gain = _learned_gain_payload(learned_gain_path)
    latest_eval = _latest_eval(root)
    trained_eval_metrics = ((latest_eval.get("trained_eval") or {}).get("metrics") or {})
    trained_token_f1 = float(trained_eval_metrics.get("token_f1", 0.0))

    checks = {
        "current_status_tracks_v21_goal": _contains_all(
            current_status_path,
            [
                "`v2.1`",
                "提升 learned path 的实际贡献",
                "stage2_v21_robustness_score",
            ],
        ),
        "implementation_plan_tracks_v21_phases": _contains_all(
            implementation_plan_path,
            [
                "阶段 M：V2.1 Robustness 主线",
                "LongMemEval-S 专项 failure analysis",
                "`LongMemEval-S` 专项质量提升与 learned path 在线增益来组织",
            ],
        ),
        "agentos_tracks_td028_ws014": _contains_all(
            project_index_path,
            [
                "Top next action: `TD-028`",
                "Active workstreams: `WS-014`",
            ],
        ) and _contains_any(
            todo_path,
            [
                "`TD-028` `[doing]`",
            ],
        ),
        "latent_core_quality_10_of_10": latent_core_score >= 10,
        "benchmark_runner_avoids_shortcuts": _runner_avoids_shortcuts(root),
        "fresh_personamem64_current_head_completed": personamem_64 is not None,
        "fresh_longmemeval64_current_head_completed": longmemeval_64 is not None,
        "personamem128_current_head_completed": personamem_128 is not None,
        "personamem128_provider_prefix_rate_ge_0_40": personamem_128_metrics["provider_label_prefix_match_rate"] >= 0.40,
        "personamem128_local_exact_rate_ge_0_20": personamem_128_metrics["local_exact_match_rate"] >= 0.20,
        "longmemeval_layered_analysis_exists": _layered_analysis_ok(layered_analysis_path),
        "longmemeval_provider_prefix_ge_4": longmemeval_64_metrics["provider_label_prefix_match"] >= 4.0,
        "longmemeval_local_exact_ge_2": longmemeval_64_metrics["local_exact_match"] >= 2.0,
        "learned_online_gain_positive": learned_gain is not None,
        "non_tiny_trained_eval_token_f1_ge_0_08": trained_token_f1 >= 0.08,
    }

    return {
        "score": sum(1 for passed in checks.values() if passed),
        "total": len(checks),
        "checks": checks,
        "latent_core_quality_score": latent_core_score,
        "latest_personamem64_current_head": str(personamem_64) if personamem_64 else None,
        "latest_longmemeval64_current_head": str(longmemeval_64) if longmemeval_64 else None,
        "latest_personamem128_current_head": str(personamem_128) if personamem_128 else None,
        "personamem128_metrics": personamem_128_metrics,
        "longmemeval64_metrics": longmemeval_64_metrics,
        "layered_analysis_path": str(layered_analysis_path) if layered_analysis_path.exists() else None,
        "learned_gain_path": str(learned_gain_path) if learned_gain else None,
        "trained_eval_token_f1": trained_token_f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_v21_robustness(Path(args.root).resolve())
    if args.score_only:
        print(payload["score"])
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
