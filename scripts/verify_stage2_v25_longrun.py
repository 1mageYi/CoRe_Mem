"""Mechanical verifier for the v2.5 learned core-path long run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _contains_all(path: Path, patterns: list[str]) -> bool:
    text = _read_text(path)
    return bool(text) and all(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _artifact_matches_head(path: Path, head: str | None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    if head is not None and str(payload.get("commit_hash", "")) != head:
        return None
    return payload


def _current_head_payload(root: Path, name: str) -> dict[str, Any] | None:
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / name, _current_head(root))


def _code_contains_any(root: Path, paths: list[str], patterns: list[str]) -> bool:
    for rel in paths:
        text = _read_text(root / rel)
        if text and any(pattern in text for pattern in patterns):
            return True
    return False


def _historical_v24_artifacts_exist(root: Path) -> bool:
    required = [
        root / "outputs_v2" / "artifacts" / "latest_stage2_v24_train.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v24_eval.json",
        root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v24_canary.json",
        root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v24_canary.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v24_online_gain.json",
    ]
    return all(path.exists() for path in required)


def compute_v25_longrun(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    v25_plan = root / "docs" / "v25_plan.md"

    v24_eval_payload = _current_head_payload(root, "latest_stage2_v24_eval.json")
    v24_longmemeval_payload = _current_head_payload(root, "latest_longmemeval_stage2_v24_canary.json")
    v24_personamem_payload = _current_head_payload(root, "latest_personamem_stage2_v24_canary.json")

    v25_train_payload = _current_head_payload(root, "latest_stage2_v25_train.json")
    v25_eval_payload = _current_head_payload(root, "latest_stage2_v25_eval.json")
    v25_write_gain = _current_head_payload(root, "latest_stage2_v25_write_gain.json")
    v25_retrieve_gain = _current_head_payload(root, "latest_stage2_v25_retrieve_gain.json")
    v25_belief_gain = _current_head_payload(root, "latest_stage2_v25_belief_gain.json")
    v25_longmemeval_payload = _current_head_payload(root, "latest_longmemeval_stage2_v25_canary.json")
    v25_personamem_payload = _current_head_payload(root, "latest_personamem_stage2_v25_canary.json")
    v25_analysis_payload = _current_head_payload(root, "latest_longmemeval_stage2_v25_analysis.json")
    v25_full_benchmark_payload = _current_head_payload(root, "latest_stage2_v25_full_benchmark.json")

    trained_metrics = ((v24_eval_payload or {}).get("trained_eval") or {}).get("metrics") or {}

    checks = {
        "current_status_tracks_td036_v25": _contains_all(
            current_status,
            ["`TD-036`", "`v2.5`", "LongMemEval-S", "write -> retrieve -> belief", "core / residual"],
        ),
        "implementation_plan_tracks_stage_u_v25": _contains_all(
            implementation_plan,
            ["阶段 U：V2.5 Generalization-First Long Run", "LongMemEval-S", "full benchmark"],
        ),
        "agentos_tracks_td036_ws022": _contains_all(
            project_index,
            ["`TD-036 / WS-022`", "`v2.5 learned core-path long-run`"],
        ) and _contains_all(todo, ["`TD-036` `[doing]`"]),
        "v25_plan_exists": v25_plan.exists(),
        "v25_plan_freezes_core_residual": _contains_all(
            v25_plan,
            ["不改 `core / residual`", "冻结边界", "write", "retrieve", "belief"],
        ),
        "v25_plan_bans_shortcuts_and_exactness": _contains_all(
            v25_plan,
            ["不做任何 `fallback`", "不做任何 `shortcut`", "不把 raw JSON exactness 当成主优化目标"],
        ),
        "v25_plan_bans_benchmark_leakage": _contains_all(
            v25_plan,
            ["不把 full benchmark 结果回流成训练 supervision", "holdout evaluation"],
        ),
        "core_residual_structure_still_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/consolidation.py"],
            ["core_slots", "residual_slots", "core_slots:", "residual_slots:"],
        ),
        "learned_write_path_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/lifecycle.py", "src/core_mem/v2/training.py"],
            ["slot_assignment_mode", "learned_slot_assignment", "SLOT_ASSIGNMENT_TASK_NAME"],
        ),
        "learned_retrieve_path_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/encoder.py", "src/core_mem/v2/resampler.py"],
            ["query_encoder", "_ranking_score", "compose(", "retrieval_key"],
        ),
        "learned_belief_path_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/decoder.py", "src/core_mem/v2/semantic_outputs.py"],
            ["belief_state", "belief_source", "coerce_task_payload", "decode("],
        ),
        "historical_v24_baseline_artifacts_exist": _historical_v24_artifacts_exist(root),
        "current_head_v24_eval_still_strong": float(trained_metrics.get("token_f1", 0.0)) >= 0.93
        and float(trained_metrics.get("field_f1", 0.0)) >= 0.93,
        "current_head_v24_longmemeval_128_exists": bool(v24_longmemeval_payload)
        and int(v24_longmemeval_payload.get("sample_count", 0)) >= 128
        and v24_longmemeval_payload.get("status") == "completed",
        "current_head_v24_personamem_128_exists": bool(v24_personamem_payload)
        and int(v24_personamem_payload.get("sample_count", 0)) >= 128
        and v24_personamem_payload.get("status") == "completed",
        "current_head_v25_train_exists": bool(v25_train_payload),
        "current_head_v25_eval_exists": bool(v25_eval_payload),
        "current_head_v25_write_gain_exists": bool(v25_write_gain),
        "current_head_v25_retrieve_gain_exists": bool(v25_retrieve_gain),
        "current_head_v25_belief_gain_exists": bool(v25_belief_gain),
        "current_head_v25_longmemeval_128_exists": bool(v25_longmemeval_payload)
        and int(v25_longmemeval_payload.get("sample_count", 0)) >= 128
        and v25_longmemeval_payload.get("status") == "completed",
        "current_head_v25_personamem_128_exists": bool(v25_personamem_payload)
        and int(v25_personamem_payload.get("sample_count", 0)) >= 128
        and v25_personamem_payload.get("status") == "completed",
        "current_head_v25_longmemeval_analysis_exists": bool(v25_analysis_payload)
        and any(key in v25_analysis_payload for key in ["error_clusters", "layered_breakdown", "failure_buckets"]),
        "current_head_v25_full_benchmark_exists": bool(v25_full_benchmark_payload)
        and int(v25_full_benchmark_payload.get("num_benchmarks", 0)) >= 1,
    }

    score = sum(1 for value in checks.values() if value)
    return {
        "metric": "stage2_v25_longrun_score",
        "score": score,
        "total": len(checks),
        "checks": checks,
        "head": _current_head(root),
        "trained_eval_metrics": trained_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v25_longrun(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
