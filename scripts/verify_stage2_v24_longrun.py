"""Mechanical verifier for the v2.4 quality-focused long run."""

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


def _historical_v23_closeout_artifacts_exist(root: Path) -> bool:
    required = [
        root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_train.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_eval.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_gain.json",
        root / "outputs_v2" / "evals_benchmark" / "20260416T095034Z_stage2_memory_canary.json",
        root / "outputs_v2" / "evals_benchmark" / "20260416T095030Z_stage2_memory_canary.json",
    ]
    return all(path.exists() for path in required)


def compute_v24_longrun(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    v24_plan = root / "docs" / "v24_plan.md"

    historical_train = root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_train.json"
    historical_eval = root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_local_eval.json"
    historical_longmemeval = root / "outputs_v2" / "evals_benchmark" / "20260416T095034Z_stage2_memory_canary.json"
    historical_personamem = root / "outputs_v2" / "evals_benchmark" / "20260416T095030Z_stage2_memory_canary.json"

    train_payload = _current_head_payload(root, "latest_stage2_v24_train.json")
    eval_payload = _current_head_payload(root, "latest_stage2_v24_eval.json")
    longmemeval_payload = _current_head_payload(root, "latest_longmemeval_stage2_v24_canary.json")
    personamem_payload = _current_head_payload(root, "latest_personamem_stage2_v24_canary.json")
    analysis_payload = _current_head_payload(root, "latest_longmemeval_stage2_v24_analysis.json")
    gain_payload = _current_head_payload(root, "latest_stage2_v24_online_gain.json")

    metrics = ((eval_payload or {}).get("trained_eval") or {}).get("metrics") or {}
    slot_assignment_metrics = (
        (eval_payload or {}).get("slot_assignment_metrics")
        or (eval_payload or {}).get("slot_assignment")
        or {}
    )

    checks = {
        "current_status_tracks_td035_v24": _contains_all(
            current_status,
            ["`TD-035`", "`v2.4`", "LongMemEval-S", "learned slot assignment", "stronger latent"],
        ),
        "implementation_plan_tracks_stage_t_v24": _contains_all(
            implementation_plan,
            ["阶段 T：V2.4 LongMemEval-S Quality + Learned Slot Assignment + Stronger Latent", "LongMemEval-S", "learned slot assignment"],
        ),
        "agentos_tracks_td035_ws021": _contains_all(
            project_index,
            ["Top next action: `TD-035`", "Active workstreams: `WS-021`"],
        ) and _contains_all(todo, ["`TD-035` `[doing]`"]),
        "v24_plan_exists": v24_plan.exists(),
        "v24_plan_mentions_quality_constraints": _contains_all(
            v24_plan,
            ["LongMemEval-S", "learned slot assignment", "stronger latent", "fallback", "完整 public-data"],
        ),
        "historical_v23_closeout_artifacts_exist": _historical_v23_closeout_artifacts_exist(root),
        "historical_v23_longmemeval_128_exists": historical_longmemeval.exists(),
        "historical_v23_personamem_128_exists": historical_personamem.exists(),
        "historical_full_data_semantic_train_exists": historical_train.exists(),
        "historical_full_data_semantic_eval_exists": historical_eval.exists(),
        "learned_slot_assignment_module_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/lifecycle.py", "src/core_mem/v2/system.py", "src/core_mem/v2/training.py"],
            ["slot_assignment", "slot_assignment_mode", "LifecycleDecision"],
        ),
        "stronger_latent_path_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "src/core_mem/v2/resampler.py", "src/core_mem/v2/decoder.py", "src/core_mem/v2/semantic_outputs.py"],
            ["composed_memory", "cross-attention", "semantic_outputs", "belief_source"],
        ),
        "current_head_v24_train_exists": bool(train_payload),
        "current_head_v24_eval_exists": bool(eval_payload),
        "current_head_v24_eval_token_f1_ge_0_93": float(metrics.get("token_f1", 0.0)) >= 0.93,
        "current_head_v24_eval_field_f1_ge_0_93": float(metrics.get("field_f1", 0.0)) >= 0.93,
        "current_head_v24_slot_assignment_token_f1_ge_0_10": float(
            slot_assignment_metrics.get("token_f1", slot_assignment_metrics.get("field_f1", 0.0))
        ) >= 0.10,
        "current_head_longmemeval_v24_64_exists": bool(longmemeval_payload)
        and int(longmemeval_payload.get("sample_count", 0)) >= 64
        and longmemeval_payload.get("status") == "completed",
        "current_head_longmemeval_v24_128_exists": bool(longmemeval_payload)
        and int(longmemeval_payload.get("sample_count", 0)) >= 128
        and longmemeval_payload.get("status") == "completed",
        "current_head_longmemeval_v24_provider_ge_10": int(
            (longmemeval_payload or {}).get("provider_exact_match", 0)
        ) >= 10,
        "current_head_longmemeval_v24_local_ge_10": int(
            (longmemeval_payload or {}).get("local_exact_match", 0)
        ) >= 10,
        "current_head_longmemeval_v24_analysis_exists": bool(analysis_payload)
        and any(
            key in analysis_payload
            for key in ["top_error_clusters", "error_clusters", "layered_breakdown", "failure_buckets"]
        ),
        "current_head_v24_online_gain_positive_ge_4": bool(gain_payload)
        and bool(gain_payload.get("positive_gain"))
        and (
            int(gain_payload.get("delta_provider_exact_match", 0)) >= 4
            or int(gain_payload.get("delta_local_exact_match", 0)) >= 4
        ),
        "current_head_personamem_v24_guard_holds": bool(personamem_payload)
        and int(personamem_payload.get("sample_count", 0)) >= 128
        and personamem_payload.get("status") == "completed"
        and int(personamem_payload.get("provider_exact_match", 0)) >= 36
        and int(personamem_payload.get("local_exact_match", 0)) >= 14,
    }

    score = sum(1 for value in checks.values() if value)
    return {
        "metric": "stage2_v24_longrun_score",
        "score": score,
        "total": len(checks),
        "checks": checks,
        "head": _current_head(root),
        "trained_eval_metrics": metrics,
        "slot_assignment_metrics": slot_assignment_metrics,
        "gain_payload": gain_payload or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v24_longrun(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
