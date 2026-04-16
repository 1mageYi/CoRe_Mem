"""Mechanical verifier for the longer-range v2.3 learned-slot-assignment run."""

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


def _code_contains_any(root: Path, paths: list[str], patterns: list[str]) -> bool:
    for rel in paths:
        text = _read_text(root / rel)
        if text and any(pattern in text for pattern in patterns):
            return True
    return False


def _current_head_slot_assignment_payload(root: Path, name: str) -> dict[str, Any] | None:
    return _artifact_matches_head(root / "outputs_v2" / "artifacts" / name, _current_head(root))


def _historical_v22_closeout_artifacts_exist(root: Path) -> bool:
    required = [
        root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_train.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_local_eval.json",
        root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_semantic_analysis.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_online_gain.json",
        root / "outputs_v2" / "outputs_v2",  # impossible sentinel removed below
    ]
    required = required[:-1]
    required.extend(
        [
            root / "outputs_v2" / "evals_benchmark" / "20260416T021743Z_stage2_memory_canary.json",
            root / "outputs_v2" / "evals_benchmark" / "20260416T024146Z_stage2_memory_canary.json",
        ]
    )
    return all(path.exists() for path in required)


def compute_v23_longrun(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    v23_plan = root / "docs" / "v23_plan.md"
    v23_longrun_plan = root / "docs" / "v23_longrun_plan.md"

    longmemeval_semantic_baseline = root / "outputs_v2" / "evals_benchmark" / "20260416T021743Z_stage2_memory_canary.json"
    personamem_semantic_baseline = root / "outputs_v2" / "evals_benchmark" / "20260416T024146Z_stage2_memory_canary.json"
    semantic_analysis = root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_semantic_analysis.json"
    semantic_gain = root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_online_gain.json"

    train_payload = _current_head_slot_assignment_payload(root, "latest_stage2_slot_assignment_train.json")
    eval_payload = _current_head_slot_assignment_payload(root, "latest_stage2_slot_assignment_eval.json")
    gain_payload = _current_head_slot_assignment_payload(root, "latest_stage2_slot_assignment_gain.json")
    longmemeval_payload = _current_head_slot_assignment_payload(root, "latest_longmemeval_stage2_slot_assignment_canary.json")
    personamem_payload = _current_head_slot_assignment_payload(root, "latest_personamem_stage2_slot_assignment_canary.json")

    eval_metrics = ((eval_payload or {}).get("trained_eval") or {}).get("metrics") or {}
    head = _current_head(root)

    checks = {
        "current_status_tracks_td034_v23_longrun": _contains_all(
            current_status,
            ["`TD-034`", "`v2.3`", "LongMemEval-S", "learned slot assignment"],
        ),
        "implementation_plan_tracks_stage_s": _contains_all(
            implementation_plan,
            ["阶段 S：V2.3 Long-Run Stronger Learned Slot Assignment", "LongMemEval-S", "learned slot assignment"],
        ),
        "agentos_tracks_td034_ws020": _contains_all(
            project_index,
            ["Top next action: `TD-034`", "Active workstreams: `WS-020`"],
        ) and _contains_all(todo, ["`TD-034` `[doing]`"]),
        "v23_plan_exists": v23_plan.exists(),
        "v23_longrun_plan_exists": v23_longrun_plan.exists(),
        "v23_longrun_mentions_multi_gpu_no_fallback": _contains_all(
            v23_longrun_plan,
            ["多卡", "LongMemEval-S", "learned slot assignment", "fallback"],
        ),
        "historical_v22_closeout_artifacts_exist": _historical_v22_closeout_artifacts_exist(root),
        "v22_longmemeval_128_baseline_exists": longmemeval_semantic_baseline.exists(),
        "v22_personamem_128_baseline_exists": personamem_semantic_baseline.exists(),
        "v22_semantic_analysis_exists": semantic_analysis.exists(),
        "v22_semantic_online_gain_exists": semantic_gain.exists(),
        "learned_slot_assignment_module_exists": _code_contains_any(
            root,
            ["src/core_mem/v2/lifecycle.py", "src/core_mem/v2/system.py", "src/core_mem/v2/encoder.py"],
            ["slot_assignment", "slot assign", "slot_assignment_mode"],
        ),
        "system_supports_learned_slot_assignment_toggle": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "scripts/run_stage2_memory_canary.py"],
            ["learned_slot_assignment", "slot_assignment_mode", "--slot-assignment-mode"],
        ),
        "training_supports_slot_assignment_supervision": _code_contains_any(
            root,
            ["src/core_mem/v2/training.py", "scripts/train_stage2.py", "configs/stage2_train.yaml"],
            ["slot_assignment", "candidate-slot", "action classification"],
        ),
        "slot_assignment_eval_metrics_exist": bool(eval_payload) and (
            "slot_assignment" in json.dumps(eval_payload, ensure_ascii=False)
            or "slot_assignment_accuracy" in json.dumps(eval_payload, ensure_ascii=False)
        ),
        "current_head_slot_assignment_train_exists": bool(train_payload),
        "current_head_slot_assignment_gain_positive": bool(gain_payload) and bool(gain_payload.get("positive_gain")),
        "current_head_longmemeval_slot_assignment_64_exists": bool(longmemeval_payload)
        and int(longmemeval_payload.get("sample_count", 0)) >= 64
        and longmemeval_payload.get("status") == "completed",
        "current_head_longmemeval_slot_assignment_128_exists": bool(longmemeval_payload)
        and int(longmemeval_payload.get("sample_count", 0)) >= 128
        and longmemeval_payload.get("status") == "completed",
        "current_head_personamem_slot_assignment_128_exists": bool(personamem_payload)
        and int(personamem_payload.get("sample_count", 0)) >= 128
        and personamem_payload.get("status") == "completed",
        "current_head_longmemeval_beats_v22_baseline": bool(gain_payload)
        and (
            int(gain_payload.get("delta_provider_exact_match", 0)) >= 2
            or int(gain_payload.get("delta_local_exact_match", 0)) >= 2
        ),
        "current_head_personamem_guard_holds": bool(personamem_payload)
        and int(personamem_payload.get("provider_exact_match", 0)) >= 30
        and int(personamem_payload.get("local_exact_match", 0)) >= 10,
    }

    score = sum(1 for value in checks.values() if value)
    return {
        "metric": "stage2_v23_longrun_score",
        "score": score,
        "total": len(checks),
        "checks": checks,
        "head": head,
        "slot_assignment_train": train_payload or {},
        "slot_assignment_eval_metrics": eval_metrics,
        "slot_assignment_gain": gain_payload or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v23_longrun(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
