"""Mechanical verifier for the v2.6 gain-first long run."""

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


def _historical_v25_artifacts_exist(root: Path) -> bool:
    required = [
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_train.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_eval.json",
        root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v25_canary.json",
        root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v25_canary.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_write_gain.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_retrieve_gain.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_belief_gain.json",
        root / "outputs_v2" / "artifacts" / "latest_stage2_v25_full_benchmark.json",
    ]
    return all(path.exists() for path in required)


def _trained_metrics(eval_payload: dict[str, Any] | None) -> dict[str, Any]:
    return ((eval_payload or {}).get("trained_eval") or {}).get("metrics") or {}


def _gain_positive(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    if payload.get("positive_gain") is True:
        return True
    for key in ("delta_vs_v25_retained", "delta_provider_exact_match", "delta_local_exact_match"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def compute_v26_longrun(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    v26_plan = root / "docs" / "v26_plan.md"

    v25_eval_payload = _current_head_payload(root, "latest_stage2_v25_eval.json")
    v25_longmemeval_payload = _current_head_payload(root, "latest_longmemeval_stage2_v25_canary.json")
    v25_personamem_payload = _current_head_payload(root, "latest_personamem_stage2_v25_canary.json")

    v26_train_payload = _current_head_payload(root, "latest_stage2_v26_train.json")
    v26_eval_payload = _current_head_payload(root, "latest_stage2_v26_eval.json")
    v26_write_gain = _current_head_payload(root, "latest_stage2_v26_write_gain.json")
    v26_retrieve_gain = _current_head_payload(root, "latest_stage2_v26_retrieve_gain.json")
    v26_belief_gain = _current_head_payload(root, "latest_stage2_v26_belief_gain.json")
    v26_longmemeval_payload = _current_head_payload(root, "latest_longmemeval_stage2_v26_canary.json")
    v26_personamem_payload = _current_head_payload(root, "latest_personamem_stage2_v26_canary.json")
    v26_analysis_payload = _current_head_payload(root, "latest_longmemeval_stage2_v26_analysis.json")
    v26_full_benchmark_payload = _current_head_payload(root, "latest_stage2_v26_full_benchmark.json")

    v25_trained_metrics = _trained_metrics(v25_eval_payload)
    v26_trained_metrics = _trained_metrics(v26_eval_payload)
    v25_long_provider = int((v25_longmemeval_payload or {}).get("provider_exact_match", 0))
    v25_long_local = int((v25_longmemeval_payload or {}).get("local_exact_match", 0))
    v25_persona_provider = int((v25_personamem_payload or {}).get("provider_exact_match", 0))
    v25_persona_local = int((v25_personamem_payload or {}).get("local_exact_match", 0))
    v26_long_provider = int((v26_longmemeval_payload or {}).get("provider_exact_match", 0))
    v26_long_local = int((v26_longmemeval_payload or {}).get("local_exact_match", 0))
    v26_persona_provider = int((v26_personamem_payload or {}).get("provider_exact_match", 0))
    v26_persona_local = int((v26_personamem_payload or {}).get("local_exact_match", 0))

    component_positive = any(_gain_positive(payload) for payload in [v26_write_gain, v26_retrieve_gain, v26_belief_gain])

    checks = {
        "current_status_tracks_td037_v26": _contains_all(
            current_status,
            ["`TD-037`", "`v2.6`", "LongMemEval-S", "write -> retrieve -> belief", "真实正增益"],
        ),
        "implementation_plan_tracks_stage_w_v26": _contains_all(
            implementation_plan,
            ["阶段 W：V2.6 Gain-First Long Run", "LongMemEval-S", "正增益"],
        ),
        "agentos_tracks_td037_ws023": _contains_all(
            project_index,
            ["`TD-037 / WS-023`", "`v2.6 gain-first long-run`"],
        ) and _contains_all(todo, ["`TD-037` `[doing]`"]),
        "v26_plan_exists": v26_plan.exists(),
        "v26_plan_freezes_core_residual": _contains_all(
            v26_plan,
            ["不改 `core / residual`", "冻结边界", "write", "retrieve", "belief"],
        ),
        "v26_plan_bans_shortcuts_and_leakage": _contains_all(
            v26_plan,
            ["不做任何 `fallback`", "不做任何 `shortcut`", "不使用 benchmark label / answer / protocol 输出作为训练 supervision", "holdout evaluation"],
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
        "historical_v25_baseline_artifacts_exist": _historical_v25_artifacts_exist(root),
        "current_head_v25_eval_still_strong": float(v25_trained_metrics.get("token_f1", 0.0)) >= 0.99
        and float(v25_trained_metrics.get("field_f1", 0.0)) >= 0.99,
        "current_head_v25_longmemeval_128_exists": bool(v25_longmemeval_payload)
        and int(v25_longmemeval_payload.get("sample_count", 0)) >= 128
        and v25_longmemeval_payload.get("status") == "completed",
        "current_head_v25_personamem_128_exists": bool(v25_personamem_payload)
        and int(v25_personamem_payload.get("sample_count", 0)) >= 128
        and v25_personamem_payload.get("status") == "completed",
        "current_head_v26_train_exists": bool(v26_train_payload),
        "current_head_v26_eval_exists": bool(v26_eval_payload),
        "current_head_v26_write_gain_exists": bool(v26_write_gain),
        "current_head_v26_retrieve_gain_exists": bool(v26_retrieve_gain),
        "current_head_v26_belief_gain_exists": bool(v26_belief_gain),
        "at_least_one_component_gain_positive": component_positive,
        "current_head_v26_longmemeval_128_exists": bool(v26_longmemeval_payload)
        and int(v26_longmemeval_payload.get("sample_count", 0)) >= 128
        and v26_longmemeval_payload.get("status") == "completed",
        "current_head_v26_longmemeval_128_beats_v25": v26_long_provider > v25_long_provider
        and v26_long_local > v25_long_local,
        "current_head_v26_personamem_128_guard_holds": bool(v26_personamem_payload)
        and int(v26_personamem_payload.get("sample_count", 0)) >= 128
        and v26_personamem_payload.get("status") == "completed"
        and v26_persona_provider >= v25_persona_provider
        and v26_persona_local >= v25_persona_local,
        "current_head_v26_longmemeval_analysis_exists": bool(v26_analysis_payload)
        and any(key in v26_analysis_payload for key in ["error_clusters", "layered_breakdown", "failure_buckets"]),
        "current_head_v26_full_benchmark_exists": bool(v26_full_benchmark_payload)
        and int(v26_full_benchmark_payload.get("num_benchmarks", 0)) >= 1
        and bool(v26_full_benchmark_payload.get("holdout_only")),
        "current_head_v26_eval_stays_strong": float(v26_trained_metrics.get("token_f1", 0.0)) >= 0.99
        and float(v26_trained_metrics.get("field_f1", 0.0)) >= 0.99,
    }

    score = sum(1 for value in checks.values() if value)
    return {
        "metric": "stage2_v26_longrun_score",
        "score": score,
        "total": len(checks),
        "checks": checks,
        "head": _current_head(root),
        "v25_trained_eval_metrics": v25_trained_metrics,
        "v26_trained_eval_metrics": v26_trained_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v26_longrun(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
