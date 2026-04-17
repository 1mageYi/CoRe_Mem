"""Mechanical verifier for the v2.7 32k teacher-first long run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else "unknown"


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


def _trained_eval_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    return (((payload or {}).get("trained_eval")) or {}).get("metrics") or {}


def publish_v27_artifacts(
    *,
    root: Path,
    train_summary_path: Path,
    val_eval_path: Path,
    test_eval_path: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    test_manifest_path: Path,
) -> dict[str, Any]:
    current_head = _current_head(root)
    train_summary = _read_json(train_summary_path)
    val_eval = _read_json(val_eval_path)
    test_eval = _read_json(test_eval_path)
    train_manifest = _read_json(train_manifest_path)
    val_manifest = _read_json(val_manifest_path)
    test_manifest = _read_json(test_manifest_path)
    if not all([train_summary, val_eval, test_eval, train_manifest, val_manifest, test_manifest]):
        raise FileNotFoundError("Missing v27 publish inputs.")

    artifact_root = root / "outputs_v2" / "artifacts"

    train_payload = {
        "artifact_type": "stage2_v27_train",
        "commit_hash": current_head,
        "source_split": "train",
        "prepared_manifest": str(train_manifest_path),
        "task_counts": train_manifest.get("task_counts", {}),
        **train_summary,
    }
    _write_json(artifact_root / "latest_stage2_v27_train.json", train_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v27_train.json", train_payload)

    timing_payload = {
        "artifact_type": "stage2_v27_training_timing",
        "commit_hash": current_head,
        "source_split": "train",
        "prepared_manifest": str(train_manifest_path),
        "wall_clock_seconds": train_summary.get("wall_clock_seconds"),
        "examples_per_second": train_summary.get("examples_per_second"),
        "steps_per_second": train_summary.get("steps_per_second"),
        "peak_gpu_memory_mb": train_summary.get("peak_gpu_memory_mb"),
        "gpu_name": train_summary.get("gpu_name"),
        "device": train_summary.get("device"),
        "cuda_visible_devices": train_summary.get("cuda_visible_devices"),
        "effective_num_examples": train_summary.get("num_examples"),
        "effective_num_steps": train_summary.get("num_steps"),
        "optimizer_steps": train_summary.get("optimizer_steps"),
        "run_dir": train_summary.get("run_dir"),
        "metrics_path": train_summary.get("metrics_path"),
        "checkpoint_dir": train_summary.get("checkpoint_dir"),
    }
    _write_json(artifact_root / "latest_stage2_v27_training_timing.json", timing_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v27_training_timing.json", timing_payload)

    val_trained = _trained_eval_metrics(val_eval)
    test_trained = _trained_eval_metrics(test_eval)
    eval_payload = {
        "artifact_type": "stage2_v27_eval",
        "commit_hash": current_head,
        "source_split": "val",
        "prepared_manifest": str(val_manifest_path),
        "task_counts": val_manifest.get("task_counts", {}),
        "result_path": str(val_eval_path),
        **val_eval,
    }
    _write_json(artifact_root / "latest_stage2_v27_eval.json", eval_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v27_eval.json", eval_payload)

    internal_test_payload = {
        "artifact_type": "stage2_v27_internal_test",
        "commit_hash": current_head,
        "val_prepared_manifest": str(val_manifest_path),
        "test_prepared_manifest": str(test_manifest_path),
        "val_result_path": str(val_eval_path),
        "test_result_path": str(test_eval_path),
        "val_task_counts": val_manifest.get("task_counts", {}),
        "test_task_counts": test_manifest.get("task_counts", {}),
        "val_metrics": val_trained,
        "test_metrics": test_trained,
        "generalization_gap_token_f1": float(test_trained.get("token_f1", 0.0)) - float(val_trained.get("token_f1", 0.0)),
        "generalization_gap_field_f1": float(test_trained.get("field_f1", 0.0)) - float(val_trained.get("field_f1", 0.0)),
        "generalization_gap_exact_match": float(test_trained.get("exact_match", 0.0)) - float(val_trained.get("exact_match", 0.0)),
        "gate_passed": (
            float(test_trained.get("token_f1", 0.0)) >= max(float(val_trained.get("token_f1", 0.0)) - 0.20, 0.0)
            and float(test_trained.get("field_f1", 0.0)) >= max(float(val_trained.get("field_f1", 0.0)) - 0.20, 0.0)
        ),
        "gate_mode": "local_intrinsic_generalization",
    }
    _write_json(artifact_root / "latest_stage2_v27_internal_test.json", internal_test_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v27_internal_test.json", internal_test_payload)

    holdout_long = _read_json(artifact_root / "latest_longmemeval_stage2_v26_canary.json") or {}
    holdout_persona = _read_json(artifact_root / "latest_personamem_stage2_v26_canary.json") or {}
    holdout_payload = {
        "artifact_type": "stage2_v27_holdout_summary",
        "commit_hash": current_head,
        "holdout_only": True,
        "teacher_labels_used_for_training": False,
        "benchmark_runs_executed": 0,
        "training_manifests": [
            str(train_manifest_path),
            str(val_manifest_path),
            str(test_manifest_path),
        ],
        "retained_benchmark_holdouts": [
            {
                "benchmark": "longmemeval_s",
                "summary_path": str(artifact_root / "latest_longmemeval_stage2_v26_canary.json"),
                "provider_exact_match": int(holdout_long.get("provider_exact_match", 0)),
                "local_exact_match": int(holdout_long.get("local_exact_match", 0)),
            },
            {
                "benchmark": "personamem",
                "summary_path": str(artifact_root / "latest_personamem_stage2_v26_canary.json"),
                "provider_exact_match": int(holdout_persona.get("provider_exact_match", 0)),
                "local_exact_match": int(holdout_persona.get("local_exact_match", 0)),
            },
        ],
        "note": "No benchmark outputs were used as training supervision. Holdout benchmarks remain retained external guards during the 32k internal pilot.",
    }
    _write_json(artifact_root / "latest_stage2_v27_holdout_summary.json", holdout_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v27_holdout_summary.json", holdout_payload)

    return {
        "train": train_payload,
        "timing": timing_payload,
        "eval": eval_payload,
        "internal_test": internal_test_payload,
        "holdout_summary": holdout_payload,
    }


def compute_v27_longrun(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(slug: str, passed: bool, detail: str) -> None:
        checks.append({"slug": slug, "passed": bool(passed), "detail": detail})

    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo_doc = root / ".agent-os" / "todo.md"
    v27_plan = root / "docs" / "v27_plan.md"

    add(
        "current_status_tracks_td038",
        _contains_all(current_status, ["`TD-038`", "`v2.7`", "32k", "teacher-first"]),
        "current_status should track TD-038 / v2.7 / 32k teacher-first",
    )
    add(
        "implementation_plan_mentions_32k_split",
        _contains_all(implementation_plan, ["32k", "24k train", "4k val", "4k test"]),
        "implementation_plan should mention 32k 24k/4k/4k split",
    )
    add(
        "implementation_plan_mentions_teacher_gpu2",
        _contains_all(implementation_plan, ["MiniMax-M2.7", "gpu2", "训练耗时"]),
        "implementation_plan should mention MiniMax-M2.7 teacher, gpu2 and timing",
    )
    add(
        "project_index_tracks_ws024",
        _contains_all(project_index, ["`TD-038 / WS-024`", "`v2.7`"]),
        "project-index should track TD-038 / WS-024",
    )
    add(
        "todo_tracks_td038_doing",
        _contains_all(todo_doc, ["`TD-038`", "`[doing]`", "32k"]),
        "todo should track TD-038 as doing",
    )
    add(
        "v27_plan_exists",
        v27_plan.exists(),
        "docs/v27_plan.md should exist",
    )
    add(
        "v27_plan_freezes_core_residual",
        _contains_all(v27_plan, ["不改 `core / residual`", "冻结边界"]),
        "v27 plan should freeze core/residual",
    )
    add(
        "v27_plan_bans_shortcuts",
        _contains_all(v27_plan, ["不做任何 `fallback`", "不做任何 `shortcut`", "benchmark-specific heuristic"]),
        "v27 plan should ban fallback/shortcut/benchmark-specific heuristic",
    )
    add(
        "v27_plan_uses_32k_anchor",
        _contains_all(v27_plan, ["32k", "24k train", "4k val", "4k test"]),
        "v27 plan should anchor on 32k with 24k/4k/4k split",
    )
    add(
        "v27_plan_teacher_and_gpu2",
        _contains_all(v27_plan, ["MiniMax-M2.7", "gpu2", "训练耗时"]),
        "v27 plan should use MiniMax-M2.7 teacher and gpu2 with timing",
    )
    add(
        "v27_plan_defers_full_data",
        _contains_all(v27_plan, ["只有在 32k internal test work well 后", "暂不进入 full-data 训练"]),
        "v27 plan should defer full-data until 32k works",
    )

    v26_write = _read_json(root / "outputs_v2" / "artifacts" / "latest_stage2_v26_write_gain.json") or {}
    v26_belief = _read_json(root / "outputs_v2" / "artifacts" / "latest_stage2_v26_belief_gain.json") or {}
    v26_long = _read_json(root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_canary.json") or {}
    v26_persona = _read_json(root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v26_canary.json") or {}

    add(
        "v26_write_gain_retained_positive",
        bool(v26_write.get("positive_gain", False)),
        "v26 retained write gain should stay positive",
    )
    add(
        "v26_belief_gain_retained_positive",
        bool(v26_belief.get("positive_gain", False)),
        "v26 retained belief gain should stay positive",
    )
    add(
        "v26_longmemeval_baseline_retained",
        int(v26_long.get("provider_exact_match", 0)) >= 11 and int(v26_long.get("local_exact_match", 0)) >= 11,
        "v26 retained LongMemEval-S baseline should stay at least 11/11",
    )
    add(
        "v26_personamem_guard_retained",
        int(v26_persona.get("provider_exact_match", 0)) >= 44 and int(v26_persona.get("local_exact_match", 0)) >= 33,
        "v26 retained PersonaMem guard should stay at least 44/33",
    )

    add(
        "v27_32k_split_artifact_exists",
        _artifact_exists(root, "latest_stage2_v27_32k_split.json"),
        "v27 should publish 32k source-level split artifact",
    )
    add(
        "v27_32k_manifest_exists",
        _artifact_exists(root, "latest_stage2_v27_32k_manifest.json"),
        "v27 should publish 32k manifest artifact",
    )
    add(
        "v27_32k_audit_exists",
        _artifact_exists(root, "latest_stage2_v27_32k_audit.json"),
        "v27 should publish 32k data-quality audit artifact",
    )
    add(
        "v27_teacher_observation_exists",
        _artifact_exists(root, "latest_stage2_v27_teacher_observation.json"),
        "v27 should publish teacher observation artifact",
    )
    add(
        "v27_teacher_slot_assignment_exists",
        _artifact_exists(root, "latest_stage2_v27_teacher_slot_assignment.json"),
        "v27 should publish teacher slot-assignment artifact",
    )
    add(
        "v27_teacher_belief_exists",
        _artifact_exists(root, "latest_stage2_v27_teacher_belief.json"),
        "v27 should publish teacher belief-field artifact",
    )
    add(
        "v27_train_exists",
        _artifact_exists(root, "latest_stage2_v27_train.json"),
        "v27 should publish 32k train artifact",
    )
    add(
        "v27_eval_exists",
        _artifact_exists(root, "latest_stage2_v27_eval.json"),
        "v27 should publish 32k eval artifact",
    )
    add(
        "v27_training_timing_exists",
        _artifact_exists(root, "latest_stage2_v27_training_timing.json"),
        "v27 should publish 32k training timing artifact",
    )
    add(
        "v27_internal_test_exists",
        _artifact_exists(root, "latest_stage2_v27_internal_test.json"),
        "v27 should publish internal test artifact",
    )
    add(
        "v27_holdout_summary_exists",
        _artifact_exists(root, "latest_stage2_v27_holdout_summary.json"),
        "v27 should publish holdout summary artifact",
    )

    score = sum(1 for item in checks if item["passed"])
    total = len(checks)
    return {
        "metric_name": "stage2_v27_longrun_score",
        "score": score,
        "total": total,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify stage-2 v2.7 32k teacher-first long-run status")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--publish-artifacts", action="store_true")
    parser.add_argument("--train-summary", type=Path)
    parser.add_argument("--val-eval", type=Path)
    parser.add_argument("--test-eval", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.publish_artifacts:
        required = [
            args.train_summary,
            args.val_eval,
            args.test_eval,
            args.train_manifest,
            args.val_manifest,
            args.test_manifest,
        ]
        if any(item is None for item in required):
            raise ValueError("--publish-artifacts requires train/eval/manifest paths for train/val/test.")
        payload = publish_v27_artifacts(
            root=root,
            train_summary_path=args.train_summary.resolve(),
            val_eval_path=args.val_eval.resolve(),
            test_eval_path=args.test_eval.resolve(),
            train_manifest_path=args.train_manifest.resolve(),
            val_manifest_path=args.val_manifest.resolve(),
            test_manifest_path=args.test_manifest.resolve(),
        )
    else:
        payload = compute_v27_longrun(root)
    if args.score_only:
        print(payload["score"])
        return 0
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"{payload['metric_name']}: {payload['score']}/{payload['total']}")
    for check in payload["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['slug']}: {check['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
