"""Mechanical verifier for the v2.8 teacher-quality 32k generalization run."""

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


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


def _artifact_json(root: Path, name: str) -> dict[str, Any] | None:
    return _read_json(root / "outputs_v2" / "artifacts" / name)


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


def _trained_eval_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    return (((payload or {}).get("trained_eval")) or {}).get("metrics") or {}


def publish_v28_artifacts(
    *,
    root: Path,
    silver_train_summary_path: Path,
    silver_val_eval_path: Path,
    silver_test_eval_path: Path,
    silver_train_manifest_path: Path,
    silver_val_manifest_path: Path,
    silver_test_manifest_path: Path,
    teacher_train_summary_path: Path,
    teacher_val_eval_path: Path,
    teacher_test_eval_path: Path,
    teacher_train_manifest_path: Path,
    teacher_val_manifest_path: Path,
    teacher_test_manifest_path: Path,
) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    current_head = _current_head(root)

    silver_train = _read_json(silver_train_summary_path)
    silver_val_eval = _read_json(silver_val_eval_path)
    silver_test_eval = _read_json(silver_test_eval_path)
    teacher_train = _read_json(teacher_train_summary_path)
    teacher_val_eval = _read_json(teacher_val_eval_path)
    teacher_test_eval = _read_json(teacher_test_eval_path)
    if not all([silver_train, silver_val_eval, silver_test_eval, teacher_train, teacher_val_eval, teacher_test_eval]):
        raise FileNotFoundError("Missing v28 publish inputs.")

    silver_val_metrics = _trained_eval_metrics(silver_val_eval)
    silver_test_metrics = _trained_eval_metrics(silver_test_eval)
    teacher_val_metrics = _trained_eval_metrics(teacher_val_eval)
    teacher_test_metrics = _trained_eval_metrics(teacher_test_eval)

    silver_payload = {
        "artifact_type": "stage2_v28_silver_baseline",
        "commit_hash": current_head,
        "train_summary_path": str(silver_train_summary_path),
        "val_result_path": str(silver_val_eval_path),
        "test_result_path": str(silver_test_eval_path),
        "train_prepared_manifest": str(silver_train_manifest_path),
        "val_prepared_manifest": str(silver_val_manifest_path),
        "test_prepared_manifest": str(silver_test_manifest_path),
        "train_summary": silver_train,
        "val_metrics": silver_val_metrics,
        "test_metrics": silver_test_metrics,
    }
    _write_json(artifact_root / "latest_stage2_v28_silver_baseline.json", silver_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v28_silver_baseline.json", silver_payload)

    teacher_train_payload = {
        "artifact_type": "stage2_v28_teacher_train",
        "commit_hash": current_head,
        "prepared_manifest": str(teacher_train_manifest_path),
        "train_summary_path": str(teacher_train_summary_path),
        **teacher_train,
    }
    _write_json(artifact_root / "latest_stage2_v28_teacher_train.json", teacher_train_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v28_teacher_train.json", teacher_train_payload)

    teacher_eval_payload = {
        "artifact_type": "stage2_v28_teacher_eval",
        "commit_hash": current_head,
        "val_prepared_manifest": str(teacher_val_manifest_path),
        "test_prepared_manifest": str(teacher_test_manifest_path),
        "val_result_path": str(teacher_val_eval_path),
        "test_result_path": str(teacher_test_eval_path),
        "val_metrics": teacher_val_metrics,
        "test_metrics": teacher_test_metrics,
        "val_eval": teacher_val_eval,
        "test_eval": teacher_test_eval,
    }
    _write_json(artifact_root / "latest_stage2_v28_teacher_eval.json", teacher_eval_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v28_teacher_eval.json", teacher_eval_payload)

    compare_payload = {
        "artifact_type": "stage2_v28_teacher_compare",
        "commit_hash": current_head,
        "silver_val_metrics": silver_val_metrics,
        "silver_test_metrics": silver_test_metrics,
        "teacher_val_metrics": teacher_val_metrics,
        "teacher_test_metrics": teacher_test_metrics,
        "delta_val_token_f1": float(teacher_val_metrics.get("token_f1", 0.0)) - float(silver_val_metrics.get("token_f1", 0.0)),
        "delta_val_field_f1": float(teacher_val_metrics.get("field_f1", 0.0)) - float(silver_val_metrics.get("field_f1", 0.0)),
        "delta_internal_token_f1": float(teacher_test_metrics.get("token_f1", 0.0)) - float(silver_test_metrics.get("token_f1", 0.0)),
        "delta_internal_field_f1": float(teacher_test_metrics.get("field_f1", 0.0)) - float(silver_test_metrics.get("field_f1", 0.0)),
        "delta_internal_exact_match": float(teacher_test_metrics.get("exact_match", 0.0)) - float(silver_test_metrics.get("exact_match", 0.0)),
    }
    _write_json(artifact_root / "latest_stage2_v28_teacher_compare.json", compare_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v28_teacher_compare.json", compare_payload)

    internal_payload = {
        "artifact_type": "stage2_v28_internal_test",
        "commit_hash": current_head,
        "silver_test_metrics": silver_test_metrics,
        "teacher_test_metrics": teacher_test_metrics,
        "delta_internal_token_f1": compare_payload["delta_internal_token_f1"],
        "delta_internal_field_f1": compare_payload["delta_internal_field_f1"],
        "gate_mode": "teacher_vs_silver_internal_delta",
        "gate_passed": (
            compare_payload["delta_internal_token_f1"] > 0.0
            or compare_payload["delta_internal_field_f1"] > 0.0
        ),
    }
    _write_json(artifact_root / "latest_stage2_v28_internal_test.json", internal_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v28_internal_test.json", internal_payload)

    return {
        "silver_baseline": silver_payload,
        "teacher_train": teacher_train_payload,
        "teacher_eval": teacher_eval_payload,
        "teacher_compare": compare_payload,
        "internal_test": internal_payload,
    }


def compute_v28_longrun(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(slug: str, passed: bool, detail: str) -> None:
        checks.append({"slug": slug, "passed": bool(passed), "detail": detail})

    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo_doc = root / ".agent-os" / "todo.md"
    v28_plan = root / "docs" / "v28_plan.md"

    add("current_status_tracks_td039", _contains_all(current_status, ["`TD-039`", "`v2.8`", "teacher-quality", "32k"]), "current_status should track TD-039 / v2.8 teacher-quality 32k")
    add("implementation_plan_mentions_teacher_vs_silver", _contains_all(implementation_plan, ["teacher-vs-silver", "512", "128", "gpu2"]), "implementation_plan should mention teacher-vs-silver and 512/128/128 on gpu2")
    add("project_index_tracks_ws025", _contains_all(project_index, ["`TD-039 / WS-025`", "`v2.8`"]), "project-index should track TD-039 / WS-025")
    add("todo_tracks_td039_doing", _contains_all(todo_doc, ["`TD-039`", "`[doing]`", "teacher-vs-silver"]), "todo should track TD-039 as doing")
    add("v28_plan_exists", v28_plan.exists(), "docs/v28_plan.md should exist")
    add("v28_plan_mentions_no_fallback", _contains_all(v28_plan, ["不做任何 `fallback`", "不做任何 `shortcut`"]), "v28 plan should keep no-fallback / no-shortcut constraints")
    add("v28_plan_mentions_holdout_only", _contains_all(v28_plan, ["holdout-only", "不回流成训练 supervision"]), "v28 plan should keep holdout-only training boundary")
    add("v28_plan_mentions_observation_repair", _contains_all(v28_plan, ["observation teacher", "completed_with_failures", "schema obey rate"]), "v28 plan should explicitly target observation teacher repair")

    for artifact_name in [
        "latest_stage2_v27_32k_split.json",
        "latest_stage2_v27_32k_manifest.json",
        "latest_stage2_v27_32k_audit.json",
        "latest_stage2_v27_teacher_observation.json",
        "latest_stage2_v27_teacher_slot_assignment.json",
        "latest_stage2_v27_teacher_belief.json",
        "latest_stage2_v27_train.json",
        "latest_stage2_v27_eval.json",
        "latest_stage2_v27_training_timing.json",
        "latest_stage2_v27_internal_test.json",
        "latest_stage2_v27_holdout_summary.json",
    ]:
        add(artifact_name.replace(".", "_"), _artifact_exists(root, artifact_name), f"{artifact_name} should exist as retained v2.7 baseline evidence")

    add("v28_teacher_observation_exists", _artifact_exists(root, "latest_stage2_v28_teacher_observation.json"), "v28 teacher observation artifact should exist")
    add("v28_teacher_slot_assignment_exists", _artifact_exists(root, "latest_stage2_v28_teacher_slot_assignment.json"), "v28 teacher slot-assignment artifact should exist")
    add("v28_teacher_belief_exists", _artifact_exists(root, "latest_stage2_v28_teacher_belief.json"), "v28 teacher belief artifact should exist")
    add("v28_teacher_quality_audit_exists", _artifact_exists(root, "latest_stage2_v28_teacher_quality_audit.json"), "v28 teacher quality audit artifact should exist")
    add("v28_silver_baseline_exists", _artifact_exists(root, "latest_stage2_v28_silver_baseline.json"), "v28 silver baseline artifact should exist")
    add("v28_teacher_train_exists", _artifact_exists(root, "latest_stage2_v28_teacher_train.json"), "v28 teacher-enhanced train artifact should exist")
    add("v28_teacher_eval_exists", _artifact_exists(root, "latest_stage2_v28_teacher_eval.json"), "v28 teacher-enhanced eval artifact should exist")
    add("v28_teacher_compare_exists", _artifact_exists(root, "latest_stage2_v28_teacher_compare.json"), "v28 teacher-vs-silver compare artifact should exist")
    add("v28_internal_test_exists", _artifact_exists(root, "latest_stage2_v28_internal_test.json"), "v28 internal generalization artifact should exist")

    teacher_obs = _artifact_json(root, "latest_stage2_v28_teacher_observation.json") or {}
    teacher_slot = _artifact_json(root, "latest_stage2_v28_teacher_slot_assignment.json") or {}
    teacher_belief = _artifact_json(root, "latest_stage2_v28_teacher_belief.json") or {}
    teacher_audit = _artifact_json(root, "latest_stage2_v28_teacher_quality_audit.json") or {}
    teacher_compare = _artifact_json(root, "latest_stage2_v28_teacher_compare.json") or {}
    internal_test = _artifact_json(root, "latest_stage2_v28_internal_test.json") or {}

    add("v28_teacher_observation_success_ge_0_75", float(teacher_obs.get("success_rate", 0.0)) >= 0.75, "v28 observation teacher success_rate should be >= 0.75")
    add("v28_teacher_slot_assignment_count_ge_256", int(teacher_slot.get("total_labeled_examples", 0)) >= 256, "v28 slot-assignment teacher should label at least 256 examples")
    add("v28_teacher_belief_count_ge_256", int(teacher_belief.get("total_labeled_examples", 0)) >= 256, "v28 belief teacher should label at least 256 examples")
    add("v28_teacher_audit_covers_all_three_tasks", set((teacher_audit.get("tasks_covered") or [])) >= {"observation", "slot_assignment", "belief"}, "v28 teacher audit should cover observation/slot_assignment/belief")
    add("v28_teacher_compare_shows_positive_delta", (float(teacher_compare.get("delta_internal_token_f1", 0.0)) > 0.0 or float(teacher_compare.get("delta_internal_field_f1", 0.0)) > 0.0), "v28 teacher-vs-silver compare should show positive internal delta")
    add("v28_internal_gate_passed", bool(internal_test.get("gate_passed", False)), "v28 internal generalization gate should pass")

    passed = sum(1 for check in checks if check["passed"])
    total = len(checks)
    return {"score": passed, "total": total, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify stage-2 v2.8 teacher-quality long-run status")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--publish-artifacts", action="store_true")
    parser.add_argument("--silver-train-summary", type=Path)
    parser.add_argument("--silver-val-eval", type=Path)
    parser.add_argument("--silver-test-eval", type=Path)
    parser.add_argument("--silver-train-manifest", type=Path)
    parser.add_argument("--silver-val-manifest", type=Path)
    parser.add_argument("--silver-test-manifest", type=Path)
    parser.add_argument("--teacher-train-summary", type=Path)
    parser.add_argument("--teacher-val-eval", type=Path)
    parser.add_argument("--teacher-test-eval", type=Path)
    parser.add_argument("--teacher-train-manifest", type=Path)
    parser.add_argument("--teacher-val-manifest", type=Path)
    parser.add_argument("--teacher-test-manifest", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.publish_artifacts:
        required = [
            args.silver_train_summary,
            args.silver_val_eval,
            args.silver_test_eval,
            args.silver_train_manifest,
            args.silver_val_manifest,
            args.silver_test_manifest,
            args.teacher_train_summary,
            args.teacher_val_eval,
            args.teacher_test_eval,
            args.teacher_train_manifest,
            args.teacher_val_manifest,
            args.teacher_test_manifest,
        ]
        if any(item is None for item in required):
            raise ValueError("--publish-artifacts requires silver/teacher train summaries, evals, and manifests.")
        payload = publish_v28_artifacts(
            root=root,
            silver_train_summary_path=args.silver_train_summary.resolve(),
            silver_val_eval_path=args.silver_val_eval.resolve(),
            silver_test_eval_path=args.silver_test_eval.resolve(),
            silver_train_manifest_path=args.silver_train_manifest.resolve(),
            silver_val_manifest_path=args.silver_val_manifest.resolve(),
            silver_test_manifest_path=args.silver_test_manifest.resolve(),
            teacher_train_summary_path=args.teacher_train_summary.resolve(),
            teacher_val_eval_path=args.teacher_val_eval.resolve(),
            teacher_test_eval_path=args.teacher_test_eval.resolve(),
            teacher_train_manifest_path=args.teacher_train_manifest.resolve(),
            teacher_val_manifest_path=args.teacher_val_manifest.resolve(),
            teacher_test_manifest_path=args.teacher_test_manifest.resolve(),
        )
    else:
        payload = compute_v28_longrun(root)
    if args.score_only:
        print(payload["score"])
    elif args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"stage2_v28_longrun_score={payload['score']}/{payload['total']}")
        for check in payload["checks"]:
            marker = "PASS" if check["passed"] else "FAIL"
            print(f"[{marker}] {check['slug']}: {check['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
