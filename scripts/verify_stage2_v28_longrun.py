"""Mechanical verifier for the v2.8 teacher-quality 32k generalization run."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = compute_v28_longrun(REPO_ROOT)
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
