from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v28_longrun import compute_v28_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v28_longrun_verifier_scores_baseline_when_v27_and_docs_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text("`TD-039`\n`v2.8`\nteacher-quality\n32k\n", encoding="utf-8")
    (repo_root / "docs" / "implementation_plan.md").write_text("teacher-vs-silver\n512\n128\n128\ngpu2\n", encoding="utf-8")
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-039 / WS-025`\n`v2.8`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-039` `[doing]`\nteacher-vs-silver\n", encoding="utf-8")
    (repo_root / "docs" / "v28_plan.md").write_text("不做任何 `fallback`\n不做任何 `shortcut`\nholdout-only\n不回流成训练 supervision\nobservation teacher\ncompleted_with_failures\nschema obey rate\n", encoding="utf-8")

    for name in [
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
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, {"ok": True})

    payload = compute_v28_longrun(repo_root)
    assert payload["score"] == 19
    assert payload["total"] == 34


def test_v28_longrun_verifier_passes_with_teacher_quality_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text("`TD-039`\n`v2.8`\nteacher-quality\n32k\n", encoding="utf-8")
    (repo_root / "docs" / "implementation_plan.md").write_text("teacher-vs-silver\n512\n128\n128\ngpu2\n", encoding="utf-8")
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-039 / WS-025`\n`v2.8`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-039` `[doing]`\nteacher-vs-silver\n", encoding="utf-8")
    (repo_root / "docs" / "v28_plan.md").write_text("不做任何 `fallback`\n不做任何 `shortcut`\nholdout-only\n不回流成训练 supervision\nobservation teacher\ncompleted_with_failures\nschema obey rate\n", encoding="utf-8")

    for name in [
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
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, {"ok": True})

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_teacher_observation.json", {"success_rate": 0.80})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_teacher_slot_assignment.json", {"total_labeled_examples": 256})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_teacher_belief.json", {"total_labeled_examples": 256})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_teacher_quality_audit.json", {"tasks_covered": ["observation", "slot_assignment", "belief"]})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_silver_baseline.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_teacher_train.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_teacher_eval.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_teacher_compare.json", {"delta_internal_token_f1": 0.01, "delta_internal_field_f1": 0.0})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v28_internal_test.json", {"gate_passed": True})

    payload = compute_v28_longrun(repo_root)
    assert payload["score"] == payload["total"] == 34
