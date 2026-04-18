from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v28_longrun import compute_v28_longrun, publish_v28_artifacts


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


def test_publish_v28_artifacts_writes_compare_and_gate(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    run_root = repo_root / "outputs_v2" / "runs"
    eval_root = repo_root / "outputs_v2" / "evals_local"
    manifest_root = repo_root / "outputs_v2" / "manifests"
    artifact_root.mkdir(parents=True)
    run_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)

    silver_train = run_root / "silver_train.json"
    silver_val = eval_root / "silver_val.json"
    silver_test = eval_root / "silver_test.json"
    teacher_train = run_root / "teacher_train.json"
    teacher_val = eval_root / "teacher_val.json"
    teacher_test = eval_root / "teacher_test.json"
    silver_manifest = manifest_root / "silver.json"
    teacher_manifest = manifest_root / "teacher.json"

    _write_json(silver_train, {"run_dir": "silver", "num_examples": 128})
    _write_json(teacher_train, {"run_dir": "teacher", "num_examples": 128})
    _write_json(
        silver_val,
        {"trained_eval": {"metrics": {"token_f1": 0.20, "field_f1": 0.10, "exact_match": 0.0}}},
    )
    _write_json(
        silver_test,
        {"trained_eval": {"metrics": {"token_f1": 0.18, "field_f1": 0.09, "exact_match": 0.0}}},
    )
    _write_json(
        teacher_val,
        {"trained_eval": {"metrics": {"token_f1": 0.24, "field_f1": 0.12, "exact_match": 0.0}}},
    )
    _write_json(
        teacher_test,
        {"trained_eval": {"metrics": {"token_f1": 0.22, "field_f1": 0.11, "exact_match": 0.0}}},
    )
    _write_json(silver_manifest, {"task_files": {}})
    _write_json(teacher_manifest, {"task_files": {}})

    published = publish_v28_artifacts(
        root=repo_root,
        silver_train_summary_path=silver_train,
        silver_val_eval_path=silver_val,
        silver_test_eval_path=silver_test,
        silver_train_manifest_path=silver_manifest,
        silver_val_manifest_path=silver_manifest,
        silver_test_manifest_path=silver_manifest,
        teacher_train_summary_path=teacher_train,
        teacher_val_eval_path=teacher_val,
        teacher_test_eval_path=teacher_test,
        teacher_train_manifest_path=teacher_manifest,
        teacher_val_manifest_path=teacher_manifest,
        teacher_test_manifest_path=teacher_manifest,
    )

    assert published["teacher_compare"]["delta_internal_token_f1"] > 0.0
    assert published["internal_test"]["gate_passed"] is True
    assert (artifact_root / "latest_stage2_v28_silver_baseline.json").exists()
    assert (artifact_root / "latest_stage2_v28_teacher_compare.json").exists()
