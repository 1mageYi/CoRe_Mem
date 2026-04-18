from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v29_longrun import compute_v29_longrun, publish_v29_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v29_longrun_verifier_scores_baseline_when_docs_and_retained_artifacts_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text("`TD-040`\n`v2.9`\nwrite\nlatent\nbelief\n32k\n", encoding="utf-8")
    (repo_root / "docs" / "implementation_plan.md").write_text("`TD-040`\n`v2.9`\nwrite\nlatent\nbelief\n512\n", encoding="utf-8")
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-040 / WS-026`\n`v2.9`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-040` `[doing]`\nwrite\nlatent\nbelief\n", encoding="utf-8")
    (repo_root / "docs" / "v29_plan.md").write_text(
        "不做任何 `fallback`\n不做任何 `shortcut`\n不改 `core / residual`\n双银行结构\n24k train\n4k val\n4k test\nLongMemEval-S 128 -> 512\nPersonaMem 128 -> 512\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v26_write_gain.json": {"positive_gain": True},
        "latest_stage2_v26_belief_gain.json": {"positive_gain": True},
        "latest_longmemeval_stage2_v26_canary.json": {"provider_exact_match": 11, "local_exact_match": 11},
        "latest_personamem_stage2_v26_canary.json": {"provider_exact_match": 44, "local_exact_match": 33},
        "latest_stage2_v27_32k_split.json": {"ok": True},
        "latest_stage2_v27_32k_manifest.json": {"ok": True},
        "latest_stage2_v27_32k_audit.json": {"ok": True},
        "latest_stage2_v27_train.json": {"ok": True},
        "latest_stage2_v27_eval.json": {"ok": True},
        "latest_stage2_v27_training_timing.json": {"ok": True},
        "latest_stage2_v28_teacher_compare.json": {"delta_internal_token_f1": -0.01, "delta_internal_field_f1": 0.0},
        "latest_stage2_v28_internal_test.json": {"gate_passed": False},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v29_longrun(repo_root)
    assert payload["score"] == 25
    assert payload["total"] == 39


def test_v29_longrun_verifier_passes_with_full_gain_and_holdout_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text("`TD-040`\n`v2.9`\nwrite\nlatent\nbelief\n32k\n", encoding="utf-8")
    (repo_root / "docs" / "implementation_plan.md").write_text("`TD-040`\n`v2.9`\nwrite\nlatent\nbelief\n512\n", encoding="utf-8")
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-040 / WS-026`\n`v2.9`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-040` `[doing]`\nwrite\nlatent\nbelief\n", encoding="utf-8")
    (repo_root / "docs" / "v29_plan.md").write_text(
        "不做任何 `fallback`\n不做任何 `shortcut`\n不改 `core / residual`\n双银行结构\n24k train\n4k val\n4k test\nLongMemEval-S 128 -> 512\nPersonaMem 128 -> 512\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v26_write_gain.json": {"positive_gain": True},
        "latest_stage2_v26_belief_gain.json": {"positive_gain": True},
        "latest_longmemeval_stage2_v26_canary.json": {"provider_exact_match": 11, "local_exact_match": 11},
        "latest_personamem_stage2_v26_canary.json": {"provider_exact_match": 44, "local_exact_match": 33},
        "latest_stage2_v27_32k_split.json": {"ok": True},
        "latest_stage2_v27_32k_manifest.json": {"ok": True},
        "latest_stage2_v27_32k_audit.json": {"ok": True},
        "latest_stage2_v27_train.json": {"ok": True},
        "latest_stage2_v27_eval.json": {"ok": True},
        "latest_stage2_v27_training_timing.json": {"ok": True},
        "latest_stage2_v28_teacher_compare.json": {"delta_internal_token_f1": -0.01, "delta_internal_field_f1": 0.0},
        "latest_stage2_v28_internal_test.json": {"gate_passed": False},
        "latest_stage2_v29_write_gain.json": {"positive_gain": True},
        "latest_stage2_v29_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v29_belief_gain.json": {"positive_gain": True},
        "latest_stage2_v29_training_timing.json": {"device": "cuda", "cuda_visible_devices": "2"},
        "latest_stage2_v29_holdout_summary.json": {"longmemeval_sample_count": 512, "personamem_sample_count": 512},
        "latest_longmemeval_stage2_v29_canary.json": {"provider_exact_match": 12, "local_exact_match": 12},
        "latest_personamem_stage2_v29_canary.json": {"provider_exact_match": 44, "local_exact_match": 33},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v29_longrun(repo_root)
    assert payload["score"] == payload["total"] == 39


def test_publish_v29_artifacts_emits_gain_timing_and_holdout_aliases(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    artifact_root.mkdir(parents=True)

    _write_json(
        artifact_root / "latest_stage2_v27_eval.json",
        {
            "trained_eval": {
                "metrics": {"token_f1": 0.0, "field_f1": 0.0, "exact_match": 0.0},
                "per_task": {
                    "retrieval_alignment": {"token_f1": 0.0, "field_f1": 0.0, "exact_match": 0.0},
                    "lifecycle_prediction": {"token_f1": 0.0, "field_f1": 0.0, "exact_match": 0.0},
                    "composition_to_belief": {"token_f1": 0.0, "field_f1": 0.0, "exact_match": 0.0},
                },
            }
        },
    )
    _write_json(artifact_root / "latest_longmemeval_stage2_v26_canary.json", {"provider_exact_match": 11, "local_exact_match": 11})
    _write_json(artifact_root / "latest_personamem_stage2_v26_canary.json", {"provider_exact_match": 44, "local_exact_match": 33})

    run_dir = repo_root / "outputs_v2" / "runs" / "v29"
    eval_dir = repo_root / "outputs_v2" / "evals_local"
    run_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    train_manifest = repo_root / "outputs_v2" / "artifacts" / "stage2_v27_32k" / "train" / "stage2_prepared_samples_manifest.json"
    val_manifest = repo_root / "outputs_v2" / "artifacts" / "stage2_v27_32k" / "val" / "stage2_prepared_samples_manifest.json"
    test_manifest = repo_root / "outputs_v2" / "artifacts" / "stage2_v27_32k" / "test" / "stage2_prepared_samples_manifest.json"
    for manifest in [train_manifest, val_manifest, test_manifest]:
        _write_json(manifest, {"task_counts": {"retrieval_alignment": 10, "lifecycle_prediction": 10, "composition_to_belief": 10}})

    train_summary = run_dir / "execution_summary.json"
    val_eval = eval_dir / "val.json"
    test_eval = eval_dir / "test.json"
    long_summary = repo_root / "outputs_v2" / "evals_benchmark" / "long.json"
    persona_summary = repo_root / "outputs_v2" / "evals_benchmark" / "persona.json"

    _write_json(
        train_summary,
        {
            "device": "cuda",
            "cuda_visible_devices": "2",
            "gpu_name": "RTX A4000",
            "num_examples": 4096,
            "num_steps": 512,
            "optimizer_steps": 512,
            "wall_clock_seconds": 12.0,
            "examples_per_second": 341.0,
            "peak_gpu_memory_mb": 2048.0,
            "run_dir": str(run_dir),
            "metrics_path": str(run_dir / "training_metrics.json"),
            "checkpoint_dir": "outputs_v2/checkpoints/v29",
            "config_snapshot_path": "outputs_v2/runs/v29/config_snapshot.yaml",
        },
    )
    eval_payload = {
        "families": {
            "retrieval_family": {"recall_at_k": 1.0, "ndcg_at_k": 0.9, "support_slot_recall": 1.0},
            "belief_recovery_family": {"joint_belief_accuracy": 1.0, "slot_value_f1": 1.0, "answer_exact_match": 0.9},
        },
        "modules": {
            "resampler": {"joint_belief_accuracy": 1.0, "slot_value_f1": 1.0, "avg_composed_token_count": 8.0},
            "belief_decoder": {"joint_belief_accuracy": 1.0, "slot_value_f1": 1.0, "propagation_accuracy": 0.95},
        },
        "trained_eval": {
            "metrics": {"token_f1": 0.8, "field_f1": 0.7, "exact_match": 0.6},
            "per_task": {
                "retrieval_alignment": {"token_f1": 0.7, "field_f1": 0.1, "exact_match": 0.2},
                "lifecycle_prediction": {"token_f1": 0.8, "field_f1": 0.75, "exact_match": 0.7},
                "composition_to_belief": {"token_f1": 0.9, "field_f1": 0.85, "exact_match": 0.8},
            },
        },
    }
    _write_json(val_eval, eval_payload)
    _write_json(test_eval, eval_payload)
    _write_json(
        long_summary,
        {
            "provider_configured": False,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
            "sample_count": 512,
            "provider_exact_match": 0,
            "local_exact_match": 20,
        },
    )
    _write_json(
        persona_summary,
        {
            "provider_configured": False,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
            "sample_count": 512,
            "provider_exact_match": 0,
            "local_exact_match": 40,
        },
    )

    published = publish_v29_artifacts(
        root=repo_root,
        train_summary_path=train_summary,
        val_eval_path=val_eval,
        test_eval_path=test_eval,
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        test_manifest_path=test_manifest,
        longmemeval_summary_path=long_summary,
        personamem_summary_path=persona_summary,
    )

    assert published["training_timing"]["cuda_visible_devices"] == "2"
    assert published["holdout_summary"]["longmemeval_sample_count"] == 512
    assert published["write_gain"]["positive_gain"] is True
    assert published["latent_gain"]["positive_gain"] is True
    assert published["belief_gain"]["positive_gain"] is True
    assert (artifact_root / "latest_stage2_v29_write_gain.json").exists()
    assert (artifact_root / "latest_stage2_v29_latent_gain.json").exists()
    assert (artifact_root / "latest_stage2_v29_belief_gain.json").exists()
    assert (artifact_root / "latest_stage2_v29_training_timing.json").exists()
    assert (artifact_root / "latest_stage2_v29_holdout_summary.json").exists()
    assert (artifact_root / "latest_longmemeval_stage2_v29_canary.json").exists()
    assert (artifact_root / "latest_personamem_stage2_v29_canary.json").exists()
