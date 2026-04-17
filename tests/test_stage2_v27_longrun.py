from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v27_longrun import compute_v27_longrun, publish_v27_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v27_longrun_verifier_passes_when_teacher_first_32k_state_exists(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-038`\n`v2.7`\n32k\nteacher-first\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "32k\n24k train\n4k val\n4k test\nMiniMax-M2.7\ngpu2\n训练耗时\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "`TD-038 / WS-024`\n`v2.7`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text(
        "`TD-038` `[doing]`\n32k\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "v27_plan.md").write_text(
        "冻结边界\n不改 `core / residual`\n不做任何 `fallback`\n不做任何 `shortcut`\nbenchmark-specific heuristic\n32k\n24k train\n4k val\n4k test\nMiniMax-M2.7\ngpu2\n训练耗时\n只有在 32k internal test work well 后\n暂不进入 full-data 训练\n",
        encoding="utf-8",
    )

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_write_gain.json", {"positive_gain": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_belief_gain.json", {"positive_gain": True})
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_canary.json",
        {"provider_exact_match": 11, "local_exact_match": 11},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v26_canary.json",
        {"provider_exact_match": 44, "local_exact_match": 33},
    )

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

    payload = compute_v27_longrun(repo_root)
    assert payload["score"] == payload["total"] == 26


def test_v27_publish_artifacts_raise_score_for_real_train_eval_timing_chain(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-038`\n`v2.7`\n32k\nteacher-first\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "32k\n24k train\n4k val\n4k test\nMiniMax-M2.7\ngpu2\n训练耗时\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "`TD-038 / WS-024`\n`v2.7`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text(
        "`TD-038` `[doing]`\n32k\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "v27_plan.md").write_text(
        "冻结边界\n不改 `core / residual`\n不做任何 `fallback`\n不做任何 `shortcut`\nbenchmark-specific heuristic\n32k\n24k train\n4k val\n4k test\nMiniMax-M2.7\ngpu2\n训练耗时\n只有在 32k internal test work well 后\n暂不进入 full-data 训练\n",
        encoding="utf-8",
    )

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_write_gain.json", {"positive_gain": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_belief_gain.json", {"positive_gain": True})
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_canary.json",
        {"provider_exact_match": 11, "local_exact_match": 11},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v26_canary.json",
        {"provider_exact_match": 44, "local_exact_match": 33},
    )
    for name in [
        "latest_stage2_v27_32k_split.json",
        "latest_stage2_v27_32k_manifest.json",
        "latest_stage2_v27_32k_audit.json",
    ]:
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, {"ok": True})

    manifests = {}
    for split in ["train", "val", "test"]:
        path = repo_root / "outputs_v2" / "artifacts" / f"{split}_manifest.json"
        _write_json(
            path,
            {
                "task_counts": {
                    "slot_autoencoding": 10 if split == "train" else 4,
                    "retrieval_alignment": 10 if split == "train" else 4,
                    "lifecycle_prediction": 2 if split == "train" else 1,
                    "composition_to_belief": 10 if split == "train" else 4,
                }
            },
        )
        manifests[split] = path

    train_summary = repo_root / "outputs_v2" / "artifacts" / "train_summary.json"
    _write_json(
        train_summary,
        {
            "run_dir": "outputs_v2/runs/demo_train",
            "metrics_path": "outputs_v2/runs/demo_train/training_metrics.json",
            "checkpoint_dir": "outputs_v2/checkpoints/demo_train",
            "num_examples": 1024,
            "num_steps": 256,
            "optimizer_steps": 256,
            "final_loss": 0.123,
            "device": "cuda",
            "cuda_visible_devices": "2",
            "online_aligned": True,
            "wall_clock_seconds": 12.5,
            "examples_per_second": 81.92,
            "steps_per_second": 20.48,
            "peak_gpu_memory_mb": 2048.0,
            "gpu_name": "NVIDIA RTX A4000",
        },
    )
    eval_payload = {
        "metrics": {"joint_belief_accuracy": 0.5},
        "families": {},
        "modules": {},
        "budget_sweep": [],
        "trained_eval": {
            "metrics": {
                "exact_match": 0.4,
                "token_f1": 0.8,
                "field_f1": 0.75,
            }
        },
    }
    val_eval = repo_root / "outputs_v2" / "artifacts" / "val_eval.json"
    test_eval = repo_root / "outputs_v2" / "artifacts" / "test_eval.json"
    _write_json(val_eval, eval_payload)
    _write_json(
        test_eval,
        {
            **eval_payload,
            "trained_eval": {
                "metrics": {
                    "exact_match": 0.35,
                    "token_f1": 0.72,
                    "field_f1": 0.68,
                }
            },
        },
    )

    published = publish_v27_artifacts(
        root=repo_root,
        train_summary_path=train_summary,
        val_eval_path=val_eval,
        test_eval_path=test_eval,
        train_manifest_path=manifests["train"],
        val_manifest_path=manifests["val"],
        test_manifest_path=manifests["test"],
    )
    assert published["timing"]["cuda_visible_devices"] == "2"
    assert published["internal_test"]["gate_passed"] is True
    assert published["holdout_summary"]["holdout_only"] is True

    payload = compute_v27_longrun(repo_root)
    assert payload["score"] == 23
