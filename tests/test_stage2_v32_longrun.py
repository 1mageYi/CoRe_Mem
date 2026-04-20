from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v32_longrun import compute_v32_longrun, publish_v32_full_holdout_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v32_longrun_verifier_scores_baseline_when_docs_and_retained_truth_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-043`\n`v32`\nlatent\nbelief\nanswer\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-043`\n`v32`\nlatent\nbelief\nanswer\noption-scoring\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-043 / WS-029`\n`v32`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-043` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v32_plan.md").write_text(
        "不改 `core / residual`\nlatent-first\nmodular\ntrainable latent reader\nwrite head\nbelief head\nanswer / option-scoring head\noption scoring\ncandidate scoring\n有限候选答案\nLongMemEval-S = 500\nPersonaMem = 512\nholdout\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v30_task_adapter_compare.json": {"task_specific_positive_gain": True},
        "latest_stage2_v30_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v30_belief_gain.json": {"positive_gain": True},
        "latest_stage2_v30_write_gain.json": {"positive_gain": True},
        "latest_stage2_v30_full_holdout_baseline.json": {
            "holdout_only": True,
            "longmemeval_sample_count": 500,
            "personamem_overlap_sample_count": 512,
            "personamem_overlap_provider_exact_rate": 0.35,
            "personamem_overlap_local_exact_rate": 0.18,
        },
        "latest_longmemeval_stage2_v30_full.json": {"provider_exact_match": 19, "local_exact_match": 14},
        "latest_personamem_stage2_v30_full.json": {"provider_exact_rate": 0.35, "local_exact_rate": 0.18},
        "latest_stage2_v31_latent_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v31_belief_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v31_write_holdout_compare.json": {"positive_gain": True},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v32_longrun(repo_root)
    assert payload["score"] == 19
    assert payload["total"] == 44


def test_v32_longrun_verifier_passes_with_full_modular_latent_belief_answer_and_holdout_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-043`\n`v32`\nlatent\nbelief\nanswer\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-043`\n`v32`\nlatent\nbelief\nanswer\noption-scoring\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-043 / WS-029`\n`v32`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-043` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v32_plan.md").write_text(
        "不改 `core / residual`\nlatent-first\nmodular\ntrainable latent reader\nwrite head\nbelief head\nanswer / option-scoring head\noption scoring\ncandidate scoring\n有限候选答案\nLongMemEval-S = 500\nPersonaMem = 512\nholdout\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v30_task_adapter_compare.json": {"task_specific_positive_gain": True},
        "latest_stage2_v30_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v30_belief_gain.json": {"positive_gain": True},
        "latest_stage2_v30_write_gain.json": {"positive_gain": True},
        "latest_stage2_v30_full_holdout_baseline.json": {
            "holdout_only": True,
            "longmemeval_sample_count": 500,
            "personamem_overlap_sample_count": 512,
            "personamem_overlap_provider_exact_rate": 0.35,
            "personamem_overlap_local_exact_rate": 0.18,
        },
        "latest_longmemeval_stage2_v30_full.json": {"provider_exact_match": 19, "local_exact_match": 14},
        "latest_personamem_stage2_v30_full.json": {"provider_exact_rate": 0.35, "local_exact_rate": 0.18},
        "latest_stage2_v31_latent_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v31_belief_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v31_write_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v32_modular_backbone_train.json": {"positive_gain": True},
        "latest_stage2_v32_write_head_eval.json": {"positive_gain": True},
        "latest_stage2_v32_latent_module_train.json": {"trainable_latent": True, "positive_gain": True},
        "latest_stage2_v32_latent_objective_eval.json": {"positive_gain": True},
        "latest_stage2_v32_latent_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v32_belief_decoder_eval.json": {"positive_gain": True},
        "latest_stage2_v32_belief_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v32_answer_head_eval.json": {"positive_gain": True},
        "latest_stage2_v32_option_scoring_compare.json": {"positive_gain": True},
        "latest_stage2_v32_ablation_summary.json": {
            "latent_is_primary_driver": True,
            "belief_contributes": True,
            "answer_head_contributes": True,
        },
        "latest_stage2_v32_full_holdout_compare.json": {
            "holdout_only": True,
            "longmemeval_gain_confirmed": True,
            "personamem_gain_confirmed": True,
            "personamem_overlap_provider_exact_rate": 0.36,
            "personamem_overlap_local_exact_rate": 0.19,
        },
        "latest_longmemeval_stage2_v32_full.json": {"sample_count": 500, "provider_exact_match": 21, "local_exact_match": 15},
        "latest_personamem_stage2_v32_full.json": {"sample_count": 512, "provider_exact_rate": 0.36, "local_exact_rate": 0.19},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v32_longrun(repo_root)
    assert payload["score"] == payload["total"] == 44


def test_v32_publish_full_holdout_artifacts_writes_aliases_and_compare(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    run_root = repo_root / "outputs_v2" / "runs"
    artifact_root.mkdir(parents=True)
    run_root.mkdir(parents=True)

    for name, payload in {
        "latest_stage2_v30_full_holdout_baseline.json": {
            "holdout_only": True,
            "longmemeval_sample_count": 500,
            "personamem_overlap_provider_exact_rate": 0.35,
            "personamem_overlap_local_exact_rate": 0.18,
        },
        "latest_longmemeval_stage2_v30_full.json": {"provider_exact_match": 19, "local_exact_match": 14},
        "latest_personamem_stage2_v30_full.json": {"provider_exact_rate": 0.35, "local_exact_rate": 0.18},
    }.items():
        _write_json(artifact_root / name, payload)

    long_summary = run_root / "long_summary.json"
    persona_summary = run_root / "persona_summary.json"
    _write_json(
        long_summary,
        {
            "sample_count": 500,
            "provider_exact_match": 22,
            "local_exact_match": 16,
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "symbolic",
            "predictions_path": str(run_root / "long_predictions.jsonl"),
        },
    )
    _write_json(
        persona_summary,
        {
            "sample_count": 512,
            "provider_exact_match": 190,
            "local_exact_match": 100,
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "symbolic",
            "predictions_path": str(run_root / "persona_predictions.jsonl"),
        },
    )

    payload = publish_v32_full_holdout_artifacts(
        root=repo_root,
        longmemeval_summary_path=long_summary,
        personamem_summary_path=persona_summary,
    )

    holdout_compare = payload["full_holdout_compare"]
    assert payload["longmemeval_full"]["provider_exact_rate"] == 22 / 500
    assert payload["personamem_full"]["provider_exact_rate"] == 190 / 512
    assert holdout_compare["longmemeval_gain_confirmed"] is True
    assert holdout_compare["personamem_gain_confirmed"] is True
    assert (artifact_root / "latest_longmemeval_stage2_v32_full.json").exists()
    assert (artifact_root / "latest_personamem_stage2_v32_full.json").exists()
    assert (artifact_root / "latest_stage2_v32_full_holdout_compare.json").exists()
