from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v30_longrun import compute_v30_longrun, publish_v30_full_holdout_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v30_longrun_verifier_scores_baseline_when_docs_and_v29_retained_artifacts_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-041`\n`v3.0`\n`v30`\nshared backbone\nlatent\nbelief\n589\n500\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-041`\n`v30`\nshared backbone\ntask-specific\ntrainable latent\nbelief\n589\n500\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-041 / WS-027`\n`v30`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-041` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v30_plan.md").write_text(
        "不改 `core / residual`\nshared backbone\ntask-specific adapters\ntrainable latent\nencoder / resampler\ndirect latent\nretrieval ranking\nsupport-slot\nlearned belief decoder\nLongMemEval-S = 500\nPersonaMem = 589\nholdout\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v29_write_gain.json": {"positive_gain": True},
        "latest_stage2_v29_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v29_belief_gain.json": {"positive_gain": True},
        "latest_stage2_v29_holdout_summary.json": {"longmemeval_sample_count": 500, "personamem_sample_count": 512},
        "latest_longmemeval_stage2_v29_canary.json": {"provider_exact_match": 19, "local_exact_match": 14},
        "latest_personamem_stage2_v29_canary.json": {"provider_exact_match": 171, "local_exact_match": 108},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v30_longrun(repo_root)
    assert payload["score"] == 19
    assert payload["total"] == 41


def test_v30_longrun_verifier_passes_with_full_architecture_and_holdout_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-041`\n`v3.0`\n`v30`\nshared backbone\nlatent\nbelief\n589\n500\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-041`\n`v30`\nshared backbone\ntask-specific\ntrainable latent\nbelief\n589\n500\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-041 / WS-027`\n`v30`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-041` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v30_plan.md").write_text(
        "不改 `core / residual`\nshared backbone\ntask-specific adapters\ntrainable latent\nencoder / resampler\ndirect latent\nretrieval ranking\nsupport-slot\nlearned belief decoder\nLongMemEval-S = 500\nPersonaMem = 589\nholdout\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v29_write_gain.json": {"positive_gain": True},
        "latest_stage2_v29_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v29_belief_gain.json": {"positive_gain": True},
        "latest_stage2_v29_holdout_summary.json": {"longmemeval_sample_count": 500, "personamem_sample_count": 512},
        "latest_longmemeval_stage2_v29_canary.json": {"provider_exact_match": 19, "local_exact_match": 14},
        "latest_personamem_stage2_v29_canary.json": {"provider_exact_match": 171, "local_exact_match": 108},
        "latest_stage2_v30_full_holdout_baseline.json": {
            "holdout_only": True,
            "longmemeval_sample_count": 500,
            "personamem_sample_count": 589,
            "longmemeval_nonregression_guard": True,
            "personamem_nonregression_guard": True,
            "v29_personamem_provider_exact_rate": 0.33,
            "v29_personamem_local_exact_rate": 0.21,
        },
        "latest_stage2_v30_shared_backbone_train.json": {"cuda_visible_devices": "2"},
        "latest_stage2_v30_task_adapter_compare.json": {"task_specific_positive_gain": True},
        "latest_stage2_v30_latent_module_train.json": {"trainable_encoder_resampler": True},
        "latest_stage2_v30_latent_objective_eval.json": {"positive_gain": True},
        "latest_stage2_v30_belief_decoder_eval.json": {"positive_gain": True},
        "latest_stage2_v30_write_gain.json": {"positive_gain": True},
        "latest_stage2_v30_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v30_belief_gain.json": {"positive_gain": True},
        "latest_longmemeval_stage2_v30_full.json": {"provider_exact_match": 20, "local_exact_match": 15},
        "latest_personamem_stage2_v30_full.json": {"provider_exact_rate": 0.34, "local_exact_rate": 0.22},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v30_longrun(repo_root)
    assert payload["score"] == payload["total"] == 41


def test_publish_v30_full_holdout_artifacts_emits_aliases_and_guards(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    benchmark_root = repo_root / "outputs_v2" / "evals_benchmark"
    artifact_root.mkdir(parents=True)
    benchmark_root.mkdir(parents=True)

    retained_persona_manifest = benchmark_root / "retained_persona_manifest.json"
    _write_json(retained_persona_manifest, {"sample_ids": ["pm-1", "pm-2"]})
    _write_json(artifact_root / "latest_longmemeval_stage2_v29_canary.json", {"sample_count": 500, "provider_exact_match": 19, "local_exact_match": 14})
    _write_json(
        artifact_root / "latest_personamem_stage2_v29_canary.json",
        {
            "sample_count": 512,
            "provider_exact_match": 171,
            "local_exact_match": 108,
            "canary_manifest": str(retained_persona_manifest),
        },
    )

    long_summary = benchmark_root / "long.json"
    persona_summary = benchmark_root / "persona.json"
    persona_predictions = benchmark_root / "persona_predictions.jsonl"
    persona_predictions.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "pm-1", "expected_answer": "(a)", "provider_prediction": "(a)", "memory_answer_local": "(a)"}),
                json.dumps({"sample_id": "pm-2", "expected_answer": "(b)", "provider_prediction": "(b)", "memory_answer_local": "(b)"}),
                json.dumps({"sample_id": "pm-3", "expected_answer": "(c)", "provider_prediction": "(a)", "memory_answer_local": "(c)"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        long_summary,
        {
            "benchmark": "longmemeval_s",
            "sample_count": 500,
            "provider_exact_match": 20,
            "local_exact_match": 15,
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
        },
    )
    _write_json(
        persona_summary,
        {
            "benchmark": "personamem",
            "sample_count": 589,
            "provider_exact_match": 200,
            "local_exact_match": 130,
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
            "predictions_path": str(persona_predictions),
        },
    )

    payload = publish_v30_full_holdout_artifacts(
        root=repo_root,
        longmemeval_summary_path=long_summary,
        personamem_summary_path=persona_summary,
    )

    assert payload["full_holdout_baseline"]["holdout_only"] is True
    assert payload["full_holdout_baseline"]["longmemeval_nonregression_guard"] is True
    assert payload["full_holdout_baseline"]["personamem_nonregression_guard"] is True
    assert payload["full_holdout_baseline"]["personamem_overlap_sample_count"] == 2
    assert (artifact_root / "latest_stage2_v30_full_holdout_baseline.json").exists()
    assert (artifact_root / "latest_longmemeval_stage2_v30_full.json").exists()
    assert (artifact_root / "latest_personamem_stage2_v30_full.json").exists()
