from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v33_longrun import (
    compute_v33_longrun,
    publish_v33_ablation_summary,
    publish_v33_full_holdout_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v33_longrun_verifier_scores_baseline_when_docs_and_retained_truth_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-044`\n`v33`\nlearned-authoritative\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-044`\n`v33`\nlearned\nauthoritative\nbelief\nanswer\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "todo.md").write_text(
        "`TD-044`\n`v33`\nlearned-authoritative\nLongMemEval-S 500\nPersonaMem 512\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-044 / WS-030`\n`v33`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-044` `[doing]`\n`v33`\nlearned-authoritative\n", encoding="utf-8")
    (repo_root / "docs" / "v33_plan.md").write_text(
        "learned-authoritative\nTemporal-Semantic\nBelief Graph\nLongMemEval-S 500\nPersonaMem 512\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_longmemeval_stage2_v32_full.json": {"provider_exact_match": 21, "local_exact_match": 14},
        "latest_personamem_stage2_v32_full.json": {"provider_exact_rate": 0.357421875, "local_exact_rate": 0.341796875},
        "latest_stage2_v32_full_holdout_compare.json": {"longmemeval_gain_confirmed": True, "personamem_gain_confirmed": True},
        "latest_stage2_v32_ablation_summary.json": {"latent_is_primary_driver": True},
        "latest_stage2_v32_latent_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v32_belief_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v32_option_scoring_compare.json": {"positive_gain": True},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v33_longrun(repo_root)
    assert payload["score"] == 17
    assert payload["total"] == 47


def test_v33_longrun_verifier_passes_with_learned_authoritative_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-044`\n`v33`\nlearned-authoritative\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-044`\n`v33`\nlearned\nauthoritative\nbelief\nanswer\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "todo.md").write_text(
        "`TD-044`\n`v33`\nlearned-authoritative\nLongMemEval-S 500\nPersonaMem 512\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-044 / WS-030`\n`v33`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-044` `[doing]`\n`v33`\nlearned-authoritative\n", encoding="utf-8")
    (repo_root / "docs" / "v33_plan.md").write_text(
        "learned-authoritative\nTemporal-Semantic\nBelief Graph\nLongMemEval-S 500\nPersonaMem 512\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_longmemeval_stage2_v32_full.json": {"provider_exact_match": 21, "local_exact_match": 14},
        "latest_personamem_stage2_v32_full.json": {"provider_exact_rate": 0.357421875, "local_exact_rate": 0.341796875},
        "latest_stage2_v32_full_holdout_compare.json": {"longmemeval_gain_confirmed": True, "personamem_gain_confirmed": True},
        "latest_stage2_v32_ablation_summary.json": {"latent_is_primary_driver": True},
        "latest_stage2_v32_latent_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v32_belief_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v32_option_scoring_compare.json": {"positive_gain": True},
        "latest_stage2_v33_modular_authoritative_train.json": {"positive_gain": True},
        "latest_stage2_v33_learned_write_eval.json": {"positive_gain": True},
        "latest_stage2_v33_latent_reader_train.json": {"trainable_latent": True},
        "latest_stage2_v33_temporal_slot_eval.json": {"positive_gain": True},
        "latest_stage2_v33_latent_objective_eval.json": {"positive_gain": True},
        "latest_stage2_v33_belief_graph_eval.json": {"positive_gain": True},
        "latest_stage2_v33_answer_option_eval.json": {"positive_gain": True},
        "latest_stage2_v33_learned_authoritative_runtime.json": {
            "learned_authoritative": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
        },
        "latest_stage2_v33_ablation_summary.json": {
            "latent_is_primary_driver": True,
            "belief_contributes": True,
            "answer_head_contributes": True,
            "write_contributes": True,
        },
        "latest_stage2_v33_full_holdout_compare.json": {
            "learned_authoritative": True,
            "longmemeval_gain_confirmed": True,
            "personamem_gain_confirmed": True,
        },
        "latest_longmemeval_stage2_v33_full.json": {"provider_exact_match": 22, "local_exact_match": 14},
        "latest_personamem_stage2_v33_full.json": {"provider_exact_rate": 0.36, "local_exact_rate": 0.35},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v33_longrun(repo_root)
    assert payload["score"] == payload["total"] == 47


def test_v33_publish_full_holdout_artifacts_marks_learned_authoritative(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    run_root = repo_root / "outputs_v2" / "runs"
    artifact_root.mkdir(parents=True)
    run_root.mkdir(parents=True)

    _write_json(artifact_root / "latest_longmemeval_stage2_v32_full.json", {"provider_exact_match": 21, "local_exact_match": 14})
    _write_json(artifact_root / "latest_personamem_stage2_v32_full.json", {"provider_exact_rate": 0.357421875, "local_exact_rate": 0.341796875})

    long_summary = run_root / "long_summary.json"
    persona_summary = run_root / "persona_summary.json"
    _write_json(
        long_summary,
        {
            "sample_count": 500,
            "provider_exact_match": 22,
            "local_exact_match": 14,
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
            "predictions_path": str(run_root / "long_predictions.jsonl"),
        },
    )
    _write_json(
        persona_summary,
        {
            "sample_count": 512,
            "provider_exact_match": 190,
            "local_exact_match": 180,
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
            "predictions_path": str(run_root / "persona_predictions.jsonl"),
        },
    )

    payload = publish_v33_full_holdout_artifacts(
        root=repo_root,
        longmemeval_summary_path=long_summary,
        personamem_summary_path=persona_summary,
    )
    assert payload["learned_authoritative_runtime"]["learned_authoritative"] is True
    assert payload["full_holdout_compare"]["longmemeval_gain_confirmed"] is True
    assert payload["full_holdout_compare"]["personamem_gain_confirmed"] is True
    assert (artifact_root / "latest_stage2_v33_learned_authoritative_runtime.json").exists()
    assert (artifact_root / "latest_stage2_v33_full_holdout_compare.json").exists()


def test_v33_publish_ablation_summary_requires_all_contributors(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    artifact_root.mkdir(parents=True)

    for name, payload in {
        "latest_stage2_v33_learned_write_eval.json": {"positive_gain": True},
        "latest_stage2_v33_latent_reader_train.json": {"trainable_latent": True},
        "latest_stage2_v33_temporal_slot_eval.json": {"positive_gain": True},
        "latest_stage2_v33_latent_objective_eval.json": {"positive_gain": True},
        "latest_stage2_v33_belief_graph_eval.json": {"positive_gain": True},
        "latest_stage2_v33_answer_option_eval.json": {"positive_gain": True},
        "latest_stage2_v33_learned_authoritative_runtime.json": {
            "learned_authoritative": True,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
        },
    }.items():
        _write_json(artifact_root / name, payload)

    payload = publish_v33_ablation_summary(root=repo_root)
    assert payload["latent_is_primary_driver"] is True
    assert payload["belief_contributes"] is True
    assert payload["answer_head_contributes"] is True
    assert payload["write_contributes"] is True
