from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v31_longrun import compute_v31_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v31_longrun_verifier_scores_baseline_when_docs_and_v30_retained_artifacts_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-042`\n`v31`\nlatent\nbelief\nwrite\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-042`\n`v31`\nlatent\nbelief\nwrite\nablation\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-042 / WS-028`\n`v31`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-042` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v31_plan.md").write_text(
        "不改 `core / residual`\n主线 A：Latent\n主线 B：Belief\n主线 C：Write\nAblation\n主要增益来源\nLongMemEval-S 500\nPersonaMem 512\nholdout\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v30_task_adapter_compare.json": {"task_specific_positive_gain": True},
        "latest_stage2_v30_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v30_belief_gain.json": {"positive_gain": True},
        "latest_stage2_v30_write_gain.json": {"positive_gain": True},
        "latest_stage2_v30_full_holdout_baseline.json": {"holdout_only": True, "longmemeval_sample_count": 500, "personamem_sample_count": 589, "personamem_overlap_sample_count": 512},
        "latest_longmemeval_stage2_v30_full.json": {"provider_exact_match": 19, "local_exact_match": 14},
        "latest_personamem_stage2_v30_full.json": {"provider_exact_rate": 0.35, "local_exact_rate": 0.18},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v31_longrun(repo_root)
    assert payload["score"] == 16
    assert payload["total"] == 32


def test_v31_longrun_verifier_passes_with_full_latent_belief_write_and_holdout_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-042`\n`v31`\nlatent\nbelief\nwrite\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "`TD-042`\n`v31`\nlatent\nbelief\nwrite\nablation\n500\n512\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-042 / WS-028`\n`v31`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-042` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v31_plan.md").write_text(
        "不改 `core / residual`\n主线 A：Latent\n主线 B：Belief\n主线 C：Write\nAblation\n主要增益来源\nLongMemEval-S 500\nPersonaMem 512\nholdout\nfallback\nshortcut\nbenchmark leakage\n",
        encoding="utf-8",
    )

    for name, payload in {
        "latest_stage2_v30_task_adapter_compare.json": {"task_specific_positive_gain": True},
        "latest_stage2_v30_latent_gain.json": {"positive_gain": True},
        "latest_stage2_v30_belief_gain.json": {"positive_gain": True},
        "latest_stage2_v30_write_gain.json": {"positive_gain": True},
        "latest_stage2_v30_full_holdout_baseline.json": {"holdout_only": True, "longmemeval_sample_count": 500, "personamem_sample_count": 589, "personamem_overlap_sample_count": 512, "personamem_overlap_provider_exact_rate": 0.35, "personamem_overlap_local_exact_rate": 0.18},
        "latest_longmemeval_stage2_v30_full.json": {"provider_exact_match": 19, "local_exact_match": 14},
        "latest_personamem_stage2_v30_full.json": {"provider_exact_rate": 0.35, "local_exact_rate": 0.18},
        "latest_stage2_v31_latent_mainline_train.json": {"trainable_latent": True},
        "latest_stage2_v31_latent_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v31_belief_mainline_eval.json": {"positive_gain": True},
        "latest_stage2_v31_belief_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v31_write_mainline_eval.json": {"positive_gain": True},
        "latest_stage2_v31_write_holdout_compare.json": {"positive_gain": True},
        "latest_stage2_v31_ablation_summary.json": {"latent_is_primary_driver": True, "belief_contributes": True},
        "latest_stage2_v31_full_holdout_compare.json": {
            "holdout_only": True,
            "longmemeval_gain_confirmed": True,
            "personamem_nonregression_guard": True,
        },
        "latest_longmemeval_stage2_v31_full.json": {"sample_count": 500, "provider_exact_match": 20, "local_exact_match": 15},
        "latest_personamem_stage2_v31_full.json": {"sample_count": 512, "provider_exact_rate": 0.36, "local_exact_rate": 0.19},
    }.items():
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v31_longrun(repo_root)
    assert payload["score"] == payload["total"] == 32
