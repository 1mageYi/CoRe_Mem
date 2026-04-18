from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v29_longrun import compute_v29_longrun


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
