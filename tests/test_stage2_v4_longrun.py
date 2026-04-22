from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v4_longrun import compute_v4_longrun, publish_v4_persona_compare


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_docs(repo_root: Path) -> None:
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "docs" / "current_status.md").write_text("`TD-045`\n`v4`\nPersona-first\n", encoding="utf-8")
    (repo_root / "docs" / "todo.md").write_text("`TD-045`\n`v4`\nPersonaMem 512\n", encoding="utf-8")
    (repo_root / "docs" / "implementation_plan.md").write_text("`TD-045`\n`v4`\nPersona-first\n", encoding="utf-8")
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-045 / WS-031`\n`v4`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-045`\n`v4`\nPersona-first\n", encoding="utf-8")
    (repo_root / "docs" / "v4_plan.md").write_text(
        "Persona-first\nPersonaMem 512\nLongMemEval-S\nNo fallback\nNo shortcut\nNo benchmark leakage\n",
        encoding="utf-8",
    )


def _write_baseline_artifacts(repo_root: Path) -> None:
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    artifact_root.mkdir(parents=True)
    _write_json(artifact_root / "latest_personamem_stage2_v32_full.json", {"local_exact_match": 175, "provider_exact_match": 183})
    _write_json(
        artifact_root / "latest_personamem_stage2_v33_full.json",
        {"sample_count": 512, "local_exact_match": 196, "provider_exact_match": 182, "memory_mode": "learned_memory", "slot_assignment_mode": "learned"},
    )
    _write_json(
        artifact_root / "latest_longmemeval_stage2_v33_full.json",
        {"sample_count": 500, "local_exact_match": 14, "provider_exact_match": 20},
    )
    _write_json(
        artifact_root / "latest_stage2_v33_learned_authoritative_runtime.json",
        {"learned_authoritative": True, "memory_mode": "learned_memory", "slot_assignment_mode": "learned"},
    )
    _write_json(
        artifact_root / "latest_stage2_v33_full_holdout_compare.json",
        {"personamem_gain_confirmed": False, "longmemeval_gain_confirmed": False},
    )


def test_v4_longrun_scores_baseline_docs_and_v33_truth(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_docs(repo_root)
    _write_baseline_artifacts(repo_root)

    payload = compute_v4_longrun(repo_root)

    assert payload["metric"] == "stage2_v4_persona_first_score"
    assert payload["score"] == 14
    assert payload["total"] == 38


def test_v4_longrun_passes_with_persona_first_learned_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    _write_docs(repo_root)
    _write_baseline_artifacts(repo_root)

    for name, payload in {
        "latest_stage2_v4_personamem_gap_audit.json": {
            "sample_count": 512,
            "local_correct_provider_wrong": 81,
            "provider_is_auxiliary": True,
        },
        "latest_stage2_v4_persona_latent_reader_eval.json": {"positive_gain": True},
        "latest_stage2_v4_persona_belief_graph_eval.json": {"positive_gain": True},
        "latest_stage2_v4_persona_option_scorer_eval.json": {
            "positive_gain": True,
            "learned_option_authoritative": True,
        },
        "latest_personamem_stage2_v4_full.json": {
            "sample_count": 512,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
            "local_exact_match": 210,
            "option_scorer_exact_match": 211,
        },
        "latest_stage2_v4_personamem_compare.json": {
            "personamem_learned_gain_confirmed": True,
            "provider_is_auxiliary": True,
        },
        "latest_stage2_v4_longmemeval_guard.json": {"local_exact_match": 14},
        "latest_stage2_v4_ablation_summary.json": {
            "latent_contributes": True,
            "belief_contributes": True,
            "option_scorer_contributes": True,
            "no_fallback_or_shortcut": True,
        },
    }.items():
        _write_json(artifact_root / name, payload)

    payload = compute_v4_longrun(repo_root)

    assert payload["score"] == payload["total"] == 38


def test_v4_publish_persona_compare_marks_provider_auxiliary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "outputs_v2" / "artifacts"
    run_root = repo_root / "outputs_v2" / "runs"
    artifact_root.mkdir(parents=True)
    run_root.mkdir(parents=True)
    _write_json(artifact_root / "latest_personamem_stage2_v33_full.json", {"local_exact_match": 196})
    _write_json(artifact_root / "latest_longmemeval_stage2_v33_full.json", {"local_exact_match": 14})

    persona_summary = run_root / "persona_summary.json"
    gap_audit = run_root / "gap_audit.json"
    long_guard = run_root / "long_guard.json"
    _write_json(
        persona_summary,
        {
            "sample_count": 512,
            "memory_mode": "learned_memory",
            "slot_assignment_mode": "learned",
            "local_exact_match": 205,
            "option_scorer_exact_match": 207,
        },
    )
    _write_json(gap_audit, {"sample_count": 512, "local_correct_provider_wrong": 81})
    _write_json(long_guard, {"sample_count": 500, "local_exact_match": 14})

    payload = publish_v4_persona_compare(
        root=repo_root,
        personamem_summary_path=persona_summary,
        gap_audit_path=gap_audit,
        longmemeval_guard_path=long_guard,
    )

    compare = payload["personamem_compare"]
    assert compare["provider_is_auxiliary"] is True
    assert compare["personamem_learned_gain_confirmed"] is True
    assert (artifact_root / "latest_personamem_stage2_v4_full.json").exists()
    assert (artifact_root / "latest_stage2_v4_personamem_compare.json").exists()
