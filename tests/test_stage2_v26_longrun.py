from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.verify_stage2_v26_longrun import compute_v26_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v26_longrun_verifier_passes_when_positive_gains_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "src" / "core_mem" / "v2").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)

    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=repo_root, check=True, capture_output=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (repo_root / "docs" / "current_status.md").write_text(
        "`TD-037`\n`v2.6`\nLongMemEval-S\nwrite -> retrieve -> belief\n真实正增益\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 W：V2.6 Gain-First Long Run\nLongMemEval-S\n正增益\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "`TD-037 / WS-023`\n`v2.6 gain-first long-run`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-037` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v26_plan.md").write_text(
        "冻结边界\n不改 `core / residual`\nwrite\nretrieve\nbelief\n不做任何 `fallback`\n不做任何 `shortcut`\n不使用 benchmark label / answer / protocol 输出作为训练 supervision\nholdout evaluation\n",
        encoding="utf-8",
    )

    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text(
        "core_slots = []\nresidual_slots = []\nquery_encoder = True\nbelief_state = True\nbelief_source = 'learned'\n_ranking_score = True\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "consolidation.py").write_text(
        "core_slots = []\nresidual_slots = []\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "lifecycle.py").write_text(
        "slot_assignment_mode = 'learned'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "training.py").write_text(
        "SLOT_ASSIGNMENT_TASK_NAME = 'lifecycle_prediction'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "encoder.py").write_text("retrieval_key\n", encoding="utf-8")
    (repo_root / "src" / "core_mem" / "v2" / "resampler.py").write_text("compose(\n", encoding="utf-8")
    (repo_root / "src" / "core_mem" / "v2" / "decoder.py").write_text("decode(\n", encoding="utf-8")
    (repo_root / "src" / "core_mem" / "v2" / "semantic_outputs.py").write_text("coerce_task_payload\n", encoding="utf-8")

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_train.json", {"commit_hash": head})
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_eval.json",
        {"commit_hash": head, "trained_eval": {"metrics": {"token_f1": 0.9991, "field_f1": 0.9977}}},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v25_canary.json",
        {"commit_hash": head, "status": "completed", "sample_count": 128, "provider_exact_match": 10, "local_exact_match": 10},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v25_canary.json",
        {"commit_hash": head, "status": "completed", "sample_count": 128, "provider_exact_match": 38, "local_exact_match": 28},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_write_gain.json",
        {"commit_hash": head, "positive_gain": False, "delta_vs_v25_retained": 0},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_retrieve_gain.json",
        {"commit_hash": head, "positive_gain": False, "delta_vs_v25_retained": 0},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_belief_gain.json",
        {"commit_hash": head, "positive_gain": False, "delta_vs_v25_retained": 0},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_full_benchmark.json",
        {"commit_hash": head, "num_benchmarks": 2, "holdout_only": True},
    )

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_train.json", {"commit_hash": head})
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_eval.json",
        {"commit_hash": head, "trained_eval": {"metrics": {"token_f1": 0.9992, "field_f1": 0.998}}},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_write_gain.json",
        {"commit_hash": head, "positive_gain": True, "delta_vs_v25_retained": 1},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_retrieve_gain.json",
        {"commit_hash": head, "positive_gain": False, "delta_vs_v25_retained": 0},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_belief_gain.json",
        {"commit_hash": head, "positive_gain": False, "delta_vs_v25_retained": 0},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_canary.json",
        {"commit_hash": head, "status": "completed", "sample_count": 128, "provider_exact_match": 11, "local_exact_match": 11},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v26_canary.json",
        {"commit_hash": head, "status": "completed", "sample_count": 128, "provider_exact_match": 38, "local_exact_match": 28},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_analysis.json",
        {"commit_hash": head, "error_clusters": {"belief": 3}},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_full_benchmark.json",
        {"commit_hash": head, "num_benchmarks": 2, "holdout_only": True},
    )

    payload = compute_v26_longrun(repo_root)
    assert payload["score"] == payload["total"] == 26
