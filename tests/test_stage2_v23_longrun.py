from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.verify_stage2_v23_longrun import compute_v23_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v23_longrun_verifier_passes_when_all_milestones_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src" / "core_mem" / "v2").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)

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
        "`TD-034`\n`v2.3`\nLongMemEval-S\nlearned slot assignment\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 S：V2.3 Long-Run Stronger Learned Slot Assignment\nLongMemEval-S\nlearned slot assignment\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "Top next action: `TD-034`\nActive workstreams: `WS-020`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-034` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v23_plan.md").write_text("LongMemEval-S\nlearned slot assignment\n", encoding="utf-8")
    (repo_root / "docs" / "v23_longrun_plan.md").write_text(
        "多卡\nLongMemEval-S\nlearned slot assignment\nfallback\n",
        encoding="utf-8",
    )

    (repo_root / "src" / "core_mem" / "v2" / "lifecycle.py").write_text(
        "slot_assignment = True\nslot_assignment_mode = 'learned'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text(
        "learned_slot_assignment = True\nslot_assignment_mode = 'learned'\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text(
        "--slot-assignment-mode\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "training.py").write_text(
        "slot_assignment\ncandidate-slot\naction classification\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "train_stage2.py").write_text("slot_assignment\n", encoding="utf-8")

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_train.json", {"commit_hash": "old"})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_local_eval.json", {"commit_hash": "old"})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_semantic_analysis.json", {"commit_hash": "old"})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_online_gain.json", {"commit_hash": "old", "positive_gain": True})
    _write_json(repo_root / "outputs_v2" / "evals_benchmark" / "20260416T021743Z_stage2_memory_canary.json", {"sample_count": 128})
    _write_json(repo_root / "outputs_v2" / "evals_benchmark" / "20260416T024146Z_stage2_memory_canary.json", {"sample_count": 128})

    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_train.json",
        {"commit_hash": head, "num_examples": 1574, "cuda_visible_devices": "3"},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_eval.json",
        {"commit_hash": head, "trained_eval": {"metrics": {"token_f1": 0.91}}, "slot_assignment_accuracy": 0.7},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_gain.json",
        {
            "commit_hash": head,
            "positive_gain": True,
            "delta_provider_exact_match": 2,
            "delta_local_exact_match": 2,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_slot_assignment_canary.json",
        {
            "commit_hash": head,
            "status": "completed",
            "sample_count": 128,
            "provider_exact_match": 8,
            "local_exact_match": 8,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_slot_assignment_canary.json",
        {
            "commit_hash": head,
            "status": "completed",
            "sample_count": 128,
            "provider_exact_match": 35,
            "local_exact_match": 12,
        },
    )

    payload = compute_v23_longrun(repo_root)
    assert payload["score"] == payload["total"] == 22
