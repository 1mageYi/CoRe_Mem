from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.verify_stage2_v24_longrun import compute_v24_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v24_longrun_verifier_passes_when_all_milestones_exist(tmp_path: Path) -> None:
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
        "`TD-035`\n`v2.4`\nLongMemEval-S\nlearned slot assignment\nstronger latent\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 T：V2.4 LongMemEval-S Quality + Learned Slot Assignment + Stronger Latent\nLongMemEval-S\nlearned slot assignment\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "Top next action: `TD-035`\nActive workstreams: `WS-021`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-035` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v24_plan.md").write_text(
        "LongMemEval-S\nlearned slot assignment\nstronger latent\nfallback\n完整 public-data\n",
        encoding="utf-8",
    )

    (repo_root / "src" / "core_mem" / "v2" / "lifecycle.py").write_text(
        "slot_assignment = True\nslot_assignment_mode = 'learned'\nLifecycleDecision = object\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text(
        "composed_memory = True\nbelief_source = 'learned_memory'\nslot_assignment_mode = 'learned'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "resampler.py").write_text(
        "cross-attention\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "decoder.py").write_text(
        "belief_source\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "semantic_outputs.py").write_text(
        "semantic_outputs\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "training.py").write_text(
        "slot_assignment\nLifecycleDecision\n",
        encoding="utf-8",
    )

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_train.json", {"commit_hash": "old"})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_eval.json", {"commit_hash": "old"})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_slot_assignment_gain.json", {"commit_hash": "old", "positive_gain": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_train.json", {"commit_hash": "old"})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_local_eval.json", {"commit_hash": "old"})
    _write_json(repo_root / "outputs_v2" / "evals_benchmark" / "20260416T095034Z_stage2_memory_canary.json", {"sample_count": 128})
    _write_json(repo_root / "outputs_v2" / "evals_benchmark" / "20260416T095030Z_stage2_memory_canary.json", {"sample_count": 128})

    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v24_train.json",
        {"commit_hash": head, "num_examples": 1574, "online_aligned": True},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v24_eval.json",
        {
            "commit_hash": head,
            "trained_eval": {"metrics": {"token_f1": 0.94, "field_f1": 0.94}},
            "slot_assignment_metrics": {"token_f1": 0.12},
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v24_canary.json",
        {
            "commit_hash": head,
            "status": "completed",
            "sample_count": 128,
            "provider_exact_match": 12,
            "local_exact_match": 11,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v24_canary.json",
        {
            "commit_hash": head,
            "status": "completed",
            "sample_count": 128,
            "provider_exact_match": 37,
            "local_exact_match": 15,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v24_analysis.json",
        {"commit_hash": head, "error_clusters": {"projection": 4, "slot_assignment": 3}},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v24_online_gain.json",
        {
            "commit_hash": head,
            "positive_gain": True,
            "delta_provider_exact_match": 4,
            "delta_local_exact_match": 5,
        },
    )

    payload = compute_v24_longrun(repo_root)
    assert payload["score"] == payload["total"] == 24
