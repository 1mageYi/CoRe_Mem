from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.verify_stage2_v22_completion import compute_v22_completion


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v22_completion_verifier_passes_with_current_head_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
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

    (repo_root / "docs" / "current_status.md").write_text("`TD-032`\n`v2.2`\nfull-data\nLongMemEval-S\n", encoding="utf-8")
    (repo_root / "docs" / "implementation_plan.md").write_text("阶段 Q：V2.2 Full-Data Semantic Latent\nfull-data\nLongMemEval-S\n", encoding="utf-8")
    (repo_root / ".agent-os" / "project-index.md").write_text("Top next action: `TD-032`\nActive workstreams: `WS-018`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-032` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v22_plan.md").write_text("full-data\nLongMemEval-S\nno fallback\nno shortcut\n", encoding="utf-8")
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text("def main():\n    return 'clean runner'\n", encoding="utf-8")
    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text("memory_mode='learned_memory'\nuse_learned_memory=True\n", encoding="utf-8")
    (repo_root / "src" / "core_mem" / "v2" / "training.py").write_text(
        "online_aligned=True\nsemantic_validity_rate=1\nfield_accuracy=1\nfield_f1=1\njson_validity_rate=1\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "semantic_outputs.py").write_text("semantic_validity_rate=1\nfield_f1=1\n", encoding="utf-8")

    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_train.json",
        {
            "commit_hash": head,
            "cuda_visible_devices": "3",
            "num_examples": 800,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_full_local_eval.json",
        {
            "commit_hash": head,
            "trained_eval": {
                "metrics": {"token_f1": 0.86},
                "per_task": {"retrieval_alignment": {"token_f1": 0.91}},
            },
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_semantic_canary.json",
        {"commit_hash": head, "sample_count": 128, "status": "completed"},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_semantic_canary.json",
        {"commit_hash": head, "sample_count": 128, "status": "completed"},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_semantic_analysis.json",
        {"commit_hash": head},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_semantic_online_gain.json",
        {"commit_hash": head, "positive_gain": True},
    )

    payload = compute_v22_completion(repo_root)
    assert payload["score"] == payload["total"] == 19
