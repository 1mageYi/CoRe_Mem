from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.verify_stage2_v25_longrun import compute_v25_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v25_longrun_verifier_passes_when_all_milestones_exist(tmp_path: Path) -> None:
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
        "`TD-036`\n`v2.5`\nLongMemEval-S\nwrite -> retrieve -> belief\ncore / residual\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 U：V2.5 Generalization-First Long Run\nLongMemEval-S\nfull benchmark\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "`TD-036 / WS-022`\n`v2.5 learned core-path long-run`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-036` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v25_plan.md").write_text(
        "冻结边界\n不改 `core / residual`\nwrite\nretrieve\nbelief\n不做任何 `fallback`\n不做任何 `shortcut`\n不把 raw JSON exactness 当成主优化目标\n不把 full benchmark 结果回流成训练 supervision\nholdout evaluation\n",
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

    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v24_train.json",
        {"commit_hash": head},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v24_eval.json",
        {"commit_hash": head, "trained_eval": {"metrics": {"token_f1": 0.95, "field_f1": 0.95}}},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v24_canary.json",
        {"commit_hash": head, "status": "completed", "sample_count": 128},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v24_canary.json",
        {"commit_hash": head, "status": "completed", "sample_count": 128},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v24_online_gain.json",
        {"commit_hash": head, "delta_provider_exact_match": 4, "delta_local_exact_match": 4},
    )

    for name in [
        "latest_stage2_v25_train.json",
        "latest_stage2_v25_eval.json",
        "latest_stage2_v25_write_gain.json",
        "latest_stage2_v25_retrieve_gain.json",
        "latest_stage2_v25_belief_gain.json",
        "latest_longmemeval_stage2_v25_canary.json",
        "latest_personamem_stage2_v25_canary.json",
        "latest_longmemeval_stage2_v25_analysis.json",
        "latest_stage2_v25_full_benchmark.json",
    ]:
        payload = {"commit_hash": head}
        if "canary" in name:
            payload.update({"status": "completed", "sample_count": 128})
        if "analysis" in name:
            payload.update({"error_clusters": {"belief": 3}})
        if "full_benchmark" in name:
            payload.update({"num_benchmarks": 2})
        _write_json(repo_root / "outputs_v2" / "artifacts" / name, payload)

    payload = compute_v25_longrun(repo_root)
    assert payload["score"] == payload["total"] == 24
