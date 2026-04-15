from __future__ import annotations

import json
from pathlib import Path
import subprocess

from scripts.verify_stage2_v21_longrun import compute_v21_longrun


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_v21_longrun_verifier_passes_when_all_milestones_exist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src" / "core_mem" / "v2").mkdir(parents=True)
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)

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
        "\n".join(
            [
                "`TD-030`",
                "stage2_v21_longrun_score",
                "不做任何兜底/fallback/benchmark-specific shortcut",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "\n".join(
            [
                "阶段 O：V2.1 Learned-Model Long Run",
                "更加注重 learned model",
                "不做任何兜底/fallback/benchmark-specific shortcut",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "\n".join(
            [
                "Top next action: `TD-030`",
                "Active workstreams: `WS-016`",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-030` `[doing]`\n", encoding="utf-8")
    (repo_root / "docs" / "v21_longrun_plan.md").write_text(
        "\n".join(
            [
                "阶段 A：把 learned path 从“存在”推进到“可测”",
                "阶段 B：让训练真正服务在线链路",
                "阶段 C：专项提升 LongMemEval-S",
                "阶段 D：扩大样本，验证 robustness",
                "阶段 E：收口成 v2.1",
                "不做任何兜底/fallback/benchmark-specific shortcut",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text(
        "def main():\n    return 'clean runner'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "system.py").write_text(
        "\n".join(
            [
                "memory_mode = 'learned_memory'",
                "use_learned_memory = True",
                "learned_memory_checkpoint_dir = 'ckpt'",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "src" / "core_mem" / "v2" / "training.py").write_text(
        "\n".join(
            [
                "online_aligned = True",
                "belief_selection = True",
                "retrieval = True",
                "composition_to_belief = True",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "configs" / "stage2_train.yaml").write_text("online_aligned: true\n", encoding="utf-8")
    (repo_root / "configs" / "stage2_train_tiny.yaml").write_text("online_aligned: true\n", encoding="utf-8")

    _write_json(
        repo_root / "outputs_v2" / "evals_local" / "20260415T000000Z_stage2_local_eval.json",
        {"trained_eval": {"metrics": {"token_f1": 0.25}}},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_learned_online_gain.json",
        {"positive_gain": True, "delta_provider_exact_match": 1},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_layered_analysis.json",
        {"layers": {"parser": {}, "retrieval": {}, "belief": {}, "projection": {}, "provider": {}}},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_learned_canary.json",
        {
            "benchmark": "personamem",
            "sample_count": 128,
            "live_predictions_completed": 128,
            "status": "completed",
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "use_learned_memory": True,
            "commit_hash": head,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_learned_canary.json",
        {
            "benchmark": "longmemeval_s",
            "sample_count": 64,
            "live_predictions_completed": 64,
            "status": "completed",
            "provider_configured": True,
            "memory_mode": "learned_memory",
            "use_learned_memory": True,
            "commit_hash": head,
        },
    )

    payload = compute_v21_longrun(repo_root)
    assert payload["score"] == payload["total"] == 16
    assert payload["best_trained_eval_token_f1"] == 0.25
