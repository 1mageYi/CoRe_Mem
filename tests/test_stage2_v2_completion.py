from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_stage2_v2_completion_verifier(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "runs" / "demo_stage2_train_exec").mkdir(parents=True)
    (repo_root / "outputs_v2" / "checkpoints" / "demo_stage2_train_exec").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "完整的 `v2`\n不要做任何偷懒兜底 fallback\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 L：完整 V2 长跑里程碑\n不允许 benchmark-specific heuristic / fallback 成为 retained 收益\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text("`TD-027` `[doing]`\n", encoding="utf-8")
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-027` `[doing]`\n", encoding="utf-8")
    (repo_root / "scripts" / "verify_stage2_latent_status.py").write_text("print(9)\n", encoding="utf-8")
    (repo_root / "scripts" / "verify_stage2_latent_core_quality.py").write_text("print(10)\n", encoding="utf-8")
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text(
        "print('clean runner')\n",
        encoding="utf-8",
    )

    _write_json(
        repo_root / "outputs_v2" / "evals_benchmark" / "20260415T000000Z_stage2_memory_canary.json",
        {
            "benchmark": "personamem",
            "status": "completed",
            "sample_count": 64,
            "live_predictions_completed": 64,
            "provider_configured": True,
            "commit_hash": "ignored_without_git_repo",
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "evals_benchmark" / "20260415T000100Z_stage2_memory_canary.json",
        {
            "benchmark": "longmemeval",
            "status": "completed",
            "sample_count": 64,
            "live_predictions_completed": 64,
            "provider_configured": True,
            "commit_hash": "ignored_without_git_repo",
        },
    )
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_canary_analysis.json", {"ok": True})
    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_canary_analysis.json", {"ok": True})

    run_dir = repo_root / "outputs_v2" / "runs" / "demo_stage2_train_exec"
    _write_json(
        run_dir / "execution_summary.json",
        {
            "checkpoint_dir": "outputs_v2/checkpoints/demo_stage2_train_exec",
        },
    )
    (run_dir / "config_snapshot.yaml").write_text("model:\n  backbone: google/flan-t5-base\n", encoding="utf-8")
    (repo_root / "outputs_v2" / "checkpoints" / "demo_stage2_train_exec" / "adapter_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _write_json(
        repo_root / "outputs_v2" / "evals_local" / "20260415T000200Z_stage2_local_eval.json",
        {
            "trained_eval": {
                "metrics": {
                    "exact_match": 0.1,
                    "token_f1": 0.2,
                }
            }
        },
    )

    result = _run("scripts/verify_stage2_v2_completion.py", "--root", str(repo_root))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score"] == payload["total"] == 14
    assert payload["checks"]["trained_eval_positive"] is True
