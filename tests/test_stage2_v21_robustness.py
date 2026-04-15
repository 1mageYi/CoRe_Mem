from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_stage2_v21_robustness_verifier(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "runs").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "第二阶段主线切换：当前 `v2` 已解决“闭环是否成立”的问题，因此现阶段主线应切换为 **`v2.1`**。\n"
        "提升 learned path 的实际贡献\n"
        "stage2_v21_robustness_score\n"
        "LongMemEval-S 不再只是“能跑”，而是指标明显提升\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "implementation_plan.md").write_text(
        "阶段 M：V2.1 Robustness 主线\n"
        "LongMemEval-S 专项 failure analysis\n"
        "`LongMemEval-S` 专项质量提升与 learned path 在线增益来组织\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "Top next action: `TD-028`\nActive workstreams: `WS-014`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "todo.md").write_text("`TD-028` `[doing]`\n", encoding="utf-8")
    (repo_root / "scripts" / "verify_stage2_latent_core_quality.py").write_text("print(10)\n", encoding="utf-8")
    (repo_root / "scripts" / "run_stage2_memory_canary.py").write_text("print('clean runner')\n", encoding="utf-8")

    eval_path = repo_root / "outputs_v2" / "evals_local" / "20260415T000200Z_stage2_local_eval.json"
    _write_json(
        eval_path,
        {
            "trained_eval": {
                "metrics": {
                    "token_f1": 0.12,
                }
            }
        },
    )

    personamem64_predictions = repo_root / "outputs_v2" / "runs" / "personamem64" / "predictions.jsonl"
    _write_jsonl(
        personamem64_predictions,
        [
            {
                "expected_answer": "(a)",
                "provider_prediction": "(a)",
                "memory_answer_local": "(a)",
            }
            for _ in range(64)
        ],
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
            "predictions_path": str(personamem64_predictions),
        },
    )

    personamem128_predictions = repo_root / "outputs_v2" / "runs" / "personamem128" / "predictions.jsonl"
    rows_128 = []
    for idx in range(128):
        provider = "(a)" if idx < 64 else "(b)"
        local = "(a)" if idx < 32 else "(b)"
        rows_128.append(
            {
                "expected_answer": "(a)",
                "provider_prediction": provider,
                "memory_answer_local": local,
            }
        )
    _write_jsonl(personamem128_predictions, rows_128)
    _write_json(
        repo_root / "outputs_v2" / "evals_benchmark" / "20260415T000100Z_stage2_memory_canary.json",
        {
            "benchmark": "personamem",
            "status": "completed",
            "sample_count": 128,
            "live_predictions_completed": 128,
            "provider_configured": True,
            "commit_hash": "ignored_without_git_repo",
            "predictions_path": str(personamem128_predictions),
        },
    )

    longmemeval_predictions = repo_root / "outputs_v2" / "runs" / "longmemeval64" / "predictions.jsonl"
    rows_long = []
    for idx in range(64):
        provider = "answer" if idx < 4 else "wrong"
        local = "answer" if idx < 2 else "wrong"
        rows_long.append(
            {
                "expected_answer": "answer",
                "provider_prediction": provider,
                "memory_answer_local": local,
            }
        )
    _write_jsonl(longmemeval_predictions, rows_long)
    _write_json(
        repo_root / "outputs_v2" / "evals_benchmark" / "20260415T000200Z_stage2_memory_canary.json",
        {
            "benchmark": "longmemeval_s",
            "status": "completed",
            "sample_count": 64,
            "live_predictions_completed": 64,
            "provider_configured": True,
            "commit_hash": "ignored_without_git_repo",
            "predictions_path": str(longmemeval_predictions),
        },
    )

    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_layered_analysis.json",
        {
            "layers": {
                "parser": {},
                "retrieval": {},
                "belief": {},
                "projection": {},
                "provider": {},
            }
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_learned_online_gain.json",
        {
            "positive_gain": True,
            "delta_provider_exact_match": 3,
            "delta_provider_label_prefix_match": 4,
            "delta_local_exact_match": 2,
        },
    )

    result = _run("scripts/verify_stage2_v21_robustness.py", "--root", str(repo_root))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score"] == payload["total"] == 15
    assert payload["checks"]["personamem128_provider_prefix_rate_ge_0_40"] is True
    assert payload["checks"]["longmemeval_local_exact_ge_2"] is True
