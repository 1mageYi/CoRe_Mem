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


def test_stage2_latent_core_quality_verifier(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "把主指标重新锚定到 `stage-2 local intrinsic quality`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "`WS-012` `[done]`: Stage-2 latent-core robustness 目标已机械达成\n"
        "历史主线兼容记录：`TD-026` `[doing]`: 以系统/模型/latent 本体更强、更稳健为锚点\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "verify_stage2_latent_status.py").write_text(
        "print(9)\n",
        encoding="utf-8",
    )

    eval_path = repo_root / "outputs_v2" / "evals_local" / "demo_stage2_local_eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "joint_belief_accuracy": 0.89,
                    "slot_value_f1": 0.95,
                    "support_slot_recall": 0.98,
                    "answer_exact_match": 0.8,
                    "compression_fidelity": 0.88,
                }
            }
        ),
        encoding="utf-8",
    )
    canary_path = repo_root / "outputs_v2" / "evals_benchmark" / "demo_stage2_memory_canary.json"
    predictions_path = repo_root / "outputs_v2" / "runs" / "demo" / "predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "expected_answer": "(a)",
                    "provider_prediction": "(a)",
                    "memory_answer_local": "(a)",
                }
            )
            + "\n"
            for _ in range(64)
        ),
        encoding="utf-8",
    )
    canary_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "sample_count": 64,
                "summary_path": "demo_stage2_memory_canary.json",
                "predictions_path": str(predictions_path),
            }
        ),
        encoding="utf-8",
    )

    result = _run(
        "scripts/verify_stage2_latent_core_quality.py",
        "--root",
        str(repo_root),
        "--eval-path",
        str(eval_path),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score"] == 10
    assert payload["checks"]["joint_belief_accuracy_ge_0_88"] is True


def test_stage2_latent_core_quality_ignores_newer_blocked_probe(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_local").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)
    (repo_root / "outputs_v2" / "runs" / "demo").mkdir(parents=True)
    (repo_root / "scripts").mkdir(parents=True)

    (repo_root / "docs" / "current_status.md").write_text(
        "把主指标重新锚定到 `stage-2 local intrinsic quality`\n",
        encoding="utf-8",
    )
    (repo_root / ".agent-os" / "project-index.md").write_text(
        "`WS-012` `[done]`: Stage-2 latent-core robustness 目标已机械达成\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "verify_stage2_latent_status.py").write_text("print(9)\n", encoding="utf-8")
    (repo_root / "outputs_v2" / "evals_local" / "demo_stage2_local_eval.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "joint_belief_accuracy": 1.0,
                    "slot_value_f1": 1.0,
                    "support_slot_recall": 1.0,
                    "answer_exact_match": 0.8,
                    "compression_fidelity": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    predictions_path = repo_root / "outputs_v2" / "runs" / "demo" / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "expected_answer": "(a)",
                    "provider_prediction": "(a)",
                    "memory_answer_local": "(a)",
                }
            )
            + "\n"
            for _ in range(64)
        ),
        encoding="utf-8",
    )
    (repo_root / "outputs_v2" / "evals_benchmark" / "20260415T000000Z_stage2_memory_canary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "sample_count": 64,
                "predictions_path": str(predictions_path),
            }
        ),
        encoding="utf-8",
    )
    (repo_root / "outputs_v2" / "evals_benchmark" / "20260415T000100Z_stage2_memory_canary.json").write_text(
        json.dumps(
            {
                "status": "blocked_provider_not_configured",
                "sample_count": 1,
                "predictions_path": str(predictions_path),
            }
        ),
        encoding="utf-8",
    )

    result = _run("scripts/verify_stage2_latent_core_quality.py", "--root", str(repo_root))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["checks"]["personamem_canary_quality_ge_9"] is True
