from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_publish_stage2_v26_artifacts_emits_current_head_aliases(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "outputs_v2" / "artifacts").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)
    (repo_root / "outputs_v2" / "runs" / "long").mkdir(parents=True)
    (repo_root / "outputs_v2" / "runs" / "persona").mkdir(parents=True)

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

    _write_json(repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_train.json", {"commit_hash": "old"})
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_eval.json",
        {
            "commit_hash": "old",
            "trained_eval": {"metrics": {"token_f1": 0.9992, "field_f1": 0.998, "field_accuracy": 0.997, "exact_match": 0.95}},
            "slot_assignment_metrics": {"token_f1": 0.99},
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v25_canary.json",
        {"commit_hash": "old", "status": "completed", "sample_count": 128, "provider_exact_match": 10, "local_exact_match": 10},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_v25_canary.json",
        {"commit_hash": "old", "status": "completed", "sample_count": 128, "provider_exact_match": 38, "local_exact_match": 28},
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v25_analysis.json",
        {
            "commit_hash": "old",
            "layered_breakdown": {
                "parser": {"count": 0},
                "retrieval": {"count": 2},
                "belief": {"count": 0},
                "projection": {"count": 118},
                "provider": {"count": 0},
            },
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_write_gain.json",
        {
            "commit_hash": "old",
            "positive_gain": False,
            "delta_vs_v25_retained": 0,
            "delta_provider_exact_match": 0,
            "delta_local_exact_match": 0,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_retrieve_gain.json",
        {
            "commit_hash": "old",
            "positive_gain": False,
            "delta_vs_v25_retained": 0,
            "layered_breakdown": {"retrieval": {"count": 2}},
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_belief_gain.json",
        {
            "commit_hash": "old",
            "positive_gain": False,
            "delta_vs_v25_retained": 0,
            "v25_projection_bottleneck_count": 118,
        },
    )
    _write_json(
        repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_full_benchmark.json",
        {
            "commit_hash": "old",
            "holdout_only": True,
            "num_benchmarks": 2,
        },
    )

    long_predictions = repo_root / "outputs_v2" / "runs" / "long" / "predictions.jsonl"
    persona_predictions = repo_root / "outputs_v2" / "runs" / "persona" / "predictions.jsonl"
    _write_jsonl(
        long_predictions,
        [
            {
                "sample_id": "q1",
                "benchmark": "longmemeval_s",
                "expected_answer": "Seattle",
                "memory_answer_local": "Seattle",
                "provider_prediction": "Seattle",
                "provider_status": "completed",
                "question_type": "single-session-user",
                "belief_state": {"belief_items": [{"relation": "location"}]},
                "selected_slot_ids": ["slot_1"],
            }
        ],
    )
    _write_jsonl(
        persona_predictions,
        [
            {
                "sample_id": "p1",
                "benchmark": "personamem",
                "expected_answer": "(a)",
                "memory_answer_local": "(a)",
                "provider_prediction": "(a)",
                "provider_status": "completed",
                "question_type": "recall_user_shared_facts",
                "topic": "food",
                "belief_state": {"belief_items": [{"relation": "food_preference"}]},
                "selected_slot_ids": ["slot_2"],
            }
        ],
    )
    long_summary = repo_root / "outputs_v2" / "evals_benchmark" / "long.json"
    persona_summary = repo_root / "outputs_v2" / "evals_benchmark" / "persona.json"
    _write_json(
        long_summary,
        {
            "benchmark": "longmemeval_s",
            "commit_hash": head,
            "status": "completed",
            "sample_count": 128,
            "provider_exact_match": 11,
            "provider_label_prefix_match": 11,
            "local_exact_match": 11,
            "predictions_path": str(long_predictions),
            "memory_mode": "symbolic",
            "slot_assignment_mode": "learned",
        },
    )
    _write_json(
        persona_summary,
        {
            "benchmark": "personamem",
            "commit_hash": head,
            "status": "completed",
            "sample_count": 128,
            "provider_exact_match": 38,
            "provider_label_prefix_match": 38,
            "local_exact_match": 28,
            "predictions_path": str(persona_predictions),
            "memory_mode": "symbolic",
            "slot_assignment_mode": "learned",
        },
    )

    result = _run(
        str(REPO_ROOT / "scripts" / "verify_stage2_v26_longrun.py"),
        "--publish-artifacts",
        "--root",
        str(repo_root),
        "--longmemeval-summary-path",
        str(long_summary),
        "--personamem-summary-path",
        str(persona_summary),
        "--json",
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["head"] == head
    assert payload["v25_eval"]["commit_hash"] == head
    assert payload["v25_longmemeval_canary"]["commit_hash"] == head
    assert payload["v25_personamem_canary"]["commit_hash"] == head

    write_gain = json.loads((repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_write_gain.json").read_text(encoding="utf-8"))
    assert write_gain["positive_gain"] is True
    assert write_gain["delta_provider_exact_match"] == 1
    assert write_gain["delta_local_exact_match"] == 1

    v25_eval = json.loads((repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v25_eval.json").read_text(encoding="utf-8"))
    assert v25_eval["commit_hash"] == head

    full_benchmark = json.loads((repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_full_benchmark.json").read_text(encoding="utf-8"))
    assert full_benchmark["holdout_only"] is True
    assert full_benchmark["num_benchmarks"] == 2
    assert not (repo_root / "outputs_v2" / "artifacts" / "latest_stage2_v26_full_benchmark_gain.json").exists()

    canary_payload = json.loads((repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_v26_canary.json").read_text(encoding="utf-8"))
    assert canary_payload["commit_hash"] == head
