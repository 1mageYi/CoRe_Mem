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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_stage2_memory_canary_analysis_and_quality_verifier(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / ".agent-os").mkdir(parents=True)
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)
    (repo_root / "outputs_v2" / "runs" / "demo").mkdir(parents=True)

    current_status = (
        repo_root / "docs" / "current_status.md"
    )
    current_status.write_text(
        "对 live canary 做 failure analysis，并迭代修复 online memory / belief / answer 链路\n",
        encoding="utf-8",
    )
    project_index = repo_root / ".agent-os" / "project-index.md"
    project_index.write_text(
        "`TD-024` `[doing]`: 基于 `PersonaMem 64` live stage-2 canary 做 failure analysis\n",
        encoding="utf-8",
    )

    predictions_path = repo_root / "outputs_v2" / "runs" / "demo" / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "a",
                "benchmark": "personamem",
                "expected_answer": "(a)",
                "memory_answer_local": "(a)",
                "provider_prediction": "(a) correct",
                "provider_status": "completed",
                "question_type": "single_fact",
                "topic": "music",
                "belief_state": {"belief_items": [{"relation": "music_preference"}]},
                "selected_slot_ids": ["slot_1"],
            },
            {
                "sample_id": "b",
                "benchmark": "personamem",
                "expected_answer": "(b)",
                "memory_answer_local": "wrong",
                "provider_prediction": "(b) prefix only",
                "provider_status": "completed",
                "question_type": "single_fact",
                "topic": "music",
                "belief_state": {"belief_items": [{"relation": "food_preference"}]},
                "selected_slot_ids": ["slot_2", "slot_3"],
            },
        ],
    )
    summary_path = repo_root / "outputs_v2" / "evals_benchmark" / "20260414T000000Z_stage2_memory_canary.json"
    _write_json(
        summary_path,
        {
            "benchmark": "personamem",
            "sample_count": 64,
            "live_predictions_completed": 64,
            "provider_configured": True,
            "status": "completed",
            "predictions_path": str(predictions_path),
        },
    )

    analysis = _run(
        "scripts/analyze_stage2_memory_canary_failures.py",
        "--root",
        str(repo_root),
        "--benchmark",
        "personamem",
        "--summary-path",
        str(summary_path),
        "--json",
    )
    assert analysis.returncode == 0, analysis.stderr
    analysis_payload = json.loads(analysis.stdout)
    assert analysis_payload["provider_exact_match"] == 0
    assert analysis_payload["provider_label_prefix_match"] == 2
    assert analysis_payload["local_exact_match"] == 1
    latest_analysis = repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_canary_analysis.json"
    assert latest_analysis.exists()
    latest_layered = repo_root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_layered_analysis.json"
    assert latest_layered.exists()

    verify = _run(
        "scripts/verify_stage2_memory_canary_quality.py",
        "--root",
        str(repo_root),
        "--benchmark",
        "personamem",
        "--summary-path",
        str(summary_path),
    )
    assert verify.returncode == 0, verify.stderr
    verify_payload = json.loads(verify.stdout)
    assert verify_payload["metrics"]["provider_exact_match"] == 0
    assert verify_payload["metrics"]["provider_label_prefix_match"] == 2
    assert verify_payload["metrics"]["local_exact_match"] == 1
    assert verify_payload["checks"]["analysis_artifact_exists"] is True


def test_stage2_memory_canary_analysis_writes_longmemeval_layered_artifact(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "outputs_v2" / "evals_benchmark").mkdir(parents=True)
    (repo_root / "outputs_v2" / "runs" / "demo").mkdir(parents=True)

    predictions_path = repo_root / "outputs_v2" / "runs" / "demo" / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "a",
                "benchmark": "longmemeval_s",
                "expected_answer": "answer",
                "memory_answer_local": "",
                "provider_prediction": "wrong",
                "provider_status": "completed",
                "question_type": "single-session-user",
                "belief_state": {"belief_items": []},
                "selected_slot_ids": [],
            },
            {
                "sample_id": "b",
                "benchmark": "longmemeval_s",
                "expected_answer": "answer",
                "memory_answer_local": "answer",
                "provider_prediction": "still wrong",
                "provider_status": "completed",
                "question_type": "single-session-assistant",
                "belief_state": {"belief_items": [{"relation": "fact"}]},
                "selected_slot_ids": ["slot_1"],
            },
        ],
    )
    summary_path = repo_root / "outputs_v2" / "evals_benchmark" / "20260415T000000Z_stage2_memory_canary.json"
    _write_json(
        summary_path,
        {
            "benchmark": "longmemeval_s",
            "sample_count": 64,
            "live_predictions_completed": 64,
            "provider_configured": True,
            "status": "completed",
            "predictions_path": str(predictions_path),
        },
    )

    analysis = _run(
        "scripts/analyze_stage2_memory_canary_failures.py",
        "--root",
        str(repo_root),
        "--benchmark",
        "longmemeval_s",
        "--summary-path",
        str(summary_path),
        "--json",
    )
    assert analysis.returncode == 0, analysis.stderr
    layered_path = repo_root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_layered_analysis.json"
    assert layered_path.exists()
    layered_payload = json.loads(layered_path.read_text(encoding="utf-8"))
    assert set(layered_payload["layers"]) == {"parser", "retrieval", "belief", "projection", "provider"}
    assert layered_payload["layers"]["parser"]["count"] == 1
    assert layered_payload["layers"]["provider"]["count"] == 1
