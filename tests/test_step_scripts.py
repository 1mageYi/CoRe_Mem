from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_and_make_tables_smoke(tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "demo_personamem"
    run_dir.mkdir(parents=True)
    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps({"prediction": "A", "expected_answer": "A"}),
                json.dumps({"prediction": "B", "expected_answer": "C"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_run.py"),
        "--run-dir",
        str(run_dir),
    ]
    eval_result = subprocess.run(eval_cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert eval_result.returncode == 0
    eval_payload = json.loads(eval_result.stdout.strip())
    assert eval_payload["correct"] == 1
    assert eval_payload["total"] == 2

    table_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "make_tables.py"),
        "--outputs-root",
        str(tmp_path / "outputs"),
    ]
    table_result = subprocess.run(table_cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert table_result.returncode == 0
    table_payload = json.loads(table_result.stdout.strip())
    assert table_payload["rows"] == 1

    summary_rows = json.loads((tmp_path / "outputs" / "tables" / "stage1_summary.json").read_text(encoding="utf-8"))
    assert summary_rows[0]["accuracy"] == 0.5


def test_stage1_preflight_reports_missing_external_dependencies(monkeypatch):
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "stage1_preflight.py"),
        "--config",
        str(REPO_ROOT / "configs" / "defaults.yaml"),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 1
    payload = json.loads(result.stdout.strip())
    assert payload["ready"] is False
    assert "llm_api_key" in payload["blockers"]
