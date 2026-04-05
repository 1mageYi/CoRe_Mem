from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiment_verify_only_smoke():
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"), "--verify-only"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "16"


def test_run_experiment_personamem_dry_run_smoke():
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "personamem"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiment.py"),
        "personamem",
        "--config",
        str(REPO_ROOT / "configs" / "defaults.yaml"),
        "--questions-csv",
        str(fixture_root / "questions_32k.csv"),
        "--contexts-jsonl",
        str(fixture_root / "shared_contexts_32k.jsonl"),
        "--limit",
        "1",
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip())
    assert payload["benchmark"] == "personamem"
    assert payload["record_count"] == 1
