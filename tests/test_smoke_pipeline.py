from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiment_verify_only_smoke():
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"), "--verify-only"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "36"


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
    assert payload["prompt_version"] == "personamem_mc_v1"
    assert payload["commit_hash"]
    assert payload["run_timestamp"]

    run_dir = Path(payload["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["config_snapshot_path"].endswith("config_snapshot.yaml")
    assert (run_dir / "config_snapshot.yaml").exists()


def test_prepare_personamem_smoke():
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "personamem"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_data.py"),
        "personamem",
        "--config",
        str(REPO_ROOT / "configs" / "defaults.yaml"),
        "--questions-csv",
        str(fixture_root / "questions_32k.csv"),
        "--contexts-jsonl",
        str(fixture_root / "shared_contexts_32k.jsonl"),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    payload = json.loads(result.stdout.strip())
    assert payload["benchmark"] == "personamem"
    assert payload["ready"] is True
