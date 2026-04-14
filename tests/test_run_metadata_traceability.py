from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_personamem_dry_run_persists_traceability_artifacts():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiment.py"),
        "personamem",
        "--config",
        str(REPO_ROOT / "configs" / "defaults.yaml"),
        "--questions-csv",
        str(REPO_ROOT / "tests" / "fixtures" / "personamem" / "questions_32k.csv"),
        "--contexts-jsonl",
        str(REPO_ROOT / "tests" / "fixtures" / "personamem" / "shared_contexts_32k.jsonl"),
        "--limit",
        "1",
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0

    payload = json.loads(result.stdout.strip())
    run_dir = Path(payload["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["prompt_version"] == "personamem_mc_v1"
    assert metadata["commit_hash"]
    assert metadata["run_timestamp"]
    assert (run_dir / "config_snapshot.yaml").exists()
