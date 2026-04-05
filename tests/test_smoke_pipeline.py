from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiment_verify_only_smoke():
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"), "--verify-only"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "10"

