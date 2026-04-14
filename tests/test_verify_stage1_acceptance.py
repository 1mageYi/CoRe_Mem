from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.verify_stage1_acceptance import compute_acceptance_status


def test_acceptance_verifier_reports_current_partial_state():
    payload = compute_acceptance_status(REPO_ROOT, run_pytest=False)
    assert payload["criteria"]["AC-001"]["passed"] is True
    assert payload["criteria"]["AC-002"]["passed"] is False
    assert payload["criteria"]["AC-003"]["passed"] is False
    assert payload["criteria"]["AC-004"]["passed"] is True
    assert payload["criteria"]["AC-005"]["passed"] is True
    assert payload["criteria"]["AC-007"]["passed"] is True


def test_acceptance_verifier_score_only_cli():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_stage1_acceptance.py"),
        "--root",
        str(REPO_ROOT),
        "--score-only",
        "--skip-pytest",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 1
    assert int(result.stdout.strip()) >= 1
