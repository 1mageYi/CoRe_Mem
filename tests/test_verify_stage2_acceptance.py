from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.verify_stage2_acceptance import compute_acceptance_status


def test_stage2_acceptance_reports_doc_and_schema_baseline():
    payload = compute_acceptance_status(REPO_ROOT, run_pytest=False)
    assert payload["criteria"]["AC-008"]["passed"] is True
    assert payload["criteria"]["AC-009"]["passed"] is True
    assert payload["criteria"]["AC-010"]["passed"] is True
    assert payload["criteria"]["AC-011"]["passed"] is True
    assert payload["criteria"]["AC-012"]["passed"] is True
    assert payload["criteria"]["AC-013"]["passed"] is True


def test_stage2_acceptance_score_only_cli():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_stage2_acceptance.py"),
        "--root",
        str(REPO_ROOT),
        "--score-only",
        "--skip-pytest",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 1
    assert int(result.stdout.strip()) == 6
