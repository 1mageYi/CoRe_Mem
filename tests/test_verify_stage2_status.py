from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.verify_stage2_status import CHECKS, compute_status


def test_stage2_verifier_finds_stage2_scaffold_baseline():
    score, checks = compute_status(REPO_ROOT)
    for name in CHECKS:
        assert checks[name] is True
    assert checks["stage2_docs_ready"] is True
    assert checks["stage2_schema_ready"] is True
    assert score == 50
