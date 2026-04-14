from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.memory.core import CoreMemoryState, ResidualSlot
from scripts.verify_stage1_status import CHECKS, compute_status


def test_stage1_verifier_finds_all_expected_scaffold_files():
    score, checks = compute_status(REPO_ROOT)
    for name in CHECKS:
        assert checks[name] is True
    assert checks["summary_table"] is True
    assert checks["resumable_runner_support"] is True
    assert checks["core_mem_named_env"] is True
    assert checks["outputs_dir_structure"] is True
    assert checks["formal_personamem_result"] is False
    assert checks["formal_longmemeval_result"] is False
    assert score == 36


def test_core_memory_state_serializes_residual_slots():
    state = CoreMemoryState(
        core_vector=[0.1, 0.2],
        residual_slots=[ResidualSlot(latent=[0.3], timestamp="2026-04-04", salience=0.8, support=["s1"])],
    )
    payload = state.as_dict()
    assert payload["core_vector"] == [0.1, 0.2]
    assert payload["residual_slots"][0]["support"] == ["s1"]


def test_stage1_verifier_score_reflects_expanded_readiness_checks():
    score, _ = compute_status(REPO_ROOT)
    assert score == 36
