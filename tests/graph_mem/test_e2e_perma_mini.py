from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_mem.perma_eval import PermaEvalConfig, run_perma_eval


@pytest.mark.skipif(
    not os.getenv("LLM_API_KEY"),
    reason="LLM_API_KEY is required for PERMA e2e mini test.",
)
def test_perma_mini_e2e() -> None:
    """
    Mini e2e regression guard:
    - runs true PERMA samples
    - compares graph-full vs semantic-only
    - asserts output shape and non-empty execution
    """
    cfg = PermaEvalConfig(user_id="user108", variant="c", limit=2)
    summary = run_perma_eval(cfg, root_dir=ROOT, out_dir=None, show_progress=False)
    assert summary["n_samples"] == 2
    assert "acc_graph_full" in summary
    assert "acc_semantic_only" in summary
    assert "acc_delta_graph_minus_semantic" in summary
