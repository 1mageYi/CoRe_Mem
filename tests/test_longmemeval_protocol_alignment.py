from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.benchmarks.longmemeval import LongMemEvalAdapter


def test_longmemeval_prompt_requires_exact_answer_phrase():
    adapter = LongMemEvalAdapter(data_root=REPO_ROOT / "tests" / "fixtures" / "longmemeval")
    records = adapter.build_records(
        dataset_path=REPO_ROOT / "tests" / "fixtures" / "longmemeval" / "longmemeval_s_cleaned.json",
        limit=1,
    )
    assert len(records) == 1
    assert "Return only the exact answer phrase" in records[0].prompt
    assert "no leading article" in records[0].prompt
