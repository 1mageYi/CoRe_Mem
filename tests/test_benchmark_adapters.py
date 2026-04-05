from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.benchmarks.longmemeval import LongMemEvalAdapter
from core_mem.benchmarks.personamem import PersonaMemAdapter


def test_personamem_adapter_builds_prompt_records():
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "personamem"
    adapter = PersonaMemAdapter(data_root=fixture_root)
    records = adapter.build_records(
        questions_path=fixture_root / "questions_32k.csv",
        contexts_path=fixture_root / "shared_contexts_32k.jsonl",
        limit=1,
    )
    assert len(records) == 1
    record = records[0]
    assert record.sample_id == "q_1"
    assert "spicy vegetarian dishes" in record.prompt
    assert record.expected_answer == "Try the spicy tofu curry."


def test_longmemeval_adapter_builds_prompt_records():
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "longmemeval"
    adapter = LongMemEvalAdapter(data_root=fixture_root)
    records = adapter.build_records(
        dataset_path=fixture_root / "longmemeval_s_cleaned.json",
        limit=1,
    )
    assert len(records) == 1
    record = records[0]
    assert record.sample_id == "lm_1"
    assert "I switched to matcha recently." in record.prompt
    assert record.expected_answer == "Matcha"
