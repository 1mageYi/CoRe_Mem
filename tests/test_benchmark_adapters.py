from pathlib import Path
import json
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


def test_personamem_adapter_supports_official_single_key_context_format(tmp_path):
    questions_path = tmp_path / "questions_32k.csv"
    contexts_path = tmp_path / "shared_contexts_32k.jsonl"
    questions_path.write_text(
        (
            "persona_id,question_id,question_type,topic,user_question_or_message,correct_answer,all_options,"
            "shared_context_id,end_index_in_shared_context\n"
            "0,q_official,recall,music,What fits best?,(a),"
            "\"['(a) Match the context','(b) Ignore the context']\","
            "ctx_hash,2\n"
        ),
        encoding="utf-8",
    )
    contexts_path.write_text(
        json.dumps(
            {
                "ctx_hash": [
                    {"role": "user", "content": "I compose with software."},
                    {"role": "assistant", "content": "You enjoy music production."},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    adapter = PersonaMemAdapter(data_root=tmp_path)
    records = adapter.build_records(
        questions_path=questions_path,
        contexts_path=contexts_path,
        limit=1,
    )
    assert len(records) == 1
    assert "I compose with software." in records[0].prompt
    assert "Return only the best option label" in records[0].prompt
    assert records[0].expected_answer == "(a)"


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
