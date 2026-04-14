from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.benchmarks.personamem import PersonaMemAdapter


def test_personamem_official_variant_uses_label_answer_protocol():
    adapter = PersonaMemAdapter(data_root=REPO_ROOT / "data" / "personamem")
    records = adapter.build_records(limit=1)
    assert len(records) == 1
    assert "Return only the best option label" in records[0].prompt
    assert records[0].expected_answer.startswith("(")
