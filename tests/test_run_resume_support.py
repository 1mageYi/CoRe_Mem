from pathlib import Path
from types import SimpleNamespace
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.benchmarks.common import PromptRecord
import scripts.run_experiment as run_experiment


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeProvider:
    calls: list[str] = []

    def __init__(self, config) -> None:
        self.config = config

    def chat(self, prompt: str, temperature: float, max_tokens: int) -> FakeResponse:
        self.calls.append(prompt)
        return FakeResponse("predicted")


def test_execute_records_resume_skips_completed_predictions(monkeypatch, tmp_path):
    FakeProvider.calls = []
    run_dir = tmp_path / "outputs" / "runs" / "resume_longmemeval"
    run_dir.mkdir(parents=True)
    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "benchmark": "longmemeval",
                "prediction": "done",
                "expected_answer": "done",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(run_experiment, "OpenAICompatibleProvider", FakeProvider)
    records = [
        PromptRecord(sample_id="s1", benchmark="longmemeval", prompt="p1", expected_answer="a1", metadata={}),
        PromptRecord(sample_id="s2", benchmark="longmemeval", prompt="p2", expected_answer="a2", metadata={}),
    ]
    llm_config = SimpleNamespace(
        api_key_env="ALIYUN_API_KEY",
        base_url="https://example.com/v1",
        model="fake-model",
        provider="aliyun_openai_compatible",
        timeout_seconds=1,
        temperature=0.0,
        max_tokens=8,
    )

    result = run_experiment.execute_records(
        benchmark="longmemeval",
        records=records,
        config_path=REPO_ROOT / "configs" / "defaults.yaml",
        outputs_root=tmp_path / "outputs",
        llm_config=llm_config,
        dry_run=False,
        requested_run_dir=str(run_dir),
        resume=True,
    )

    assert result == 0
    assert FakeProvider.calls == ["p2"]
    lines = predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_prediction_count"] == 1
    assert metadata["completed_predictions"] == 2
