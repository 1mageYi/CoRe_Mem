from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.datasets import build_prepared_payload_from_sources, summarize_stage2_sources
from core_mem.v2.training import PreparedSeq2SeqDataset, TrainingExample


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage2_data_summary_and_prepare_from_config(tmp_path: Path):
    raw_file = tmp_path / "sgd.jsonl"
    _write_jsonl(
        raw_file,
        [
            {
                "dataset": "sgd",
                "sample_id": "sgd-1",
                "tasks": {
                    "slot_autoencoding": {
                        "input_observation": {
                            "obs_id": "obs_1",
                            "source_dataset": "sgd",
                            "source_dialogue_id": "dlg_1",
                            "source_turn_id": "turn_1",
                            "session_id": "sess_1",
                            "speaker": "user",
                            "entity": "user",
                            "relation": "drink_preference",
                            "value": "matcha",
                            "value_type": "preference",
                            "time_scope": "current",
                            "status_hint": "active",
                            "polarity": "positive",
                            "confidence": 0.9,
                            "evidence_text": "I like matcha.",
                            "canonical_gloss": "drink_preference=matcha",
                            "metadata": {}
                        },
                        "target_record": {
                            "entity": "user",
                            "relation": "drink_preference",
                            "value": "matcha",
                            "time_scope": "current",
                            "status": "active"
                        }
                    }
                }
            }
        ],
    )
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "sgd": {"enabled": True, "path": str(raw_file), "format": "normalized_jsonl"},
                    "mquake": {"enabled": True, "path": str(tmp_path / "missing.jsonl"), "format": "normalized_jsonl"},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = summarize_stage2_sources(config_path)
    payload = build_prepared_payload_from_sources(config_path)

    assert summary["available"] == ["sgd"]
    assert summary["missing"] == ["mquake"]
    assert len(payload["slot_autoencoding"]) == 1


def test_stage2_prepare_can_limit_rows_per_dataset(tmp_path: Path):
    raw_file = tmp_path / "sgd.jsonl"
    _write_jsonl(
        raw_file,
        [
            {
                "dataset": "sgd",
                "sample_id": "sgd-1",
                "tasks": {"slot_autoencoding": {"input_observation": {"obs_id": "obs_1"}, "target_record": {"value": "a"}}},
            },
            {
                "dataset": "sgd",
                "sample_id": "sgd-2",
                "tasks": {"slot_autoencoding": {"input_observation": {"obs_id": "obs_2"}, "target_record": {"value": "b"}}},
            },
        ],
    )
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        json.dumps(
            {"datasets": {"sgd": {"enabled": True, "path": str(raw_file), "format": "normalized_jsonl"}}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = build_prepared_payload_from_sources(config_path, max_rows_per_dataset=1)

    assert len(payload["slot_autoencoding"]) == 1
    assert payload["slot_autoencoding"][0]["_meta"]["dataset"] == "sgd"


def test_stage2_prepare_script_can_use_source_config(tmp_path: Path):
    raw_file = tmp_path / "sgd.jsonl"
    _write_jsonl(
        raw_file,
        [
            {
                "dataset": "sgd",
                "sample_id": "sgd-1",
                "tasks": {
                    "slot_autoencoding": {
                        "input_observation": {
                            "obs_id": "obs_1",
                            "source_dataset": "sgd",
                            "source_dialogue_id": "dlg_1",
                            "source_turn_id": "turn_1",
                            "session_id": "sess_1",
                            "speaker": "user",
                            "entity": "user",
                            "relation": "drink_preference",
                            "value": "matcha",
                            "value_type": "preference",
                            "time_scope": "current",
                            "status_hint": "active",
                            "polarity": "positive",
                            "confidence": 0.9,
                            "evidence_text": "I like matcha.",
                            "canonical_gloss": "drink_preference=matcha",
                            "metadata": {}
                        },
                        "target_record": {
                            "entity": "user",
                            "relation": "drink_preference",
                            "value": "matcha",
                            "time_scope": "current",
                            "status": "active"
                        }
                    }
                }
            }
        ],
    )
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        json.dumps(
            {"datasets": {"sgd": {"enabled": True, "path": str(raw_file), "format": "normalized_jsonl"}}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs_v2"
    result = _run(
        "scripts/prepare_stage2_data.py",
        "--output-root",
        str(output_root),
        "--source-config",
        str(config_path),
        "--max-rows-per-dataset",
        "1",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    manifest = json.loads(Path(payload["prepared_manifest"]).read_text(encoding="utf-8"))
    assert manifest["mode"] == "configured_sources"
    assert manifest["task_counts"]["slot_autoencoding"] == 1
    assert manifest["max_rows_per_dataset"] == 1


class _FakeHFTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text: str | None = None,
        *,
        text_target: str | None = None,
        truncation: bool,
        max_length: int,
        padding: str,
        return_tensors: str,
    ) -> dict[str, object]:
        del truncation, padding, return_tensors
        payload = text if text is not None else text_target
        assert isinstance(payload, str)
        token_ids = [index + 1 for index, _ in enumerate(payload[:max_length])]
        token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        attention_mask = [1 if token != self.pad_token_id else 0 for token in token_ids]
        return {
            "input_ids": __import__("torch").tensor([token_ids]),
            "attention_mask": __import__("torch").tensor([attention_mask]),
        }


def test_prepared_seq2seq_dataset_supports_hf_style_tokenizer():
    dataset = PreparedSeq2SeqDataset(
        [
            TrainingExample(
                task_name="slot_autoencoding",
                input_text="hello world",
                target_text='{"value":"matcha"}',
            )
        ],
        _FakeHFTokenizer(),
        max_source_length=8,
        max_target_length=10,
    )

    sample = dataset[0]
    assert sample["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sample["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
    assert sample["labels"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
