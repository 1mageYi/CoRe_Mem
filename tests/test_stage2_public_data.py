from pathlib import Path
import io
import json
import tarfile
import zipfile

from core_mem.v2.public_data import (
    normalize_mquake_dataset,
    normalize_multiwoz24_dataset,
    normalize_personachat_dataset,
    normalize_sgd_dataset,
)
from core_mem.v2.training import configure_hf_cache


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_normalize_sgd_dataset_emits_real_task_rows(tmp_path: Path):
    raw_root = tmp_path / "sgd_raw" / "train"
    raw_root.mkdir(parents=True)
    dialogue_path = raw_root / "dialogues_001.json"
    dialogue_path.write_text(
        json.dumps(
            [
                {
                    "dialogue_id": "dlg_1",
                    "turns": [
                        {
                            "speaker": "USER",
                            "utterance": "I want a cheap hotel in the north.",
                            "frames": [
                                {
                                    "service": "Hotels_1",
                                    "state": {"slot_values": {"pricerange": ["cheap"], "area": ["north"]}},
                                }
                            ],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "sgd" / "normalized.jsonl"

    result = normalize_sgd_dataset(tmp_path / "sgd_raw", output_path)

    rows = _read_jsonl(output_path)
    assert result["rows"] == 2
    assert rows[0]["dataset"] == "sgd"
    assert "slot_autoencoding" in rows[0]["tasks"]
    assert "composition_to_belief" in rows[0]["tasks"]


def test_normalize_multiwoz24_dataset_reads_zip_bundle(tmp_path: Path):
    zip_path = tmp_path / "MULTIWOZ2.4.zip"
    data_payload = {
        "MUL0001.json": {
            "goal": {},
            "log": [
                {"text": "I need a train to cambridge.", "metadata": {}},
                {
                    "text": "Sure.",
                    "metadata": {
                        "train": {
                            "book": {"booked": [], "people": "2"},
                            "semi": {
                                "leaveAt": "",
                                "destination": "cambridge",
                                "day": "",
                                "arriveBy": "",
                                "departure": "ely",
                            },
                        }
                    },
                },
            ],
        }
    }
    with zipfile.ZipFile(zip_path, "w") as bundle:
        bundle.writestr("MULTIWOZ2.4/data.json", json.dumps(data_payload))
        bundle.writestr("MULTIWOZ2.4/valListFile.json", "")
        bundle.writestr("MULTIWOZ2.4/testListFile.json", "")
    output_path = tmp_path / "multiwoz24" / "normalized.jsonl"

    result = normalize_multiwoz24_dataset(zip_path, output_path)

    rows = _read_jsonl(output_path)
    assert result["rows"] == 3
    assert rows[0]["dataset"] == "multiwoz24"
    assert rows[0]["tasks"]["retrieval_alignment"]["gold_support_slot_ids"]


def test_normalize_personachat_dataset_reads_tarball(tmp_path: Path):
    tar_path = tmp_path / "convai2_fix_723.tgz"
    sample = "\n".join(
        [
            "1 your persona: i love pizza.",
            "2 your persona: i am a nurse.",
            "3 hello there.\thi!",
            "1 your persona: i live in boston.",
            "2 your persona: i enjoy jazz music.",
            "3 hello.\thello!",
        ]
    )
    with tarfile.open(tar_path, "w:gz") as bundle:
        payload = sample.encode("utf-8")
        info = tarfile.TarInfo(name="train_self_revised.txt")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    output_path = tmp_path / "personachat" / "normalized.jsonl"

    result = normalize_personachat_dataset(tar_path, output_path)

    rows = _read_jsonl(output_path)
    assert result["rows"] >= 2
    assert rows[0]["dataset"] == "personachat"
    assert rows[0]["tasks"]["slot_autoencoding"]["input_observation"]["source_dataset"] == "personachat"


def test_normalize_mquake_dataset_uses_counterfactual_answer(tmp_path: Path):
    raw_root = tmp_path / "MQuAKE" / "datasets"
    raw_root.mkdir(parents=True)
    (raw_root / "MQuAKE-CF-3k-v2.json").write_text(
        json.dumps(
            [
                {
                    "case_id": 1,
                    "questions": ["Who now leads the edited country?"],
                    "answer": "Old Answer",
                    "new_answer": "New Answer",
                }
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mquake" / "normalized.jsonl"

    result = normalize_mquake_dataset(tmp_path / "MQuAKE", output_path)

    rows = _read_jsonl(output_path)
    assert result["rows"] == 1
    assert rows[0]["tasks"]["lifecycle_prediction"]["target_action"] == "overwrite"
    assert rows[0]["tasks"]["composition_to_belief"]["target_belief_json"]["belief_items"][0]["value"] == "New Answer"


def test_configure_hf_cache_uses_repo_local_paths(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    config = {"runtime": {"hf_cache": {"root": str(tmp_path / "hf_cache")}}}

    env_updates = configure_hf_cache(config)

    assert Path(env_updates["HF_HOME"]).exists()
    assert Path(env_updates["HF_HUB_CACHE"]).exists()
    assert Path(env_updates["HF_DATASETS_CACHE"]).exists()
