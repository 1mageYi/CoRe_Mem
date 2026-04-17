from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_stage2_data import publish_v27_teacher_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeProvider:
    def is_configured(self) -> bool:
        return True

    def chat(self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int | None = None):
        examples = json.loads(prompt.split("examples=", 1)[1])
        if "teacher_task: observation" in prompt:
            labels = [
                {
                    "sample_id": example["sample_id"],
                    "relation": example["candidate_observation"]["relation"],
                    "value": example["candidate_observation"]["value"],
                    "time_scope": example["candidate_observation"]["time_scope"],
                    "status_hint": example["candidate_observation"]["status_hint"],
                    "polarity": example["candidate_observation"]["polarity"],
                }
                for example in examples
            ]
            return _FakeResponse(json.dumps({"labels": labels}, ensure_ascii=False))
        if "teacher_task: slot_assignment" in prompt:
            labels = []
            for example in examples:
                slot_id = example["memory_context"][0]["slot_id"] if example["memory_context"] else None
                labels.append(
                    {
                        "sample_id": example["sample_id"],
                        "target_action": "overwrite" if slot_id else "new",
                        "matched_slot_id": slot_id,
                        "promote": False,
                        "stale_old": bool(slot_id),
                    }
                )
            return _FakeResponse(json.dumps({"labels": labels}, ensure_ascii=False))
        labels = []
        for example in examples:
            first_slot = example["memory_slots"][0]
            relation, value = first_slot["canonical_gloss"].split("=", 1)
            labels.append(
                {
                    "sample_id": example["sample_id"],
                    "belief_items": [
                        {
                            "relation": relation,
                            "value": value,
                            "status": "active",
                            "time_scope": "current",
                            "support_slot_ids": [first_slot["slot_id"]],
                        }
                    ],
                }
            )
        return _FakeResponse(json.dumps({"labels": labels}, ensure_ascii=False))


def _build_split_manifest(root: Path, split: str) -> Path:
    split_root = root / "artifacts" / "stage2_v27_32k" / split
    observation = {
        "input_observation": {
            "obs_id": f"obs_{split}",
            "source_dataset": "sgd",
            "source_dialogue_id": f"dlg_{split}",
            "source_turn_id": f"turn_{split}",
            "session_id": f"sess_{split}",
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
            "metadata": {},
        },
        "target_record": {
            "entity": "user",
            "relation": "drink_preference",
            "value": "matcha",
            "time_scope": "current",
            "status": "active",
        },
        "_meta": {"dataset": "sgd", "sample_id": f"slot-{split}", "task_name": "slot_autoencoding"},
    }
    lifecycle = {
        "memory_context": [
            {
                "slot_id": f"slot_old_{split}",
                "bank": "core",
                "entity": "user",
                "relation": "drink_preference",
                "canonical_gloss": "drink_preference=coffee",
                "confidence": 0.8,
                "active_flag": True,
                "revision_count": 0,
                "soft_role_scores": {"stable": 0.2, "preference": 0.9},
            }
        ],
        "new_observation": observation["input_observation"],
        "target_action": "overwrite",
        "target_flags": {"promote": False, "stale_old": True},
        "affected_slot_ids": [f"slot_old_{split}"],
        "_meta": {"dataset": "sgd", "sample_id": f"life-{split}", "task_name": "lifecycle_prediction"},
    }
    belief = {
        "query": "What drink does the user currently prefer?",
        "memory_slots": [
            {
                "slot_id": f"slot_new_{split}",
                "bank": "residual",
                "entity": "user",
                "relation": "drink_preference",
                "canonical_gloss": "drink_preference=matcha",
                "confidence": 0.9,
                "active_flag": True,
                "revision_count": 0,
                "soft_role_scores": {"stable": 0.2, "preference": 0.9},
            }
        ],
        "target_belief_json": {
            "belief_items": [
                {
                    "relation": "drink_preference",
                    "value": "matcha",
                    "status": "active",
                    "time_scope": "current",
                    "support_slot_ids": [f"slot_new_{split}"],
                }
            ]
        },
        "retrieved_slot_ids": [f"slot_new_{split}"],
        "_meta": {"dataset": "sgd", "sample_id": f"belief-{split}", "task_name": "composition_to_belief"},
    }
    task_paths = {
        "slot_autoencoding": split_root / "slot_autoencoding.jsonl",
        "lifecycle_prediction": split_root / "lifecycle_prediction.jsonl",
        "composition_to_belief": split_root / "composition_to_belief.jsonl",
    }
    _write_jsonl(task_paths["slot_autoencoding"], [observation])
    _write_jsonl(task_paths["lifecycle_prediction"], [lifecycle])
    _write_jsonl(task_paths["composition_to_belief"], [belief])
    manifest_path = split_root / "stage2_prepared_samples_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_files": {task: str(path) for task, path in task_paths.items()},
            "task_counts": {task: 1 for task in task_paths},
        },
    )
    return manifest_path


def test_publish_v27_teacher_artifacts_writes_real_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs_v2"
    manifest_artifact = output_root / "artifacts" / "latest_stage2_v27_32k_manifest.json"
    split_payload = {
        split: {"prepared_manifest": str(_build_split_manifest(output_root, split))}
        for split in ("train", "val", "test")
    }
    _write_json(
        manifest_artifact,
        {
            "artifact_type": "stage2_v27_32k_manifest",
            "splits": split_payload,
        },
    )
    teacher_config = tmp_path / "minimax.yaml"
    teacher_config.write_text(
        "\n".join(
            [
                "project:",
                "  name: core_mem",
                "  stage: stage2",
                "llm:",
                "  provider: openai_compatible",
                "  api_key_env: GPT_AGENT_API_KEY",
                "  base_url: https://example.com/v1",
                "  model: MiniMax-M2.7",
                "embedding:",
                "  provider: sentence_transformers",
                "  model_name: sentence-transformers/all-MiniLM-L6-v2",
                "benchmarks:",
                "  primary: personamem",
                "  secondary: longmemeval_s",
                "  personamem: {}",
                "  longmemeval: {}",
                "outputs:",
                "  root: outputs_v2",
            ]
        ),
        encoding="utf-8",
    )

    published = publish_v27_teacher_artifacts(
        output_root,
        v27_manifest_artifact=manifest_artifact,
        teacher_config_path=teacher_config,
        train_max_rows=1,
        val_max_rows=1,
        test_max_rows=1,
        batch_size=1,
        provider=_FakeProvider(),
    )

    assert published["observation"]["total_labeled_examples"] == 3
    assert published["slot_assignment"]["total_labeled_examples"] == 3
    assert published["belief"]["total_labeled_examples"] == 3
    assert Path(published["observation"]["artifact_path"]).exists()
    assert Path(published["slot_assignment"]["artifact_path"]).exists()
    assert Path(published["belief"]["artifact_path"]).exists()
    observation_payload = json.loads(Path(published["observation"]["artifact_path"]).read_text(encoding="utf-8"))
    assert observation_payload["status"] == "completed"
    assert observation_payload["split_metrics"]["train"]["relation_match_rate"] == 1.0
    slot_payload = json.loads(Path(published["slot_assignment"]["artifact_path"]).read_text(encoding="utf-8"))
    assert slot_payload["split_metrics"]["train"]["target_action_match_rate"] == 1.0
    belief_payload = json.loads(Path(published["belief"]["artifact_path"]).read_text(encoding="utf-8"))
    assert belief_payload["split_metrics"]["train"]["first_value_match_rate"] == 1.0
