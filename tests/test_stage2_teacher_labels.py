from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_stage2_data import (
    _build_matched_teacher_vs_silver_split_manifests,
    publish_v27_teacher_artifacts,
    publish_v28_teacher_suite,
)


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
                    "relation": "drink_preference",
                    "value": example["raw_observation"]["evidence_text"].split("I like ", 1)[1].rstrip("."),
                    "time_scope": "current",
                    "status_hint": "active",
                    "polarity": "positive",
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


class _RetryObservationProvider(_FakeProvider):
    def __init__(self) -> None:
        self._observation_calls = 0

    def chat(self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int | None = None):
        examples = json.loads(prompt.split("examples=", 1)[1])
        if "teacher_task: observation" in prompt:
            self._observation_calls += 1
            if len(examples) > 1 and self._observation_calls == 1:
                broken = []
                for example in examples:
                    label = {
                        "sample_id": example["sample_id"],
                        "relation": "drink_preference",
                        "value": example["raw_observation"]["evidence_text"].split("I like ", 1)[1].rstrip("."),
                        "time_scope": "current",
                        "status_hint": "active",
                        "polarity": "positive",
                    }
                    if example["sample_id"] == "slot-train":
                        label.pop("polarity")
                    broken.append(label)
                return _FakeResponse(json.dumps({"labels": broken}, ensure_ascii=False))
        return super().chat(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)


class _MissingLabelsListProvider(_FakeProvider):
    def __init__(self) -> None:
        self._observation_calls = 0

    def chat(self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int | None = None):
        examples = json.loads(prompt.split("examples=", 1)[1])
        if "teacher_task: observation" in prompt:
            self._observation_calls += 1
            if len(examples) > 1 and self._observation_calls == 1:
                return _FakeResponse(json.dumps({"note": "labels omitted"}, ensure_ascii=False))
        return super().chat(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)


class _BeliefNormalizationProvider(_FakeProvider):
    def chat(self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int | None = None):
        examples = json.loads(prompt.split("examples=", 1)[1])
        if "teacher_task: belief_fields" in prompt:
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
                                "status": "affirmed",
                                "time_scope": "temporal",
                                "support_slot_ids": [first_slot["slot_id"]],
                            }
                        ],
                    }
                )
            return _FakeResponse(json.dumps({"labels": labels}, ensure_ascii=False))
        return super().chat(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)


class _CountingProvider(_FakeProvider):
    def __init__(self) -> None:
        self.observation_calls = 0

    def chat(self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int | None = None):
        if "teacher_task: observation" in prompt:
            self.observation_calls += 1
        return super().chat(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)


class _InterruptingObservationProvider(_FakeProvider):
    def __init__(self) -> None:
        self._observation_calls = 0

    def chat(self, prompt: str, *, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int | None = None):
        if "teacher_task: observation" in prompt:
            self._observation_calls += 1
            if self._observation_calls == 2:
                raise RuntimeError("simulated provider interruption")
        return super().chat(prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)


def _build_split_manifest(root: Path, split: str, row_count: int = 1, *, aligned_sample_ids: bool = False) -> Path:
    split_root = root / "artifacts" / "stage2_v27_32k" / split
    observations = []
    lifecycles = []
    beliefs = []
    for index in range(row_count):
        suffix = f"{split}-{index}"
        drink = f"matcha-{index}"
        observation = {
            "input_observation": {
                "obs_id": f"obs_{suffix}",
                "source_dataset": "sgd",
                "source_dialogue_id": f"dlg_{suffix}",
                "source_turn_id": f"turn_{suffix}",
                "session_id": f"sess_{suffix}",
                "speaker": "user",
                "entity": "user",
                "relation": "drink_preference",
                "value": drink,
                "value_type": "preference",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.9,
                "evidence_text": f"I like {drink}.",
                "canonical_gloss": f"drink_preference={drink}",
                "metadata": {},
            },
            "target_record": {
                "entity": "user",
                "relation": "drink_preference",
                "value": drink,
                "time_scope": "current",
                "status": "active",
            },
            "_meta": {
                "dataset": "sgd",
                "sample_id": suffix if aligned_sample_ids else f"slot-{suffix}",
                "task_name": "slot_autoencoding",
            },
        }
        lifecycle = {
            "memory_context": [
                {
                    "slot_id": f"slot_old_{suffix}",
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
            "affected_slot_ids": [f"slot_old_{suffix}"],
            "_meta": {
                "dataset": "sgd",
                "sample_id": suffix if aligned_sample_ids else f"life-{suffix}",
                "task_name": "lifecycle_prediction",
            },
        }
        belief = {
            "query": "What drink does the user currently prefer?",
            "memory_slots": [
                {
                    "slot_id": f"slot_new_{suffix}",
                    "bank": "residual",
                    "entity": "user",
                    "relation": "drink_preference",
                    "canonical_gloss": f"drink_preference={drink}",
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
                        "value": drink,
                        "status": "active",
                        "time_scope": "current",
                        "support_slot_ids": [f"slot_new_{suffix}"],
                    }
                ]
            },
            "retrieved_slot_ids": [f"slot_new_{suffix}"],
            "_meta": {
                "dataset": "sgd",
                "sample_id": suffix if aligned_sample_ids else f"belief-{suffix}",
                "task_name": "composition_to_belief",
            },
        }
        observations.append(observation)
        lifecycles.append(lifecycle)
        beliefs.append(belief)
    task_paths = {
        "slot_autoencoding": split_root / "slot_autoencoding.jsonl",
        "lifecycle_prediction": split_root / "lifecycle_prediction.jsonl",
        "composition_to_belief": split_root / "composition_to_belief.jsonl",
    }
    _write_jsonl(task_paths["slot_autoencoding"], observations)
    _write_jsonl(task_paths["lifecycle_prediction"], lifecycles)
    _write_jsonl(task_paths["composition_to_belief"], beliefs)
    manifest_path = split_root / "stage2_prepared_samples_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_files": {task: str(path) for task, path in task_paths.items()},
            "task_counts": {task: row_count for task in task_paths},
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


def test_publish_v27_teacher_artifacts_retries_incomplete_observation_labels(tmp_path: Path) -> None:
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
        batch_size=2,
        provider=_RetryObservationProvider(),
    )

    observation_payload = json.loads(Path(published["observation"]["artifact_path"]).read_text(encoding="utf-8"))
    assert observation_payload["status"] == "completed"
    assert observation_payload["total_labeled_examples"] == 3
    assert observation_payload["total_failed_examples"] == 0
    assert observation_payload["success_rate"] == 1.0


def test_publish_v27_teacher_artifacts_retries_when_batch_response_omits_labels_list(tmp_path: Path) -> None:
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
        batch_size=2,
        provider=_MissingLabelsListProvider(),
    )

    observation_payload = json.loads(Path(published["observation"]["artifact_path"]).read_text(encoding="utf-8"))
    assert observation_payload["status"] == "completed"
    assert observation_payload["total_labeled_examples"] == 3
    assert observation_payload["total_failed_examples"] == 0


def test_publish_v28_teacher_suite_writes_quality_audit_and_manifests(tmp_path: Path) -> None:
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

    published = publish_v28_teacher_suite(
        output_root,
        v27_manifest_artifact=manifest_artifact,
        teacher_config_path=teacher_config,
        train_max_rows=1,
        val_max_rows=1,
        test_max_rows=1,
        batch_size=1,
        provider=_FakeProvider(),
    )

    quality_audit = json.loads(Path(published["quality_audit"]["artifact_path"]).read_text(encoding="utf-8"))
    assert set(quality_audit["tasks_covered"]) == {"observation", "slot_assignment", "belief"}
    train_manifest = Path(published["teacher_enhanced_manifests"]["train"]["prepared_manifest"])
    assert train_manifest.exists()
    manifest_payload = json.loads(train_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["prepared_by"] == "stage2_v28_teacher_suite"
    assert manifest_payload["teacher_replaced_counts"]["lifecycle_prediction"] == 1


def test_publish_v27_teacher_artifacts_normalizes_belief_status_and_time_scope(tmp_path: Path) -> None:
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
        provider=_BeliefNormalizationProvider(),
    )

    belief_path = Path(published["belief"]["artifact_path"])
    belief_payload = json.loads(belief_path.read_text(encoding="utf-8"))
    train_labels = Path(belief_payload["label_files"]["train"])
    first_row = json.loads(train_labels.read_text(encoding="utf-8").splitlines()[0])
    belief_item = first_row["teacher_label"]["belief_items"][0]
    assert belief_item["status"] == "active"
    assert belief_item["time_scope"] == "recent_change"


def test_publish_v27_teacher_artifacts_resumes_from_incremental_label_files(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs_v2"
    manifest_artifact = output_root / "artifacts" / "latest_stage2_v27_32k_manifest.json"
    split_payload = {
        split: {
            "prepared_manifest": str(
                _build_split_manifest(output_root, split, row_count=2 if split == "train" else 1)
            )
        }
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

    try:
        publish_v27_teacher_artifacts(
            output_root,
            v27_manifest_artifact=manifest_artifact,
            teacher_config_path=teacher_config,
            train_max_rows=2,
            val_max_rows=1,
            test_max_rows=1,
            batch_size=1,
            provider=_InterruptingObservationProvider(),
        )
    except RuntimeError as exc:
        assert "simulated provider interruption" in str(exc)
    else:
        raise AssertionError("expected simulated interruption")

    partial_labels = output_root / "artifacts" / "stage2_v27_teacher" / "observation" / "train_labels.jsonl"
    assert partial_labels.exists()
    assert len(partial_labels.read_text(encoding="utf-8").splitlines()) == 1

    resume_provider = _CountingProvider()
    published = publish_v27_teacher_artifacts(
        output_root,
        v27_manifest_artifact=manifest_artifact,
        teacher_config_path=teacher_config,
        train_max_rows=2,
        val_max_rows=1,
        test_max_rows=1,
        batch_size=1,
        provider=resume_provider,
    )

    observation_payload = json.loads(Path(published["observation"]["artifact_path"]).read_text(encoding="utf-8"))
    assert observation_payload["total_labeled_examples"] == 4
    assert observation_payload["total_failed_examples"] == 0
    assert observation_payload["split_metrics"]["train"]["reused_labeled_examples"] == 1
    assert resume_provider.observation_calls == 3


def test_build_matched_teacher_vs_silver_split_manifests_filters_to_changed_rows(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs_v2"
    manifest_artifact = output_root / "artifacts" / "latest_stage2_v27_32k_manifest.json"
    split_payload = {
        split: {"prepared_manifest": str(_build_split_manifest(output_root, split, row_count=3))}
        for split in ("train", "val", "test")
    }
    _write_json(
        manifest_artifact,
        {
            "artifact_type": "stage2_v27_32k_manifest",
            "splits": split_payload,
        },
    )

    teacher_root = output_root / "artifacts" / "stage2_v27_teacher"
    for split in ("train", "val", "test"):
        _write_jsonl(
            teacher_root / "slot_assignment" / f"{split}_labels.jsonl",
            [
                {
                    "sample_id": f"life-{split}-1",
                    "teacher_label": {
                        "sample_id": f"life-{split}-1",
                        "target_action": "ignore",
                        "matched_slot_id": None,
                        "promote": False,
                        "stale_old": False,
                    },
                }
            ],
        )
        _write_jsonl(
            teacher_root / "belief" / f"{split}_labels.jsonl",
            [
                {
                    "sample_id": f"belief-{split}-2",
                    "teacher_label": {
                        "sample_id": f"belief-{split}-2",
                        "belief_items": [
                            {
                                "relation": "drink_preference",
                                "value": "teacher-matcha",
                                "status": "active",
                                "time_scope": "current",
                                "support_slot_ids": [f"slot_new_{split}-2"],
                            }
                        ],
                    },
                }
            ],
        )

    matched = _build_matched_teacher_vs_silver_split_manifests(
        output_root,
        v27_manifest_artifact=manifest_artifact,
        teacher_root=teacher_root,
    )

    train_silver_manifest = Path(matched["train"]["silver"]["prepared_manifest"])
    train_teacher_manifest = Path(matched["train"]["teacher"]["prepared_manifest"])
    silver_payload = json.loads(train_silver_manifest.read_text(encoding="utf-8"))
    teacher_payload = json.loads(train_teacher_manifest.read_text(encoding="utf-8"))

    assert silver_payload["task_counts"]["lifecycle_prediction"] == 1
    assert silver_payload["task_counts"]["composition_to_belief"] == 1
    assert set(silver_payload["task_counts"]) == {"slot_autoencoding", "lifecycle_prediction", "composition_to_belief"}
    assert teacher_payload["changed_sample_counts"]["lifecycle_prediction"] == 1
    assert teacher_payload["changed_sample_counts"]["composition_to_belief"] == 1
    assert teacher_payload["teacher_replaced_counts"]["lifecycle_prediction"] == 1
    assert teacher_payload["teacher_replaced_counts"]["composition_to_belief"] == 1

    lifecycle_rows = [
        json.loads(line)
        for line in Path(teacher_payload["task_files"]["lifecycle_prediction"]).read_text(encoding="utf-8").splitlines()
    ]
    belief_rows = [
        json.loads(line)
        for line in Path(teacher_payload["task_files"]["composition_to_belief"]).read_text(encoding="utf-8").splitlines()
    ]

    assert lifecycle_rows[0]["_meta"]["sample_id"] == "life-train-1"
    assert lifecycle_rows[0]["target_action"] == "ignore"
    assert belief_rows[0]["_meta"]["sample_id"] == "belief-train-2"
    assert belief_rows[0]["target_belief_json"]["belief_items"][0]["value"] == "teacher-matcha"


def test_build_matched_teacher_vs_silver_split_manifests_aligns_source_records_for_observation_changes(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs_v2"
    manifest_artifact = output_root / "artifacts" / "latest_stage2_v27_32k_manifest.json"
    split_payload = {
        split: {"prepared_manifest": str(_build_split_manifest(output_root, split, row_count=2, aligned_sample_ids=True))}
        for split in ("train", "val", "test")
    }
    _write_json(
        manifest_artifact,
        {
            "artifact_type": "stage2_v27_32k_manifest",
            "splits": split_payload,
        },
    )

    teacher_root = output_root / "artifacts" / "stage2_v27_teacher"
    for split in ("train", "val", "test"):
        _write_jsonl(
            teacher_root / "observation" / f"{split}_labels.jsonl",
            [
                {
                    "sample_id": f"{split}-0",
                    "source_label": {
                        "sample_id": f"{split}-0",
                        "relation": "drink_preference",
                        "value": "matcha-0",
                        "time_scope": "current",
                        "status_hint": "active",
                        "polarity": "positive",
                    },
                    "teacher_label": {
                        "sample_id": f"{split}-0",
                        "relation": "drink_preference",
                        "value": "oolong",
                        "time_scope": "current",
                        "status_hint": "active",
                        "polarity": "positive",
                    },
                }
            ],
        )

    matched = _build_matched_teacher_vs_silver_split_manifests(
        output_root,
        v27_manifest_artifact=manifest_artifact,
        teacher_root=teacher_root,
    )

    train_silver_manifest = Path(matched["train"]["silver"]["prepared_manifest"])
    train_teacher_manifest = Path(matched["train"]["teacher"]["prepared_manifest"])
    silver_payload = json.loads(train_silver_manifest.read_text(encoding="utf-8"))
    teacher_payload = json.loads(train_teacher_manifest.read_text(encoding="utf-8"))

    assert silver_payload["matched_sample_id_count"] == 1
    assert silver_payload["task_counts"]["slot_autoencoding"] == 1
    assert silver_payload["task_counts"]["lifecycle_prediction"] == 1
    assert silver_payload["task_counts"]["composition_to_belief"] == 1
    assert teacher_payload["changed_sample_counts"]["slot_autoencoding"] == 1
    assert teacher_payload["teacher_replaced_counts"]["slot_autoencoding"] == 1

    slot_rows = [
        json.loads(line)
        for line in Path(teacher_payload["task_files"]["slot_autoencoding"]).read_text(encoding="utf-8").splitlines()
    ]
    assert slot_rows[0]["_meta"]["sample_id"] == "train-0"
    assert slot_rows[0]["target_record"]["value"] == "oolong"


def test_build_matched_teacher_vs_silver_split_manifests_can_keep_lifecycle_silver(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs_v2"
    manifest_artifact = output_root / "artifacts" / "latest_stage2_v27_32k_manifest.json"
    split_payload = {
        split: {"prepared_manifest": str(_build_split_manifest(output_root, split, row_count=3, aligned_sample_ids=True))}
        for split in ("train", "val", "test")
    }
    _write_json(
        manifest_artifact,
        {
            "artifact_type": "stage2_v27_32k_manifest",
            "splits": split_payload,
        },
    )

    teacher_root = output_root / "artifacts" / "stage2_v27_teacher"
    for split in ("train", "val", "test"):
        _write_jsonl(
            teacher_root / "slot_assignment" / f"{split}_labels.jsonl",
            [
                {
                    "sample_id": f"{split}-1",
                    "source_label": {
                        "sample_id": f"{split}-1",
                        "target_action": "merge",
                        "matched_slot_id": f"slot_{split}_1",
                        "promote": True,
                        "stale_old": False,
                    },
                    "teacher_label": {
                        "sample_id": f"{split}-1",
                        "target_action": "ignore",
                        "matched_slot_id": None,
                        "promote": False,
                        "stale_old": False,
                    },
                }
            ],
        )
        _write_jsonl(
            teacher_root / "belief" / f"{split}_labels.jsonl",
            [
                {
                    "sample_id": f"{split}-2",
                    "source_label": {
                        "sample_id": f"{split}-2",
                        "belief_items": [
                            {
                                "relation": "drink_preference",
                                "value": "matcha-2",
                                "status": "active",
                                "time_scope": "current",
                                "support_slot_ids": [f"slot_new_{split}-2"],
                            }
                        ],
                    },
                    "teacher_label": {
                        "sample_id": f"{split}-2",
                        "belief_items": [
                            {
                                "relation": "drink_preference",
                                "value": "teacher-matcha",
                                "status": "active",
                                "time_scope": "current",
                                "support_slot_ids": [f"slot_new_{split}-2"],
                            }
                        ],
                    },
                }
            ],
        )

    matched = _build_matched_teacher_vs_silver_split_manifests(
        output_root,
        v27_manifest_artifact=manifest_artifact,
        teacher_root=teacher_root,
        matched_kinds=("slot_assignment", "belief"),
        teacher_apply_kinds=("belief",),
        manifest_namespace="stage2_v28_matched_manifests_obs_belief",
    )

    train_silver_manifest = Path(matched["train"]["silver"]["prepared_manifest"])
    train_teacher_manifest = Path(matched["train"]["teacher"]["prepared_manifest"])
    silver_payload = json.loads(train_silver_manifest.read_text(encoding="utf-8"))
    teacher_payload = json.loads(train_teacher_manifest.read_text(encoding="utf-8"))
    silver_lifecycle_rows = [
        json.loads(line)
        for line in Path(silver_payload["task_files"]["lifecycle_prediction"]).read_text(encoding="utf-8").splitlines()
    ]
    lifecycle_rows = [
        json.loads(line)
        for line in Path(teacher_payload["task_files"]["lifecycle_prediction"]).read_text(encoding="utf-8").splitlines()
    ]
    belief_rows = [
        json.loads(line)
        for line in Path(teacher_payload["task_files"]["composition_to_belief"]).read_text(encoding="utf-8").splitlines()
    ]

    assert teacher_payload["manifest_namespace"] == "stage2_v28_matched_manifests_obs_belief"
    assert teacher_payload["matched_sample_id_count"] == 2
    assert teacher_payload["teacher_apply_kinds"] == ["belief"]
    assert teacher_payload["changed_sample_counts"]["lifecycle_prediction"] == 1
    assert teacher_payload["teacher_replaced_counts"]["lifecycle_prediction"] == 0
    assert teacher_payload["teacher_replaced_counts"]["composition_to_belief"] == 1
    assert lifecycle_rows[0]["target_action"] == silver_lifecycle_rows[0]["target_action"]
    assert lifecycle_rows[0]["target_action"] != "ignore"
    assert any(
        row["target_belief_json"]["belief_items"][0]["value"] == "teacher-matcha"
        for row in belief_rows
    )
