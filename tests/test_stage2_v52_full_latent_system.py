from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from scripts.publish_stage2_v52_backbone_compare import publish_v52_backbone_compare
from scripts.train_stage2_v52_multitask import train_v52_multitask
from scripts.verify_stage2_v52_full_latent_system import compute_v52_full_latent_system


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v52_baseline_caps_without_full_system(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v52_plan.md").write_text(
        "V5.2 Full Learned Latent Memory System\ntext-only\n", encoding="utf-8"
    )

    payload = compute_v52_full_latent_system(repo)

    assert payload["score"] == 4
    assert payload["applied_cap"] == 20
    assert payload["checks"]["multi_task_training"] is False


def test_v52_v51_style_single_retrieval_model_is_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    checkpoint = repo / "outputs_v2" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v52_backbone_compare.json",
        {
            "pretrained_weights_loaded": True,
            "evaluation_backend": "sentence_transformers",
            "loaded_backbones": ["BAAI/bge-base-en-v1.5"],
            "attempted_backbones": ["BAAI/bge-base-en-v1.5"],
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_multitask_training.json",
        {
            "checkpoint_path": "outputs_v2/checkpoints/model.pt",
            "train_samples": 12000,
            "trained_tasks": ["retrieval_alignment"],
            "trained_modules": ["query_projection", "slot_projection"],
            "retrieval_only": True,
            "uses_personamem_gold": False,
        },
    )

    payload = compute_v52_full_latent_system(repo)

    assert payload["checks"]["at_least_two_real_backbones"] is False
    assert payload["checks"]["not_retrieval_only"] is False
    assert payload["score"] <= 20


def test_v52_multitask_without_personamem_text_gain_is_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    checkpoint = repo / "outputs_v2" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v52_backbone_compare.json",
        {
            "pretrained_weights_loaded": True,
            "evaluation_backend": "sentence_transformers",
            "loaded_backbones": ["bge", "e5"],
            "attempted_backbones": ["bge", "e5", "contriever"],
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_multitask_training.json",
        {
            "checkpoint_path": "outputs_v2/checkpoints/model.pt",
            "train_samples": 24000,
            "trained_tasks": [
                "retrieval_alignment",
                "slot_autoencoding",
                "composition_to_belief",
                "lifecycle_prediction",
            ],
            "trained_modules": [
                "encoder_adapter",
                "latent_reader",
                "write_controller",
                "belief_decoder",
            ],
            "uses_personamem_gold": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_full_latent_eval.json",
        {
            "trained_beats_frozen": True,
            "multitask_latent": 0.8,
            "retrieval_only_latent": 0.7,
            "latent_only": 0.8,
            "shuffled_latent": 0.5,
            "full": 0.9,
            "text_only": 0.6,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_ablation_summary.json",
        {
            "no_controller_drops": True,
            "no_belief_drops": True,
            "no_core_residual_drops": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_personamem_full589.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 210,
            "text_only_correct": 214,
            "gold_used_for_memory_substrate": False,
        },
    )

    payload = compute_v52_full_latent_system(repo)

    assert payload["checks"]["personamem_no_cal_beats_text_only"] is False
    assert payload["score"] <= 80


def test_v52_full_system_can_reach_stop_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v52_plan.md").write_text(
        "V5.2 Full Learned Latent Memory System\ntext-only\n", encoding="utf-8"
    )
    checkpoint = repo / "outputs_v2" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v52_backbone_compare.json",
        {
            "pretrained_weights_loaded": True,
            "evaluation_backend": "sentence_transformers",
            "loaded_backbones": ["bge", "e5"],
            "attempted_backbones": ["bge", "e5", "contriever"],
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_multitask_training.json",
        {
            "checkpoint_path": "outputs_v2/checkpoints/model.pt",
            "train_samples": 24000,
            "trained_tasks": [
                "retrieval_alignment",
                "slot_autoencoding",
                "composition_to_belief",
                "lifecycle_prediction",
            ],
            "trained_modules": [
                "encoder_adapter",
                "latent_reader",
                "write_controller",
                "belief_decoder",
            ],
            "uses_personamem_gold": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_full_latent_eval.json",
        {
            "trained_beats_frozen": True,
            "multitask_latent": 0.82,
            "retrieval_only_latent": 0.7,
            "latent_only": 0.82,
            "shuffled_latent": 0.52,
            "full": 0.91,
            "text_only": 0.6,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_ablation_summary.json",
        {
            "no_controller_drops": True,
            "no_belief_drops": True,
            "no_core_residual_drops": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v52_personamem_full589.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 225,
            "text_only_correct": 214,
            "gold_used_for_memory_substrate": False,
        },
    )
    _write_json(artifact_root / "latest_stage2_v52_decision.json", {"result_type": "positive_gain"})

    payload = compute_v52_full_latent_system(repo)

    assert payload["score"] == 100
    assert payload["stop_ready"] is True


def test_v52_backbone_publisher_records_multiple_gold_free_real_backbones(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    retrieval_file = repo / "outputs_v2" / "artifacts" / "stage2_v27_32k" / "val" / "retrieval_alignment.jsonl"
    retrieval_file.parent.mkdir(parents=True)
    retrieval_file.write_text(
        json.dumps(
            {
                "query": "What food does the user currently prefer?",
                "positive_slot": {"relation": "food_preference", "canonical_gloss": "food_preference=breakfast"},
                "negative_slots": [
                    {"relation": "location", "canonical_gloss": "location=Denver"},
                    {"relation": "occupation", "canonical_gloss": "occupation=teacher"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, device: str, cache_folder: str) -> None:
            self.model_id = model_id
            self.device = device
            self.cache_folder = cache_folder

        def encode(self, texts, **kwargs):
            vectors = []
            for text in texts:
                if "breakfast" in text or "food" in text:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return np.asarray(vectors, dtype=np.float32)

        def get_embedding_dimension(self) -> int:
            return 2

        def parameters(self):
            return []

        def modules(self):
            return iter([self])

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    payload = publish_v52_backbone_compare(
        root=repo,
        retrieval_file=retrieval_file,
        model_ids=("fake/bge", "fake/e5", "fake/contriever"),
        eval_samples=1,
        device="cpu",
    )

    assert payload["artifact_type"] == "stage2_v52_backbone_compare"
    assert payload["pretrained_weights_loaded"] is True
    assert payload["minimum_two_real_backbones_loaded"] is True
    assert payload["uses_personamem_gold"] is False
    assert payload["evaluation_backend"] == "sentence_transformers"
    assert payload["loaded_backbones"] == ["fake/bge", "fake/e5", "fake/contriever"]
    assert payload["results"][0]["metrics"]["top1_accuracy"] == 1.0


def test_v52_multitask_trainer_publishes_checkpoint_and_all_task_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v52_backbone_compare.json",
        {
            "selected_backbone": "fake/bge",
            "loaded_backbones": ["fake/bge", "fake/e5"],
        },
    )
    data_root = artifact_root / "stage2_v27_32k"
    for split in ("train", "val"):
        (data_root / split).mkdir(parents=True)
    retrieval_row = {
        "query": "What food does the user prefer?",
        "positive_slot": {"bank": "residual", "relation": "food_preference", "canonical_gloss": "food_preference=breakfast"},
        "negative_slots": [{"relation": "location", "canonical_gloss": "location=Denver"}],
    }
    slot_row = {
        "input_observation": {
            "relation": "food_preference",
            "value": "breakfast",
            "canonical_gloss": "food_preference=breakfast",
        },
        "target_record": {
            "relation": "food_preference",
            "value": "breakfast",
            "status": "active",
            "time_scope": "current",
        },
    }
    composition_row = {
        "query": "What food does the user prefer?",
        "memory_slots": [
            {"slot_id": "slot_food", "relation": "food_preference", "canonical_gloss": "food_preference=breakfast"},
            {"slot_id": "slot_loc", "relation": "location", "canonical_gloss": "location=Denver"},
        ],
        "retrieved_slot_ids": ["slot_food"],
        "target_belief_json": {
            "belief_items": [
                {
                    "relation": "food_preference",
                    "value": "breakfast",
                    "status": "active",
                    "time_scope": "current",
                }
            ]
        },
    }
    lifecycle_row = {
        "memory_context": [{"relation": "food_preference", "canonical_gloss": "food_preference=toast"}],
        "new_observation": {
            "relation": "food_preference",
            "value": "breakfast",
            "canonical_gloss": "food_preference=breakfast",
        },
        "target_action": "overwrite",
    }
    for split in ("train", "val"):
        (data_root / split / "retrieval_alignment.jsonl").write_text(json.dumps(retrieval_row) + "\n", encoding="utf-8")
        (data_root / split / "slot_autoencoding.jsonl").write_text(json.dumps(slot_row) + "\n", encoding="utf-8")
        (data_root / split / "composition_to_belief.jsonl").write_text(json.dumps(composition_row) + "\n", encoding="utf-8")
        (data_root / split / "lifecycle_prediction.jsonl").write_text(json.dumps(lifecycle_row) + "\n", encoding="utf-8")

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, device: str, cache_folder: str) -> None:
            self.model_id = model_id

        def encode(self, texts, **kwargs):
            vectors = []
            for text in texts:
                lowered = str(text).lower()
                if "breakfast" in lowered or "food" in lowered:
                    vectors.append([1.0, 0.0, 0.0, 0.0])
                elif "denver" in lowered or "location" in lowered:
                    vectors.append([0.0, 1.0, 0.0, 0.0])
                else:
                    vectors.append([0.0, 0.0, 1.0, 0.0])
            return np.asarray(vectors, dtype=np.float32)

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    payload = train_v52_multitask(
        root=repo,
        data_root=data_root,
        backbone_artifact=artifact_root / "latest_stage2_v52_backbone_compare.json",
        retrieval_samples=1,
        slot_samples=1,
        composition_samples=1,
        lifecycle_samples=1,
        val_samples=1,
        epochs=1,
        batch_size=1,
        encode_batch_size=1,
        latent_dim=8,
        device_name="cpu",
    )

    assert payload["artifact_type"] == "stage2_v52_multitask_training"
    assert payload["checkpoint_exists"] is True
    assert set(payload["trained_tasks"]) == {
        "retrieval_alignment",
        "slot_autoencoding",
        "composition_to_belief",
        "lifecycle_prediction",
    }
    assert payload["retrieval_only"] is False
    assert payload["uses_personamem_gold"] is False
    assert "latent_resampler" in payload["trained_modules"]
    assert "belief_decoder" in payload["trained_modules"]
