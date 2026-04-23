from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from scripts.publish_stage2_v51_real_backbone import publish_real_backbone_compare
from scripts.verify_stage2_v51_real_training import compute_v51_real_training


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v51_baseline_caps_missing_real_training(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v51_plan.md").write_text(
        "V5.1 Real Pretrained Training\nscore cap\n", encoding="utf-8"
    )
    _write_json(
        repo / "outputs_v2" / "artifacts" / "latest_stage2_v5_encoder_compare.json",
        {"pretrained_weights_loaded": False},
    )

    payload = compute_v51_real_training(repo)

    assert payload["score"] == 10
    assert payload["applied_cap"] == 20
    assert payload["checks"]["real_pretrained_loaded"] is False
    assert payload["checks"]["real_checkpoint_exists"] is False


def test_v51_proxy_backend_is_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    checkpoint = repo / "outputs_v2" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    _write_json(
        repo / "outputs_v2" / "artifacts" / "latest_stage2_v51_real_backbone_compare.json",
        {
            "pretrained_weights_loaded": True,
            "backend": "deterministic_hashing_proxy",
            "uses_personamem_gold": False,
        },
    )
    _write_json(
        repo / "outputs_v2" / "artifacts" / "latest_stage2_v51_real_training.json",
        {
            "checkpoint_path": "outputs_v2/checkpoints/model.pt",
            "train_samples": 12000,
            "uses_personamem_gold": False,
        },
    )

    payload = compute_v51_real_training(repo)

    assert payload["checks"]["real_pretrained_loaded"] is True
    assert payload["checks"]["real_backend"] is False
    assert payload["score"] <= 20


def test_v51_real_training_without_personamem_eval_is_not_gold_cap_blocked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v51_plan.md").write_text(
        "V5.1 Real Pretrained Training\nscore cap\n", encoding="utf-8"
    )
    checkpoint = repo / "outputs_v2" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v5_encoder_compare.json",
        {"pretrained_weights_loaded": False},
    )
    _write_json(
        artifact_root / "latest_stage2_v51_real_backbone_compare.json",
        {
            "pretrained_weights_loaded": True,
            "backend": "sentence_transformers",
            "uses_personamem_gold": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v51_real_training.json",
        {
            "checkpoint_path": "outputs_v2/checkpoints/model.pt",
            "train_samples": 12000,
            "uses_personamem_gold": False,
        },
    )

    payload = compute_v51_real_training(repo)

    assert payload["checks"]["personamem_full589"] is False
    assert "gold_isolation_not_fully_proven" not in {cap["reason"] for cap in payload["caps"]}
    assert payload["score"] > 30


def test_v51_full_evidence_can_reach_stop_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v51_plan.md").write_text(
        "V5.1 Real Pretrained Training\nscore cap\n", encoding="utf-8"
    )
    checkpoint = repo / "outputs_v2" / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v5_encoder_compare.json",
        {"pretrained_weights_loaded": False},
    )
    _write_json(
        artifact_root / "latest_stage2_v51_real_backbone_compare.json",
        {
            "pretrained_weights_loaded": True,
            "backend": "sentence_transformers",
            "uses_personamem_gold": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v51_real_training.json",
        {
            "checkpoint_path": "outputs_v2/checkpoints/model.pt",
            "train_samples": 24000,
            "uses_personamem_gold": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v51_latent_eval.json",
        {
            "trained_beats_frozen": True,
            "latent_only": 0.5,
            "shuffled_latent": 0.2,
            "full": 0.6,
            "text_only": 0.4,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v51_personamem_full589.json",
        {
            "sample_count": 589,
            "no_calibration_accuracy": 0.35,
            "option_only_accuracy": 0.25,
            "random_baseline": 0.25,
            "gold_used_for_memory_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v51_real_training_decision.json",
        {"result_type": "positive_gain"},
    )

    payload = compute_v51_real_training(repo)

    assert payload["score"] == 100
    assert payload["stop_ready"] is True


def test_v51_real_backbone_publisher_records_gold_free_sentence_transformer(
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

    payload = publish_real_backbone_compare(
        root=repo,
        retrieval_file=retrieval_file,
        model_ids=("fake/bge",),
        eval_samples=1,
        device="cpu",
    )

    assert payload["pretrained_weights_loaded"] is True
    assert payload["uses_personamem_gold"] is False
    assert payload["evaluation_backend"] == "sentence_transformers"
    assert payload["loaded_backbones"] == ["fake/bge"]
    assert payload["results"][0]["metrics"]["top1_accuracy"] == 1.0
