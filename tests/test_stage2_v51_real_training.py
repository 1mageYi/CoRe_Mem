from __future__ import annotations

import json
from pathlib import Path

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
