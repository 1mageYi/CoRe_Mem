from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v6_persistent_latent_memory import compute_v6_persistent_latent_memory


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v6_plan_only_is_capped_without_persistent_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v6_plan.md").write_text(
        "Persistent Core-Residual Latent Memory\nanswer-time confidence routing is disabled\n",
        encoding="utf-8",
    )

    payload = compute_v6_persistent_latent_memory(repo)

    assert payload["checks"]["plan_exists"] is True
    assert payload["checks"]["persistent_core_bank"] is False
    assert payload["score"] <= 20


def test_v6_rejects_v52_style_answer_time_routing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v6_plan.md").write_text(
        "Persistent Core-Residual Latent Memory\nanswer-time confidence routing is disabled\n",
        encoding="utf-8",
    )
    state_path = repo / "outputs_v2" / "checkpoints" / "state.pt"
    trace_path = repo / "outputs_v2" / "artifacts" / "write_trace.jsonl"
    state_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    state_path.write_text("state", encoding="utf-8")
    trace_path.write_text("{}\n", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_train.json",
        {
            "write_time_router_trained": True,
            "learned_update_actions": ["new_core", "new_residual", "overwrite", "promote_to_core"],
            "trained_modules": ["latent_reader", "belief_readout", "answer_head"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 3,
            "residual_bank_exists": True,
            "residual_bank_size": 5,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.pt",
            "write_trace_path": "outputs_v2/artifacts/write_trace.jsonl",
            "stream_observations_written": 12,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "evaluation_consumes_persistent_state": True,
            "authoritative_input": "persistent_core_residual_state",
            "raw_context_retrieval_disabled": True,
            "latent_only_accuracy": 0.5,
            "shuffled_bank_accuracy": 0.25,
            "full_accuracy": 0.6,
            "text_only_accuracy": 0.4,
            "core_only_accuracy": 0.3,
            "residual_only_accuracy": 0.35,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 258,
            "text_only_correct": 227,
            "raw_context_retrieval_disabled": True,
            "answer_time_routing_used": True,
            "score_mode": "confidence_routed_text_or_latent_margin",
            "gold_used_for_memory_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_arch_ablation.json",
        {
            "ablation_mode": "true_disabled_architecture_reruns",
            "disabled_writer_drops": True,
            "disabled_belief_drops": True,
            "disabled_core_residual_split_drops": True,
            "shuffled_persona_drops": True,
        },
    )
    _write_json(artifact_root / "latest_stage2_v6_decision.json", {"result_type": "positive_gain"})

    payload = compute_v6_persistent_latent_memory(repo)

    assert payload["checks"]["answer_time_routing_disabled"] is False
    assert payload["score"] <= 40
    assert any(cap["reason"] == "answer_time_confidence_routing_still_used" for cap in payload["caps"])


def test_v6_requires_meaningful_personamem_margin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v6_plan.md").write_text(
        "Persistent Core-Residual Latent Memory\nanswer-time confidence routing is disabled\n",
        encoding="utf-8",
    )
    state_path = repo / "outputs_v2" / "checkpoints" / "state.pt"
    trace_path = repo / "outputs_v2" / "artifacts" / "write_trace.jsonl"
    state_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    state_path.write_text("state", encoding="utf-8")
    trace_path.write_text("{}\n", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_train.json",
        {
            "write_time_router_trained": True,
            "learned_update_actions": ["new_core", "new_residual", "overwrite", "promote_to_core"],
            "trained_modules": ["latent_reader", "belief_readout", "answer_head"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 3,
            "residual_bank_exists": True,
            "residual_bank_size": 5,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.pt",
            "write_trace_path": "outputs_v2/artifacts/write_trace.jsonl",
            "stream_observations_written": 12,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "evaluation_consumes_persistent_state": True,
            "authoritative_input": "persistent_core_residual_state",
            "raw_context_retrieval_disabled": True,
            "latent_only_accuracy": 0.5,
            "shuffled_bank_accuracy": 0.25,
            "full_accuracy": 0.6,
            "text_only_accuracy": 0.4,
            "core_only_accuracy": 0.3,
            "residual_only_accuracy": 0.35,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 228,
            "text_only_correct": 227,
            "raw_context_retrieval_disabled": True,
            "answer_time_routing_used": False,
            "score_mode": "persistent_latent_readout",
            "gold_used_for_memory_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_arch_ablation.json",
        {
            "ablation_mode": "true_disabled_architecture_reruns",
            "disabled_writer_drops": True,
            "disabled_belief_drops": True,
            "disabled_core_residual_split_drops": True,
            "shuffled_persona_drops": True,
        },
    )
    _write_json(artifact_root / "latest_stage2_v6_decision.json", {"result_type": "positive_gain"})

    payload = compute_v6_persistent_latent_memory(repo)

    assert payload["checks"]["personamem_no_cal_beats_text_only"] is True
    assert payload["checks"]["personamem_significant_margin"] is False
    assert payload["score"] <= 85


def test_v6_full_persistent_system_can_reach_stop_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v6_plan.md").write_text(
        "Persistent Core-Residual Latent Memory\nanswer-time confidence routing is disabled\n",
        encoding="utf-8",
    )
    state_path = repo / "outputs_v2" / "checkpoints" / "state.pt"
    trace_path = repo / "outputs_v2" / "artifacts" / "write_trace.jsonl"
    state_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    state_path.write_text("state", encoding="utf-8")
    trace_path.write_text("{}\n", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_train.json",
        {
            "write_time_router_trained": True,
            "learned_update_actions": ["new_core", "new_residual", "overwrite", "promote_to_core"],
            "trained_modules": ["latent_reader", "belief_readout", "answer_head"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 3,
            "residual_bank_exists": True,
            "residual_bank_size": 5,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.pt",
            "write_trace_path": "outputs_v2/artifacts/write_trace.jsonl",
            "stream_observations_written": 12,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "evaluation_consumes_persistent_state": True,
            "authoritative_input": "persistent_core_residual_state",
            "raw_context_retrieval_disabled": True,
            "latent_only_accuracy": 0.5,
            "shuffled_bank_accuracy": 0.25,
            "full_accuracy": 0.6,
            "text_only_accuracy": 0.4,
            "core_only_accuracy": 0.3,
            "residual_only_accuracy": 0.35,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 258,
            "text_only_correct": 227,
            "raw_context_retrieval_disabled": True,
            "answer_time_routing_used": False,
            "score_mode": "persistent_latent_readout",
            "gold_used_for_memory_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_arch_ablation.json",
        {
            "ablation_mode": "true_disabled_architecture_reruns",
            "disabled_writer_drops": True,
            "disabled_belief_drops": True,
            "disabled_core_residual_split_drops": True,
            "shuffled_persona_drops": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_decision.json",
        {"result_type": "positive_gain", "significant_margin_confirmed": True},
    )

    payload = compute_v6_persistent_latent_memory(repo)

    assert payload["score"] == 100
    assert payload["stop_ready"] is True
