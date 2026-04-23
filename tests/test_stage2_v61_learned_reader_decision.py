from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v61_learned_reader_decision import compute_v61_learned_reader_decision


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v61_plan_only_is_low_score(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v61_plan.md").write_text(
        "Learned Reader/Decision over Persistent Memory\nDo not go back to answer-time routing\n",
        encoding="utf-8",
    )

    payload = compute_v61_learned_reader_decision(repo)

    assert payload["checks"]["plan_exists"] is True
    assert payload["score"] <= 20


def test_v61_fallback_v6_evidence_is_capped_without_learned_reader_and_decision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v61_plan.md").write_text(
        "Learned Reader/Decision over Persistent Memory\nDo not go back to answer-time routing\n",
        encoding="utf-8",
    )
    state_path = repo / "outputs_v2" / "checkpoints" / "state.json"
    trace_path = repo / "outputs_v2" / "artifacts" / "trace.jsonl"
    state_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    trace_path.write_text("{}\n", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "trained_modules": ["latent_reader", "query_conditioned_reader", "belief_readout"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 10,
            "residual_bank_exists": True,
            "residual_bank_size": 20,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 30,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_persistent_memory_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "authoritative_reader_uses_dot_product_only": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v6_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 146,
            "text_only_correct": 205,
            "option_only_correct": 235,
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
            "disabled_reader_drops": True,
            "disabled_decision_drops": True,
            "disabled_core_residual_split_drops": True,
        },
    )
    _write_json(artifact_root / "latest_stage2_v6_decision.json", {"result_type": "negative_result"})

    payload = compute_v61_learned_reader_decision(repo)

    assert payload["checks"]["persistent_core_bank"] is True
    assert payload["checks"]["authoritative_learned_reader"] is False
    assert any(cap["reason"] == "learned_reader_not_in_authoritative_path" for cap in payload["caps"])
    assert payload["score"] <= 45


def test_v61_rejects_handcrafted_decision_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v61_plan.md").write_text(
        "Learned Reader/Decision over Persistent Memory\nDo not go back to answer-time routing\n",
        encoding="utf-8",
    )
    state_path = repo / "outputs_v2" / "checkpoints" / "state.json"
    trace_path = repo / "outputs_v2" / "artifacts" / "trace.jsonl"
    state_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    trace_path.write_text("{}\n", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v61_reader_decision_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "trained_modules": ["query_conditioned_reader", "decision_head"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 10,
            "residual_bank_exists": True,
            "residual_bank_size": 20,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 30,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_cross_attention_reader",
            "authoritative_reader_uses_dot_product_only": False,
            "natural_language_queries": True,
            "gold_value_echo_queries": False,
            "hard_negative_source": "same_persona_same_relation",
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 240,
            "text_only_correct": 205,
            "option_only_correct": 235,
            "raw_context_retrieval_disabled": True,
            "answer_time_routing_used": False,
            "score_mode": "persistent_latent_readout",
            "decision_mode": "handcrafted_similarity",
            "handcrafted_option_scoring_used": True,
            "lexical_jaccard_used": True,
            "gold_used_for_memory_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_arch_ablation.json",
        {
            "ablation_mode": "true_disabled_architecture_reruns",
            "disabled_writer_drops": True,
            "disabled_reader_drops": True,
            "disabled_decision_drops": True,
            "disabled_core_residual_split_drops": True,
        },
    )
    _write_json(artifact_root / "latest_stage2_v61_decision.json", {"result_type": "positive_gain"})

    payload = compute_v61_learned_reader_decision(repo)

    assert payload["checks"]["authoritative_learned_reader"] is True
    assert payload["checks"]["authoritative_learned_decision"] is False
    assert any(cap["reason"] == "learned_decision_not_in_authoritative_path" for cap in payload["caps"])
    assert payload["score"] <= 50


def test_v61_full_system_can_reach_stop_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v61_plan.md").write_text(
        "Learned Reader/Decision over Persistent Memory\nDo not go back to answer-time routing\n",
        encoding="utf-8",
    )
    state_path = repo / "outputs_v2" / "checkpoints" / "state.json"
    trace_path = repo / "outputs_v2" / "artifacts" / "trace.jsonl"
    state_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    trace_path.write_text("{}\n", encoding="utf-8")
    artifact_root = repo / "outputs_v2" / "artifacts"
    _write_json(
        artifact_root / "latest_stage2_v61_reader_decision_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "semantic_slot_matching_trained": True,
            "trained_modules": ["query_conditioned_reader", "decision_head", "semantic_slot_matching"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 10,
            "residual_bank_exists": True,
            "residual_bank_size": 20,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 30,
            "bank_compaction_enabled": True,
            "revision_chain_enabled": True,
            "typed_residual_relations": ["reason_fact", "constraint_fact", "social_fact", "environment_fact"],
            "other_fact_share": 0.2,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_cross_attention_reader",
            "authoritative_reader_uses_dot_product_only": False,
            "natural_language_queries": True,
            "gold_value_echo_queries": False,
            "hard_negative_source": "same_persona_same_relation",
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 245,
            "text_only_correct": 205,
            "option_only_correct": 235,
            "raw_context_retrieval_disabled": True,
            "answer_time_routing_used": False,
            "score_mode": "learned_decision_head",
            "decision_mode": "learned_answer_head",
            "handcrafted_option_scoring_used": False,
            "lexical_jaccard_used": False,
            "gold_used_for_memory_substrate": False,
            "bootstrap_significant_vs_text_only": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_arch_ablation.json",
        {
            "ablation_mode": "true_disabled_architecture_reruns",
            "disabled_writer_drops": True,
            "disabled_reader_drops": True,
            "disabled_decision_drops": True,
            "disabled_core_residual_split_drops": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_decision.json",
        {
            "result_type": "positive_gain",
            "significant_margin_confirmed": True,
        },
    )

    payload = compute_v61_learned_reader_decision(repo)

    assert payload["score"] == 100
    assert payload["stop_ready"] is True

