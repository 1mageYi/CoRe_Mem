from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v62_write_quality import compute_v62_write_quality


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v62_plan_only_is_low_score(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v62_plan.md").write_text(
        "Learned Write-Worthiness / Attribute-Validity Before Extraction\nDo not go back to answer-time routing\n",
        encoding="utf-8",
    )

    payload = compute_v62_write_quality(repo)

    assert payload["checks"]["plan_exists"] is True
    assert payload["score"] <= 20


def test_v62_v61_fallback_is_capped_without_write_quality_models(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v62_plan.md").write_text(
        "Learned Write-Worthiness / Attribute-Validity Before Extraction\nDo not go back to answer-time routing\n",
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
            "slot_compaction_trained": True,
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
            "typed_residual_relations": ["reason_fact", "social_fact", "environment_fact"],
            "other_fact_share": 0.0,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_query_slot_reader",
            "authoritative_reader_uses_dot_product_only": False,
            "typed_residuals_used": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 191,
            "text_only_correct": 180,
            "option_only_correct": 235,
            "raw_context_retrieval_disabled": True,
            "answer_time_routing_used": False,
            "score_mode": "learned_decision_head",
            "decision_mode": "learned_answer_head",
            "handcrafted_option_scoring_used": False,
            "lexical_jaccard_used": False,
            "gold_used_for_memory_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v61_arch_ablation.json",
        {
            "ablation_mode": "true_disabled_architecture_reruns",
        },
    )
    _write_json(artifact_root / "latest_stage2_v61_decision.json", {"result_type": "negative_result"})

    payload = compute_v62_write_quality(repo)

    assert payload["checks"]["authoritative_learned_reader"] is True
    assert payload["checks"]["authoritative_learned_decision"] is True
    assert payload["checks"]["learned_write_worthiness_authoritative"] is False
    assert any(cap["reason"] == "write_worthiness_not_in_authoritative_write_path" for cap in payload["caps"])
    assert payload["score"] <= 60


def test_v62_attribute_validity_missing_is_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v62_plan.md").write_text(
        "Learned Write-Worthiness / Attribute-Validity Before Extraction\nDo not go back to answer-time routing\n",
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
        artifact_root / "latest_stage2_v62_write_quality_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "semantic_slot_matching_trained": True,
            "write_worthiness_trained": True,
            "trained_modules": ["query_conditioned_reader", "decision_head", "write_worthiness"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_write_quality_eval.json",
        {
            "write_worthiness_in_authoritative_write_path": True,
            "uses_raw_dialogue_observations": True,
            "hard_negatives_enabled": True,
            "hard_negative_count": 32,
            "write_worthiness_eval_accuracy": 0.88,
            "disabled_write_worthiness_accuracy": 0.61,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 10,
            "residual_bank_exists": True,
            "residual_bank_size": 20,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 30,
            "bank_precision_estimate": 0.83,
            "typed_residual_relations": ["reason_fact", "social_fact", "environment_fact"],
            "other_fact_share": 0.0,
            "revision_chain_enabled": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_query_slot_reader",
            "authoritative_reader_uses_dot_product_only": False,
            "typed_residuals_used": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 240,
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
        artifact_root / "latest_stage2_v62_arch_ablation.json",
        {"ablation_mode": "true_disabled_architecture_reruns"},
    )
    _write_json(artifact_root / "latest_stage2_v62_decision.json", {"result_type": "positive_gain", "significant_margin_confirmed": True})

    payload = compute_v62_write_quality(repo)

    assert payload["checks"]["learned_write_worthiness_authoritative"] is True
    assert payload["checks"]["attribute_validity_prewrite"] is False
    assert any(cap["reason"] == "attribute_validity_not_in_prewrite_path" for cap in payload["caps"])
    assert payload["score"] <= 65


def test_v62_full_system_can_reach_stop_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v62_plan.md").write_text(
        "Learned Write-Worthiness / Attribute-Validity Before Extraction\nDo not go back to answer-time routing\n",
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
        artifact_root / "latest_stage2_v62_write_quality_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "semantic_slot_matching_trained": True,
            "slot_compaction_trained": True,
            "write_worthiness_trained": True,
            "attribute_validity_trained": True,
            "trained_modules": [
                "query_conditioned_reader",
                "decision_head",
                "semantic_slot_matching",
                "write_worthiness",
                "attribute_validity",
            ],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_write_quality_eval.json",
        {
            "write_worthiness_in_authoritative_write_path": True,
            "attribute_validity_before_write": True,
            "uses_raw_dialogue_observations": True,
            "hard_negatives_enabled": True,
            "hard_negative_count": 64,
            "conflict_negative_count": 24,
            "write_worthiness_eval_accuracy": 0.91,
            "disabled_write_worthiness_accuracy": 0.66,
            "attribute_validity_f1": 0.84,
            "disabled_attribute_validity_f1": 0.57,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 10,
            "residual_bank_exists": True,
            "residual_bank_size": 20,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 30,
            "bank_precision_estimate": 0.84,
            "typed_residual_relations": ["reason_fact", "social_fact", "environment_fact", "constraint_fact"],
            "other_fact_share": 0.0,
            "revision_chain_enabled": True,
            "duplicate_relation_clusters_reduced": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_query_slot_reader",
            "authoritative_reader_uses_dot_product_only": False,
            "typed_residuals_used": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_personamem_no_routing.json",
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
        artifact_root / "latest_stage2_v62_arch_ablation.json",
        {"ablation_mode": "true_disabled_architecture_reruns"},
    )
    _write_json(
        artifact_root / "latest_stage2_v62_decision.json",
        {"result_type": "positive_gain", "significant_margin_confirmed": True},
    )

    payload = compute_v62_write_quality(repo)

    assert payload["score"] == 100
    assert payload["stop_ready"] is True
