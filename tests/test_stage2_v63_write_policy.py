from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_stage2_v63_write_policy import compute_v63_write_policy


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v63_plan_only_is_low_score(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v63_plan.md").write_text(
        "Recall-Preserving Confidence-Aware Write Policy\ncore-worthy\nweak-but-keep\n",
        encoding="utf-8",
    )

    payload = compute_v63_write_policy(repo)

    assert payload["checks"]["plan_exists"] is True
    assert payload["score"] <= 20


def test_v63_v62_fallback_is_capped_without_four_way_policy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v63_plan.md").write_text(
        "Recall-Preserving Confidence-Aware Write Policy\ncore-worthy\nweak-but-keep\n",
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
            "trained_modules": ["query_conditioned_reader", "decision_head", "write_worthiness", "attribute_validity"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v62_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 4,
            "residual_bank_exists": True,
            "residual_bank_size": 42,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 64,
            "bank_compaction_enabled": True,
            "revision_chain_enabled": True,
            "typed_residual_relations": ["reason_fact", "social_fact", "environment_fact"],
            "other_fact_share": 0.0,
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
            "no_calibration_correct": 160,
            "text_only_correct": 170,
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
    _write_json(artifact_root / "latest_stage2_v62_decision.json", {"result_type": "negative_result"})

    payload = compute_v63_write_policy(repo)

    assert payload["checks"]["confidence_aware_four_way_policy"] is False
    assert any(cap["reason"] == "confidence_aware_four_way_policy_missing" for cap in payload["caps"])
    assert payload["score"] <= 60


def test_v63_weak_keep_missing_is_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v63_plan.md").write_text(
        "Recall-Preserving Confidence-Aware Write Policy\ncore-worthy\nweak-but-keep\n",
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
        artifact_root / "latest_stage2_v63_write_policy_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "semantic_slot_matching_trained": True,
            "slot_compaction_trained": True,
            "trained_modules": ["query_conditioned_reader", "decision_head", "write_policy", "confidence_aware_write"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_write_policy_eval.json",
        {
            "confidence_aware_policy_in_authoritative_write_path": True,
            "policy_labels": ["core-worthy", "residual-worthy", "weak-but-keep", "drop"],
            "support_coverage_recall": 0.76,
            "disabled_support_coverage_recall": 0.62,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 16,
            "residual_bank_exists": True,
            "residual_bank_size": 80,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 128,
            "typed_residual_relations": ["reason_fact", "social_fact", "environment_fact"],
            "other_fact_share": 0.0,
            "revision_chain_enabled": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_internal_eval.json",
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
        artifact_root / "latest_stage2_v63_personamem_no_routing.json",
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
        artifact_root / "latest_stage2_v63_error_attribution.json",
        {
            "never_written_count": 10,
            "written_but_compacted_away_count": 8,
            "written_but_reader_missed_count": 7,
            "reader_selected_but_decision_failed_count": 6,
        },
    )
    _write_json(artifact_root / "latest_stage2_v63_decision.json", {"result_type": "positive_gain", "significant_margin_confirmed": True})

    payload = compute_v63_write_policy(repo)

    assert payload["checks"]["confidence_aware_four_way_policy"] is True
    assert payload["checks"]["weak_but_keep_residual"] is False
    assert any(cap["reason"] == "weak_but_keep_not_preserved_in_residual" for cap in payload["caps"])
    assert payload["score"] <= 65


def test_v63_full_system_can_reach_stop_ready(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v63_plan.md").write_text(
        "Recall-Preserving Confidence-Aware Write Policy\ncore-worthy\nweak-but-keep\n",
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
        artifact_root / "latest_stage2_v63_write_policy_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "semantic_slot_matching_trained": True,
            "slot_compaction_trained": True,
            "trained_modules": [
                "query_conditioned_reader",
                "decision_head",
                "write_policy",
                "confidence_aware_write",
                "weak_keep_policy",
            ],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_write_policy_eval.json",
        {
            "confidence_aware_policy_in_authoritative_write_path": True,
            "policy_labels": ["core-worthy", "residual-worthy", "weak-but-keep", "drop"],
            "weak_but_keep_routed_to_residual": True,
            "support_coverage_recall": 0.82,
            "disabled_support_coverage_recall": 0.61,
            "write_recall": 0.79,
            "disabled_write_recall": 0.56,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 18,
            "residual_bank_exists": True,
            "residual_bank_size": 86,
            "weak_but_keep_residual_count": 24,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 148,
            "bank_compaction_enabled": True,
            "revision_chain_enabled": True,
            "duplicate_relation_clusters_reduced": True,
            "typed_residual_relations": ["reason_fact", "social_fact", "environment_fact", "temporal_fact"],
            "other_fact_share": 0.0,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_internal_eval.json",
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
        artifact_root / "latest_stage2_v63_personamem_no_routing.json",
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
        artifact_root / "latest_stage2_v63_error_attribution.json",
        {
            "never_written_count": 14,
            "written_but_compacted_away_count": 10,
            "written_but_reader_missed_count": 11,
            "reader_selected_but_decision_failed_count": 9,
        },
    )
    _write_json(artifact_root / "latest_stage2_v63_decision.json", {"result_type": "positive_gain", "significant_margin_confirmed": True})

    payload = compute_v63_write_policy(repo)

    assert payload["score"] == 100
    assert payload["stop_ready"] is True
