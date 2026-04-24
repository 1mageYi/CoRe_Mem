from __future__ import annotations

import json
from pathlib import Path

from core_mem.v2.schemas import Observation
from core_mem.v2.v62_write_memory import DialogueTurn
from core_mem.v2.v64_observation_proposal import build_learned_observation, merge_candidate_observations
from scripts.verify_stage2_v64_observation_proposal import compute_v64_observation_proposal


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _observation(**overrides: object) -> Observation:
    payload = {
        "obs_id": "obs_test",
        "source_dataset": "unit",
        "source_dialogue_id": "dialogue",
        "source_turn_id": "0",
        "session_id": "dialogue",
        "speaker": "user",
        "entity": "user",
        "relation": "music_preference",
        "value": "producing music with software",
        "value_type": "preference",
        "time_scope": "current",
        "status_hint": "active",
        "polarity": "positive",
        "confidence": 0.82,
        "evidence_text": "I like producing music with software.",
        "canonical_gloss": "music_preference=producing music with software",
        "metadata": {},
    }
    payload.update(overrides)
    return Observation.from_dict(payload)


def test_v64_merge_preserves_rule_and_learned_provenance() -> None:
    rule = _observation(metadata={"candidate_provenance": ["rule"]})
    learned = _observation(
        obs_id="obs_learned",
        value="producing music with software and digital tools",
        canonical_gloss="music_preference=producing music with software and digital tools",
        evidence_text="I like producing music with software and digital tools.",
        confidence=0.64,
        metadata={"candidate_provenance": ["learned"]},
    )

    merged, stats = merge_candidate_observations([rule], [learned])

    assert len(merged) == 1
    assert stats.deduped_candidate_count == 1
    assert merged[0].metadata["candidate_provenance"] == ["learned", "rule"]


def test_v64_build_learned_observation_keeps_local_clause_semantics() -> None:
    turn = DialogueTurn(
        source_dialogue_id="dialogue",
        source_turn_id="2",
        session_id="dialogue",
        speaker="user",
        text="I felt too pressured by reading challenge deadlines.",
        context_text="I still enjoy visiting local libraries.",
    )

    observation = build_learned_observation(
        turn,
        "I felt too pressured by reading challenge deadlines",
        proposer_score=0.73,
        clause_index=0,
    )

    assert observation is not None
    assert observation.relation == "hobby"
    assert observation.polarity == "negative"
    assert observation.metadata["parser"] == "learned_observation_proposer_v64"


def test_v64_plan_only_is_low_score(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v64_plan.md").write_text(
        "Learned Observation Proposal / Extraction Coverage\nhybrid candidate pool\nnever_written\n",
        encoding="utf-8",
    )

    payload = compute_v64_observation_proposal(repo)

    assert payload["checks"]["plan_exists"] is True
    assert payload["score"] <= 20


def test_v64_v63_fallback_is_capped_without_learned_proposer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v64_plan.md").write_text(
        "Learned Observation Proposal / Extraction Coverage\nhybrid candidate pool\nnever_written\n",
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
            "trained_modules": ["query_conditioned_reader", "decision_head", "write_policy"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 22,
            "residual_bank_exists": True,
            "residual_bank_size": 476,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 652,
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
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v63_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 180,
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
        artifact_root / "latest_stage2_v63_error_attribution.json",
        {
            "never_written_count": 409,
            "written_but_compacted_away_count": 0,
            "written_but_reader_missed_count": 0,
            "reader_selected_but_decision_failed_count": 0,
        },
    )
    _write_json(artifact_root / "latest_stage2_v63_decision.json", {"result_type": "negative_result"})

    payload = compute_v64_observation_proposal(repo)

    assert payload["checks"]["learned_observation_proposer_authoritative"] is False
    assert any(cap["reason"] == "learned_observation_proposer_missing" for cap in payload["caps"])
    assert payload["score"] <= 60


def test_v64_recall_without_never_written_reduction_is_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v64_plan.md").write_text(
        "Learned Observation Proposal / Extraction Coverage\nhybrid candidate pool\nnever_written\n",
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
        artifact_root / "latest_stage2_v64_observation_proposal_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "observation_proposer_trained": True,
            "trained_modules": [
                "query_conditioned_reader",
                "decision_head",
                "observation_proposer",
                "write_policy",
            ],
            "uses_personamem_gold_for_proposer": False,
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_observation_proposal_eval.json",
        {
            "authoritative_observation_proposer_in_candidate_path": True,
            "hybrid_candidate_pool_used": True,
            "rule_candidate_count": 120,
            "learned_candidate_count": 240,
            "merged_candidate_count": 300,
            "proposal_recall": 0.82,
            "parser_only_recall": 0.65,
            "support_recovery": 0.81,
            "parser_only_support_recovery": 0.62,
            "candidate_dedup_enabled": True,
            "candidate_validation_enabled": True,
            "write_recall": 0.85,
            "parser_only_write_recall": 0.61,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 24,
            "residual_bank_exists": True,
            "residual_bank_size": 480,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 700,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_query_slot_reader",
            "authoritative_reader_uses_dot_product_only": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 190,
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
        artifact_root / "latest_stage2_v64_error_attribution.json",
        {
            "never_written_count": 409,
            "parser_only_never_written_count": 409,
            "written_but_compacted_away_count": 0,
            "written_but_reader_missed_count": 0,
            "reader_selected_but_decision_failed_count": 0,
        },
    )
    _write_json(artifact_root / "latest_stage2_v64_decision.json", {"result_type": "negative_result"})

    payload = compute_v64_observation_proposal(repo)

    assert payload["checks"]["proposal_recall_positive"] is True
    assert payload["checks"]["never_written_reduced"] is False
    assert any(cap["reason"] == "never_written_not_reduced" for cap in payload["caps"])
    assert payload["score"] <= 75


def test_v64_positive_proposal_upgrade_can_reach_85_without_margin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v64_plan.md").write_text(
        "Learned Observation Proposal / Extraction Coverage\nhybrid candidate pool\nnever_written\n",
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
        artifact_root / "latest_stage2_v64_observation_proposal_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "observation_proposer_trained": True,
            "trained_modules": [
                "query_conditioned_reader",
                "decision_head",
                "observation_proposer",
                "write_policy",
            ],
            "uses_personamem_gold_for_proposer": False,
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_observation_proposal_eval.json",
        {
            "authoritative_observation_proposer_in_candidate_path": True,
            "hybrid_candidate_pool_used": True,
            "rule_candidate_count": 100,
            "learned_candidate_count": 220,
            "merged_candidate_count": 260,
            "deduped_candidate_count": 60,
            "proposal_recall": 0.82,
            "parser_only_recall": 0.64,
            "support_recovery": 0.51,
            "parser_only_support_recovery": 0.31,
            "candidate_dedup_enabled": True,
            "candidate_normalization_enabled": True,
            "candidate_validation_enabled": True,
            "write_recall": 0.51,
            "parser_only_write_recall": 0.31,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 22,
            "residual_bank_exists": True,
            "residual_bank_size": 476,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 652,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_query_slot_reader",
            "authoritative_reader_uses_dot_product_only": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 180,
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
        artifact_root / "latest_stage2_v64_error_attribution.json",
        {
            "never_written_count": 288,
            "parser_only_never_written_count": 409,
            "written_but_compacted_away_count": 42,
            "written_but_reader_missed_count": 30,
            "reader_selected_but_decision_failed_count": 49,
        },
    )
    _write_json(artifact_root / "latest_stage2_v64_decision.json", {"result_type": "negative_result"})

    payload = compute_v64_observation_proposal(repo)

    assert payload["checks"]["learned_observation_proposer_authoritative"] is True
    assert payload["checks"]["never_written_reduced"] is True
    assert payload["score"] == 85
