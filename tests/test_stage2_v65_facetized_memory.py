from __future__ import annotations

import json
from pathlib import Path

from core_mem.v2.schemas import Observation
from core_mem.v2.v6_persistent_memory import PersistentCoreResidualMemory
from core_mem.v2.v61_learned_memory import _option_conditioned_query, _option_features
from core_mem.v2.v65_facetized_memory import facetize_observation, materialize_facet_observation
from scripts.verify_stage2_v65_facetized_memory import compute_v65_facetized_memory


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v65_plan_only_is_low_score(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v65_plan.md").write_text(
        "Facetized Observation-to-Memory Redesign\nfacetizer\nneeded_facet_missing\n",
        encoding="utf-8",
    )

    payload = compute_v65_facetized_memory(repo)

    assert payload["checks"]["plan_exists"] is True
    assert payload["score"] <= 20


def test_v65_v64_fallback_is_capped_without_facetizer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v65_plan.md").write_text(
        "Facetized Observation-to-Memory Redesign\nfacetizer\nneeded_facet_missing\n",
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
            "trained_modules": ["query_conditioned_reader", "decision_head", "observation_proposer", "write_policy"],
            "uses_personamem_gold_for_substrate": False,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v64_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 726,
            "residual_bank_exists": True,
            "residual_bank_size": 960,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 3415,
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
            "no_calibration_correct": 138,
            "text_only_correct": 199,
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
            "never_written_count": 442,
            "parser_only_never_written_count": 409,
        },
    )
    _write_json(artifact_root / "latest_stage2_v64_decision.json", {"result_type": "negative_result"})

    payload = compute_v65_facetized_memory(repo)

    assert payload["checks"]["facet_schema_defined"] is False
    assert payload["checks"]["facetizer_authoritative"] is False
    assert any(cap["reason"] == "facet_schema_missing" for cap in payload["caps"])
    assert payload["score"] <= 55


def test_v65_missing_needed_facet_reduction_is_capped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v65_plan.md").write_text(
        "Facetized Observation-to-Memory Redesign\nfacetizer\nneeded_facet_missing\n",
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
        artifact_root / "latest_stage2_v65_facetized_memory_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "facet_schema_defined": True,
            "facetizer_trained": True,
            "uses_personamem_gold_for_facetizer": False,
            "uses_personamem_gold_for_substrate": False,
            "trained_modules": [
                "query_conditioned_reader",
                "decision_head",
                "facetizer",
                "facet_aware_write",
            ],
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_facetized_memory_eval.json",
        {
            "facet_schema_defined": True,
            "facet_types": ["preference_target", "preference_mode", "environment_aversion"],
            "authoritative_facetizer_in_path": True,
            "facet_aware_write_utility_used": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 120,
            "residual_bank_exists": True,
            "residual_bank_size": 360,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 640,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_query_slot_reader",
            "authoritative_reader_uses_dot_product_only": False,
            "facet_aware_write_utility_used": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 181,
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
        artifact_root / "latest_stage2_v65_error_attribution.json",
        {
            "needed_facet_missing_count": 220,
            "parser_only_needed_facet_missing_count": 220,
            "wrong_sibling_facet_selected_count": 40,
            "parser_only_wrong_sibling_facet_selected_count": 60,
            "facet_written_but_reader_missed_count": 10,
            "facet_selected_but_decision_failed_count": 5,
        },
    )
    _write_json(artifact_root / "latest_stage2_v65_decision.json", {"result_type": "negative_result"})

    payload = compute_v65_facetized_memory(repo)

    assert payload["checks"]["facet_schema_defined"] is True
    assert payload["checks"]["facetizer_authoritative"] is True
    assert payload["checks"]["needed_facet_reduced"] is False
    assert any(cap["reason"] == "needed_facet_not_reduced" for cap in payload["caps"])
    assert payload["score"] <= 85


def test_v65_complete_payload_scores_100(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "v65_plan.md").write_text(
        "Facetized Observation-to-Memory Redesign\nfacetizer\nneeded_facet_missing\n",
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
        artifact_root / "latest_stage2_v65_facetized_memory_train.json",
        {
            "write_time_router_trained": True,
            "query_conditioned_reader_trained": True,
            "decision_head_trained": True,
            "facet_schema_defined": True,
            "facetizer_trained": True,
            "uses_personamem_gold_for_facetizer": False,
            "uses_personamem_gold_for_substrate": False,
            "trained_modules": [
                "query_conditioned_reader",
                "decision_head",
                "facetizer",
                "facet_aware_write",
            ],
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_facetized_memory_eval.json",
        {
            "facet_schema_defined": True,
            "facet_types": ["preference_target", "preference_mode", "environment_aversion", "update_reason"],
            "authoritative_facetizer_in_path": True,
            "facet_aware_write_utility_used": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_persistent_state.json",
        {
            "core_bank_exists": True,
            "core_bank_size": 140,
            "residual_bank_exists": True,
            "residual_bank_size": 380,
            "state_checkpoint_path": "outputs_v2/checkpoints/state.json",
            "write_trace_path": "outputs_v2/artifacts/trace.jsonl",
            "stream_observations_written": 700,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_internal_eval.json",
        {
            "write_time_router_applied_to_state": True,
            "raw_context_retrieval_disabled": True,
            "reader_in_authoritative_path": True,
            "authoritative_reader_path": "learned_query_slot_reader",
            "authoritative_reader_uses_dot_product_only": False,
            "facet_aware_write_utility_used": True,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_personamem_no_routing.json",
        {
            "sample_count": 589,
            "no_calibration_correct": 260,
            "text_only_correct": 200,
            "option_only_correct": 230,
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
        artifact_root / "latest_stage2_v65_error_attribution.json",
        {
            "needed_facet_missing_count": 140,
            "parser_only_needed_facet_missing_count": 220,
            "wrong_sibling_facet_selected_count": 15,
            "parser_only_wrong_sibling_facet_selected_count": 60,
            "facet_written_but_reader_missed_count": 8,
            "facet_selected_but_decision_failed_count": 3,
        },
    )
    _write_json(
        artifact_root / "latest_stage2_v65_decision.json",
        {
            "result_type": "positive_gain",
            "significant_margin_confirmed": True,
        },
    )

    payload = compute_v65_facetized_memory(repo)

    assert payload["score"] == 100
    assert payload["stop_ready"] is True


def test_v65_facetizer_splits_preference_and_reason_facets() -> None:
    observation = Observation(
        obs_id="obs_demo",
        source_dataset="demo",
        source_dialogue_id="dlg",
        source_turn_id="3",
        session_id="dlg",
        speaker="user",
        entity="user",
        relation="music_preference",
        value="producing music with software feels more spontaneous and less pressured",
        value_type="preference",
        time_scope="recent_change",
        status_hint="active",
        polarity="negative",
        confidence=0.8,
        evidence_text="Producing music with software feels more spontaneous and less pressured.",
        canonical_gloss="music_preference=producing music with software feels more spontaneous and less pressured",
        metadata={},
    )

    facet_types = {facet.facet_type for facet in facetize_observation(observation)}

    assert "preference_target" in facet_types
    assert "preference_mode" in facet_types
    assert "update_reason" in facet_types
    assert "temporal_state" in facet_types


def test_v65_preference_target_uses_salient_phrase_not_leading_boilerplate() -> None:
    observation = Observation(
        obs_id="obs_pref_phrase",
        source_dataset="demo",
        source_dialogue_id="dlg",
        source_turn_id="29",
        session_id="dlg",
        speaker="user",
        entity="user",
        relation="music_preference",
        value=(
            "I tried reviewing an album in 2019, but that did not stick either, as it felt unproductive. "
            "I think I was attempting to analyze the music in a way that did not align with my natural style, "
            "which is often more spontaneous and emotional rather than analytical."
        ),
        value_type="preference",
        time_scope="recent_change",
        status_hint="active",
        polarity="negative",
        confidence=0.8,
        evidence_text=(
            "I think I was attempting to analyze the music in a way that did not align with my natural style, "
            "which is often more spontaneous and emotional rather than analytical."
        ),
        canonical_gloss="music_preference=reviewing albums felt unproductive and too analytical",
        metadata={},
    )

    preference_target = next(facet for facet in facetize_observation(observation) if facet.facet_type == "preference_target")

    assert "spontaneous" in preference_target.facet_value
    assert "emotional" in preference_target.facet_value
    assert "tried" not in preference_target.facet_value


def test_v65_option_conditioned_query_uses_option_body_tokens_only() -> None:
    query = "What activity would fit the user best right now?"
    option = "(b) Producing music with software in a calm space would feel more personal."

    conditioned = _option_conditioned_query(query, option)

    assert conditioned.startswith(query)
    assert "(b)" not in conditioned
    assert "producing music software" in conditioned


def test_v65_option_features_expand_with_option_conditioned_readout() -> None:
    base_readout = {
        "query_key": [0.1, 0.2],
        "composed_key": [0.3, 0.4],
        "belief_items": [{"relation": "music_preference", "value": "album reviews"}],
        "selected": [],
    }
    conditioned_readout = {
        "query_key": [0.2, 0.3],
        "composed_key": [0.4, 0.5],
        "belief_items": [{"relation": "music_preference", "value": "producing music with software"}],
        "selected": [],
    }

    base_features = _option_features(base_readout, "What fits the user best?", "(b) Producing music with software")
    conditioned_features = _option_features(
        base_readout,
        "What fits the user best?",
        "(b) Producing music with software",
        conditioned_readout=conditioned_readout,
        conditioned_query="What fits the user best? producing music with software",
    )

    assert len(base_features) > 0
    assert len(conditioned_features) == len(base_features) * 2
    assert conditioned_features != base_features


def test_v65_persistent_memory_matches_on_facet_key_not_relation_only() -> None:
    base = Observation(
        obs_id="obs_base",
        source_dataset="demo",
        source_dialogue_id="dlg",
        source_turn_id="1",
        session_id="dlg",
        speaker="user",
        entity="user",
        relation="music_preference",
        value="latin music in software",
        value_type="preference",
        time_scope="current",
        status_hint="active",
        polarity="positive",
        confidence=0.8,
        evidence_text="I like making latin music in software.",
        canonical_gloss="music_preference=latin music in software",
        metadata={},
    )
    variant = Observation.from_dict(
        {
            **base.to_dict(),
            "obs_id": "obs_variant",
            "value": "software feels spontaneous and personal",
            "canonical_gloss": "music_preference=software feels spontaneous and personal",
        }
    )

    first_facet = next(facet for facet in facetize_observation(base) if facet.facet_type == "preference_target")
    second_facet = next(facet for facet in facetize_observation(variant) if facet.facet_type == "preference_mode")
    first = materialize_facet_observation(base, first_facet)
    second = materialize_facet_observation(variant, second_facet)

    assert first.canonical_gloss.startswith("preference_target=")
    assert second.canonical_gloss.startswith("preference_mode=")

    memory = PersistentCoreResidualMemory()
    memory.write(first, "new_residual", turn_index=1, obs_index=0)
    memory.write(second, "merge_residual", turn_index=2, obs_index=0)

    active = [slot for slot in memory.residual_bank if slot.active_flag]
    assert len(active) == 2
