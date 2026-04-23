"""Mechanical verifier for v6.3 recall-preserving confidence-aware write policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_first_json(*paths: Path) -> dict[str, Any] | None:
    for path in paths:
        payload = _read_json(path)
        if payload is not None:
            return payload
    return None


def _list(payload: dict[str, Any] | None, key: str) -> list[Any]:
    if not payload:
        return []
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _as_bool(payload: dict[str, Any] | None, key: str) -> bool:
    return bool(payload and payload.get(key) is True)


def _as_int(payload: dict[str, Any] | None, key: str) -> int:
    if not payload:
        return 0
    try:
        return int(payload.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _as_float(payload: dict[str, Any] | None, key: str) -> float:
    if not payload:
        return 0.0
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _exists_from_payload(root: Path, payload: dict[str, Any] | None, *keys: str) -> bool:
    if not payload:
        return False
    for key in keys:
        value = payload.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return True
    return False


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def compute_v63_write_policy(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v63_plan.md")
    train = _read_first_json(
        artifact_root / "latest_stage2_v63_write_policy_train.json",
        artifact_root / "latest_stage2_v62_write_quality_train.json",
    )
    policy_eval = _read_json(artifact_root / "latest_stage2_v63_write_policy_eval.json")
    state = _read_first_json(
        artifact_root / "latest_stage2_v63_persistent_state.json",
        artifact_root / "latest_stage2_v62_persistent_state.json",
    )
    internal_eval = _read_first_json(
        artifact_root / "latest_stage2_v63_internal_eval.json",
        artifact_root / "latest_stage2_v62_internal_eval.json",
    )
    personamem = _read_first_json(
        artifact_root / "latest_stage2_v63_personamem_no_routing.json",
        artifact_root / "latest_stage2_v62_personamem_no_routing.json",
    )
    decision = _read_first_json(
        artifact_root / "latest_stage2_v63_decision.json",
        artifact_root / "latest_stage2_v62_decision.json",
    )
    error_attribution = _read_json(artifact_root / "latest_stage2_v63_error_attribution.json")

    text_only_correct = _as_int(personamem, "text_only_correct")
    option_only_correct = _as_int(personamem, "option_only_correct")
    no_cal_correct = _as_int(personamem, "no_calibration_correct")
    margin_text = no_cal_correct - text_only_correct
    margin_option = no_cal_correct - option_only_correct
    significant_margin = bool(
        margin_text >= 30
        or _as_bool(personamem, "bootstrap_significant_vs_text_only")
        or _as_bool(decision, "significant_margin_confirmed")
    )

    trained_modules = " ".join(str(item).lower() for item in _list(train, "trained_modules"))
    score_mode = str(personamem.get("score_mode", "") if personamem else "").lower()
    decision_mode = str(personamem.get("decision_mode", "") if personamem else "").lower()
    reader_path = str(internal_eval.get("authoritative_reader_path", "") if internal_eval else "").lower()
    typed_residuals = _list(state, "typed_residual_relations")
    other_fact_share = _as_float(state, "other_fact_share")

    policy_labels = {str(item).lower() for item in _list(policy_eval, "policy_labels")}
    has_four_way_policy = {"core-worthy", "residual-worthy", "weak-but-keep", "drop"} <= policy_labels
    support_coverage_positive = (
        _as_float(policy_eval, "support_coverage_recall") > _as_float(policy_eval, "disabled_support_coverage_recall")
        or _as_float(policy_eval, "write_recall") > _as_float(policy_eval, "disabled_write_recall")
    )
    non_collapsed_state = (
        _as_int(state, "stream_observations_written") >= 96
        or (_as_int(state, "core_bank_size") + _as_int(state, "residual_bank_size")) >= 80
        or _as_float(policy_eval, "support_coverage_recall") >= 0.7
    )

    checks: dict[str, bool] = {
        "plan_exists": "Recall-Preserving Confidence-Aware Write Policy" in plan_text
        and "core-worthy" in plan_text
        and "weak-but-keep" in plan_text,
        "persistent_core_bank": _as_bool(state, "core_bank_exists") and _as_int(state, "core_bank_size") > 0,
        "persistent_residual_bank": _as_bool(state, "residual_bank_exists") and _as_int(state, "residual_bank_size") > 0,
        "persistent_state_checkpoint": _exists_from_payload(root, state, "state_checkpoint_path", "persistent_state_path"),
        "stream_write_trace": _exists_from_payload(root, state, "write_trace_path") and _as_int(state, "stream_observations_written") > 0,
        "learned_write_time_routing": _as_bool(train, "write_time_router_trained") and _as_bool(internal_eval, "write_time_router_applied_to_state"),
        "authoritative_learned_reader": (
            _as_bool(train, "query_conditioned_reader_trained")
            or _contains_any(trained_modules, ("query_conditioned_reader", "latent_reader"))
        )
        and (
            _as_bool(internal_eval, "reader_in_authoritative_path")
            or _contains_any(reader_path, ("learned", "reader", "cross_attention"))
        )
        and not _as_bool(internal_eval, "authoritative_reader_uses_dot_product_only"),
        "authoritative_learned_decision": (
            _as_bool(train, "decision_head_trained")
            or _contains_any(trained_modules, ("decision_head", "answer_head", "belief_readout"))
        )
        and (
            _contains_any(score_mode, ("learned", "decision_head", "answer_head"))
            or _contains_any(decision_mode, ("learned", "decision_head", "answer_head"))
        )
        and not _as_bool(personamem, "handcrafted_option_scoring_used")
        and not _as_bool(personamem, "lexical_jaccard_used"),
        "semantic_slot_matching": (
            _as_bool(train, "semantic_slot_matching_trained")
            or _as_bool(train, "slot_compaction_trained")
            or _as_bool(state, "bank_compaction_enabled")
        )
        and (_as_bool(state, "revision_chain_enabled") or _as_bool(state, "duplicate_relation_clusters_reduced")),
        "typed_residuals": (len(typed_residuals) >= 3 or _as_bool(internal_eval, "typed_residuals_used"))
        and (other_fact_share == 0.0 or other_fact_share < 0.5),
        "confidence_aware_four_way_policy": has_four_way_policy
        and (
            _as_bool(policy_eval, "confidence_aware_policy_in_authoritative_write_path")
            or _contains_any(trained_modules, ("write_policy", "weak_keep_policy", "confidence_aware_write"))
        ),
        "weak_but_keep_residual": _as_bool(policy_eval, "weak_but_keep_routed_to_residual")
        and _as_int(state, "weak_but_keep_residual_count") > 0,
        "support_coverage_positive": support_coverage_positive,
        "state_not_collapsed": non_collapsed_state,
        "error_attribution_exists": error_attribution is not None
        and all(
            key in error_attribution
            for key in (
                "never_written_count",
                "written_but_compacted_away_count",
                "written_but_reader_missed_count",
                "reader_selected_but_decision_failed_count",
            )
        ),
        "raw_context_retrieval_disabled": internal_eval is not None
        and personamem is not None
        and _as_bool(internal_eval, "raw_context_retrieval_disabled")
        and _as_bool(personamem, "raw_context_retrieval_disabled"),
        "answer_time_routing_disabled": personamem is not None
        and _as_bool(personamem, "answer_time_routing_used") is False
        and "route" not in score_mode,
        "no_personamem_gold_for_substrate": train is not None
        and personamem is not None
        and _as_bool(train, "uses_personamem_gold_for_substrate") is False
        and _as_bool(personamem, "gold_used_for_memory_substrate") is False,
        "personamem_full589": _as_int(personamem, "sample_count") == 589,
        "personamem_beats_text_only": margin_text > 0,
        "personamem_beats_option_only": margin_option > 0,
        "personamem_significant_margin": significant_margin,
        "decision_exists": bool(decision and decision.get("result_type") in {"positive_gain", "negative_result", "blocked"}),
    }

    weights = {
        "plan_exists": 4,
        "persistent_core_bank": 4,
        "persistent_residual_bank": 4,
        "persistent_state_checkpoint": 4,
        "stream_write_trace": 4,
        "learned_write_time_routing": 5,
        "authoritative_learned_reader": 6,
        "authoritative_learned_decision": 6,
        "semantic_slot_matching": 4,
        "typed_residuals": 4,
        "confidence_aware_four_way_policy": 14,
        "weak_but_keep_residual": 10,
        "support_coverage_positive": 8,
        "state_not_collapsed": 7,
        "error_attribution_exists": 5,
        "raw_context_retrieval_disabled": 3,
        "answer_time_routing_disabled": 3,
        "no_personamem_gold_for_substrate": 3,
        "personamem_full589": 1,
        "personamem_beats_text_only": 2,
        "personamem_beats_option_only": 2,
        "personamem_significant_margin": 1,
        "decision_exists": 1,
    }
    raw_score = sum(weight for name, weight in weights.items() if checks[name])

    caps: list[dict[str, Any]] = []
    if not (checks["persistent_core_bank"] and checks["persistent_residual_bank"]):
        caps.append({"reason": "missing_persistent_banks", "cap": 20})
    if not checks["persistent_state_checkpoint"]:
        caps.append({"reason": "missing_persistent_state_checkpoint", "cap": 25})
    if not checks["learned_write_time_routing"]:
        caps.append({"reason": "write_routing_not_connected", "cap": 35})
    if not checks["authoritative_learned_reader"]:
        caps.append({"reason": "learned_reader_not_in_authoritative_path", "cap": 45})
    if not checks["authoritative_learned_decision"]:
        caps.append({"reason": "learned_decision_not_in_authoritative_path", "cap": 50})
    if not checks["confidence_aware_four_way_policy"]:
        caps.append({"reason": "confidence_aware_four_way_policy_missing", "cap": 60})
    if not checks["weak_but_keep_residual"]:
        caps.append({"reason": "weak_but_keep_not_preserved_in_residual", "cap": 65})
    if not checks["support_coverage_positive"]:
        caps.append({"reason": "support_coverage_not_positive", "cap": 70})
    if not checks["state_not_collapsed"]:
        caps.append({"reason": "state_collapsed_after_write_policy", "cap": 75})
    if not checks["error_attribution_exists"]:
        caps.append({"reason": "error_attribution_missing", "cap": 80})
    if not checks["raw_context_retrieval_disabled"]:
        caps.append({"reason": "raw_context_retrieval_still_enabled", "cap": 55})
    if not checks["answer_time_routing_disabled"]:
        caps.append({"reason": "answer_time_routing_still_used", "cap": 45})
    if not checks["personamem_beats_text_only"]:
        caps.append({"reason": "personamem_not_above_text_only", "cap": 85})
    if not checks["personamem_beats_option_only"]:
        caps.append({"reason": "personamem_not_above_option_only", "cap": 90})
    if not checks["personamem_significant_margin"]:
        caps.append({"reason": "personamem_margin_not_meaningful", "cap": 95})
    if not checks["no_personamem_gold_for_substrate"]:
        caps.append({"reason": "personamem_gold_isolation_not_proven", "cap": 30})

    applied_cap = min((item["cap"] for item in caps), default=100)
    score = min(raw_score, applied_cap)
    return {
        "metric": "stage2_v63_write_policy_score",
        "score": score,
        "raw_score": raw_score,
        "total": 100,
        "checks": checks,
        "margin_correct_vs_text_only": margin_text,
        "margin_correct_vs_option_only": margin_option,
        "caps": caps,
        "applied_cap": applied_cap,
        "stop_ready": score >= 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v63_write_policy(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
