"""Mechanical verifier for v6.2 learned write-worthiness / attribute-validity."""

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


def compute_v62_write_quality(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v62_plan.md")
    train = _read_first_json(
        artifact_root / "latest_stage2_v62_write_quality_train.json",
        artifact_root / "latest_stage2_v61_reader_decision_train.json",
    )
    write_eval = _read_json(artifact_root / "latest_stage2_v62_write_quality_eval.json")
    state = _read_first_json(
        artifact_root / "latest_stage2_v62_persistent_state.json",
        artifact_root / "latest_stage2_v61_persistent_state.json",
    )
    internal_eval = _read_first_json(
        artifact_root / "latest_stage2_v62_internal_eval.json",
        artifact_root / "latest_stage2_v61_internal_eval.json",
    )
    personamem = _read_first_json(
        artifact_root / "latest_stage2_v62_personamem_no_routing.json",
        artifact_root / "latest_stage2_v61_personamem_no_routing.json",
    )
    decision = _read_first_json(
        artifact_root / "latest_stage2_v62_decision.json",
        artifact_root / "latest_stage2_v61_decision.json",
    )
    ablation = _read_first_json(
        artifact_root / "latest_stage2_v62_arch_ablation.json",
        artifact_root / "latest_stage2_v61_arch_ablation.json",
    )

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
    ablation_mode = str(ablation.get("ablation_mode", "") if ablation else "").lower()

    write_worthiness_trained = (
        _as_bool(train, "write_worthiness_trained")
        or _as_bool(train, "fact_validity_trained")
        or _contains_any(trained_modules, ("write_worthiness", "fact_validity", "write_quality"))
    )
    attribute_validity_trained = (
        _as_bool(train, "attribute_validity_trained")
        or _as_bool(train, "relation_validity_trained")
        or _contains_any(trained_modules, ("attribute_validity", "relation_validity", "extraction_validity"))
    )
    write_quality_positive = (
        _as_float(write_eval, "write_worthiness_eval_accuracy") > _as_float(write_eval, "disabled_write_worthiness_accuracy")
        or _as_float(write_eval, "write_precision_at_1") > _as_float(write_eval, "disabled_write_precision_at_1")
        or _as_float(write_eval, "attribute_validity_f1") > _as_float(write_eval, "disabled_attribute_validity_f1")
    )
    clean_state_positive = (
        _as_float(state, "bank_precision_estimate") >= 0.7
        or _as_float(state, "invalid_slot_rate") <= 0.15
        or _as_float(state, "low_information_slot_share") <= 0.1
        or _as_float(state, "state_noise_rate") <= 0.15
    )

    checks: dict[str, bool] = {
        "plan_exists": "Learned Write-Worthiness / Attribute-Validity Before Extraction" in plan_text
        and "Do not go back to answer-time routing" in plan_text,
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
        "learned_write_worthiness_authoritative": write_worthiness_trained
        and _as_bool(write_eval, "write_worthiness_in_authoritative_write_path"),
        "attribute_validity_prewrite": attribute_validity_trained
        and (
            _as_bool(write_eval, "attribute_validity_before_write")
            or _as_bool(write_eval, "relation_validity_before_write")
        ),
        "hard_write_quality_eval": _as_bool(write_eval, "uses_raw_dialogue_observations")
        and _as_bool(write_eval, "hard_negatives_enabled")
        and (
            _as_int(write_eval, "hard_negative_count") > 0
            or _as_int(write_eval, "conflict_negative_count") > 0
        )
        and write_quality_positive,
        "clean_persistent_state": clean_state_positive,
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
        "true_architecture_ablations": "true" in ablation_mode or "disabled_architecture" in ablation_mode,
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
        "authoritative_learned_reader": 7,
        "authoritative_learned_decision": 7,
        "semantic_slot_matching": 4,
        "typed_residuals": 4,
        "learned_write_worthiness_authoritative": 12,
        "attribute_validity_prewrite": 12,
        "hard_write_quality_eval": 8,
        "clean_persistent_state": 5,
        "raw_context_retrieval_disabled": 4,
        "answer_time_routing_disabled": 4,
        "no_personamem_gold_for_substrate": 4,
        "true_architecture_ablations": 4,
        "personamem_full589": 1,
        "personamem_beats_text_only": 1,
        "personamem_beats_option_only": 1,
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
    if not checks["learned_write_worthiness_authoritative"]:
        caps.append({"reason": "write_worthiness_not_in_authoritative_write_path", "cap": 60})
    if not checks["attribute_validity_prewrite"]:
        caps.append({"reason": "attribute_validity_not_in_prewrite_path", "cap": 65})
    if not checks["hard_write_quality_eval"]:
        caps.append({"reason": "write_quality_eval_not_hard_or_positive", "cap": 70})
    if not checks["clean_persistent_state"]:
        caps.append({"reason": "persistent_state_not_clean_enough", "cap": 75})
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
        "metric": "stage2_v62_write_quality_score",
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
    payload = compute_v62_write_quality(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
