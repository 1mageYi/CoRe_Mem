"""Mechanical verifier for v6.5 facetized observation-to-memory redesign."""

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


def compute_v65_facetized_memory(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v65_plan.md")
    train = _read_first_json(
        artifact_root / "latest_stage2_v65_facetized_memory_train.json",
        artifact_root / "latest_stage2_v64_observation_proposal_train.json",
    )
    facet_eval = _read_json(artifact_root / "latest_stage2_v65_facetized_memory_eval.json")
    state = _read_first_json(
        artifact_root / "latest_stage2_v65_persistent_state.json",
        artifact_root / "latest_stage2_v64_persistent_state.json",
    )
    internal_eval = _read_first_json(
        artifact_root / "latest_stage2_v65_internal_eval.json",
        artifact_root / "latest_stage2_v64_internal_eval.json",
    )
    personamem = _read_first_json(
        artifact_root / "latest_stage2_v65_personamem_no_routing.json",
        artifact_root / "latest_stage2_v64_personamem_no_routing.json",
    )
    decision = _read_first_json(
        artifact_root / "latest_stage2_v65_decision.json",
        artifact_root / "latest_stage2_v64_decision.json",
    )
    error_attribution = _read_first_json(
        artifact_root / "latest_stage2_v65_error_attribution.json",
        artifact_root / "latest_stage2_v64_error_attribution.json",
    )

    trained_modules = " ".join(str(item).lower() for item in _list(train, "trained_modules"))
    score_mode = str(personamem.get("score_mode", "") if personamem else "").lower()
    decision_mode = str(personamem.get("decision_mode", "") if personamem else "").lower()
    reader_path = str(internal_eval.get("authoritative_reader_path", "") if internal_eval else "").lower()

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

    needed_facet_missing = _as_int(error_attribution, "needed_facet_missing_count")
    parser_only_needed_facet_missing = _as_int(error_attribution, "parser_only_needed_facet_missing_count")
    wrong_sibling_facet_selected = _as_int(error_attribution, "wrong_sibling_facet_selected_count")
    parser_only_wrong_sibling = _as_int(error_attribution, "parser_only_wrong_sibling_facet_selected_count")

    checks: dict[str, bool] = {
        "plan_exists": "Facetized Observation-to-Memory Redesign" in plan_text
        and "facetizer" in plan_text.lower()
        and "needed_facet_missing" in plan_text,
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
        "facet_schema_defined": _as_bool(facet_eval, "facet_schema_defined")
        or _as_bool(train, "facet_schema_defined")
        or len(_list(facet_eval, "facet_types")) >= 3,
        "facetizer_authoritative": _as_bool(facet_eval, "authoritative_facetizer_in_path")
        or _as_bool(train, "facetizer_trained")
        or _contains_any(trained_modules, ("facetizer", "facet_encoder", "facet_memory")),
        "facet_aware_write_utility": _as_bool(facet_eval, "facet_aware_write_utility_used")
        or _as_bool(internal_eval, "facet_aware_write_utility_used")
        or _contains_any(trained_modules, ("facet_write_utility", "facet_aware_write")),
        "facet_level_error_attribution": error_attribution is not None
        and any(
            key in error_attribution
            for key in (
                "needed_facet_missing_count",
                "wrong_sibling_facet_selected_count",
                "facet_written_but_reader_missed_count",
                "facet_selected_but_decision_failed_count",
            )
        ),
        "needed_facet_reduced": parser_only_needed_facet_missing > 0
        and needed_facet_missing < parser_only_needed_facet_missing,
        "same_relation_confusion_reduced": parser_only_wrong_sibling > 0
        and wrong_sibling_facet_selected < parser_only_wrong_sibling,
        "state_not_collapsed": (
            _as_int(state, "stream_observations_written") >= 96
            or (_as_int(state, "core_bank_size") + _as_int(state, "residual_bank_size")) >= 80
        ),
        "raw_context_retrieval_disabled": internal_eval is not None
        and personamem is not None
        and _as_bool(internal_eval, "raw_context_retrieval_disabled")
        and _as_bool(personamem, "raw_context_retrieval_disabled"),
        "answer_time_routing_disabled": personamem is not None
        and _as_bool(personamem, "answer_time_routing_used") is False
        and "route" not in score_mode,
        "no_personamem_gold_for_facetizer_or_substrate": train is not None
        and personamem is not None
        and _as_bool(train, "uses_personamem_gold_for_facetizer") is False
        and _as_bool(train, "uses_personamem_gold_for_substrate") is False
        and _as_bool(personamem, "gold_used_for_memory_substrate") is False,
        "personamem_full589": _as_int(personamem, "sample_count") == 589,
        "personamem_beats_text_only": margin_text > 0,
        "personamem_beats_option_only": margin_option > 0,
        "personamem_significant_margin": significant_margin,
    }

    weights = {
        "plan_exists": 4,
        "persistent_core_bank": 4,
        "persistent_residual_bank": 4,
        "persistent_state_checkpoint": 4,
        "stream_write_trace": 4,
        "learned_write_time_routing": 4,
        "authoritative_learned_reader": 4,
        "authoritative_learned_decision": 4,
        "facet_schema_defined": 12,
        "facetizer_authoritative": 12,
        "facet_aware_write_utility": 10,
        "facet_level_error_attribution": 8,
        "needed_facet_reduced": 8,
        "same_relation_confusion_reduced": 4,
        "state_not_collapsed": 3,
        "raw_context_retrieval_disabled": 2,
        "answer_time_routing_disabled": 2,
        "no_personamem_gold_for_facetizer_or_substrate": 3,
        "personamem_full589": 1,
        "personamem_beats_text_only": 1,
        "personamem_beats_option_only": 1,
        "personamem_significant_margin": 1,
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
        caps.append({"reason": "learned_reader_not_in_authoritative_path", "cap": 40})
    if not checks["authoritative_learned_decision"]:
        caps.append({"reason": "learned_decision_not_in_authoritative_path", "cap": 45})
    if not checks["facet_schema_defined"]:
        caps.append({"reason": "facet_schema_missing", "cap": 55})
    if not checks["facetizer_authoritative"]:
        caps.append({"reason": "facetizer_missing", "cap": 60})
    if not checks["facet_aware_write_utility"]:
        caps.append({"reason": "facet_aware_write_utility_missing", "cap": 70})
    if not checks["facet_level_error_attribution"]:
        caps.append({"reason": "facet_level_error_attribution_missing", "cap": 75})
    if not checks["needed_facet_reduced"]:
        caps.append({"reason": "needed_facet_not_reduced", "cap": 85})
    if not checks["personamem_beats_text_only"]:
        caps.append({"reason": "personamem_no_cal_not_above_text_only", "cap": 95})
    if not checks["personamem_beats_option_only"]:
        caps.append({"reason": "personamem_no_cal_not_above_option_only", "cap": 98})
    if not checks["personamem_significant_margin"]:
        caps.append({"reason": "personamem_margin_not_meaningful", "cap": 99})

    score = min([raw_score, *[cap["cap"] for cap in caps]]) if caps else raw_score
    stop_ready = score >= 100

    return {
        "metric_name": "stage2_v65_facetized_memory_score",
        "score": score,
        "raw_score": raw_score,
        "checks": checks,
        "caps": caps,
        "stop_ready": stop_ready,
        "required_stop_labels": [
            "persistent-bank",
            "write-routing",
            "learned-reader-authoritative",
            "learned-decision-authoritative",
            "facet-schema",
            "facetizer-authoritative",
            "facet-aware-write-utility",
            "facet-level-error-attribution",
            "needed-facet-reduced",
            "no-answer-routing",
            "no-gold-leakage",
            "significant-personamem-margin",
        ],
        "summary": {
            "personamem_no_calibration_correct": no_cal_correct,
            "text_only_correct": text_only_correct,
            "option_only_correct": option_only_correct,
            "margin_vs_text_only": margin_text,
            "margin_vs_option_only": margin_option,
            "needed_facet_missing_count": needed_facet_missing,
            "parser_only_needed_facet_missing_count": parser_only_needed_facet_missing,
            "wrong_sibling_facet_selected_count": wrong_sibling_facet_selected,
            "parser_only_wrong_sibling_facet_selected_count": parser_only_wrong_sibling,
            "core_bank_size": _as_int(state, "core_bank_size"),
            "residual_bank_size": _as_int(state, "residual_bank_size"),
            "stream_observations_written": _as_int(state, "stream_observations_written"),
            "facet_types": _list(facet_eval, "facet_types"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify v6.5 facetized observation-to-memory redesign.")
    parser.add_argument("--score-only", action="store_true", help="Print only the numeric score.")
    args = parser.parse_args()

    payload = compute_v65_facetized_memory()
    if args.score_only:
        print(payload["score"])
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
