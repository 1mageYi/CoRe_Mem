"""Mechanical verifier for v6.4 learned observation proposal / extraction coverage."""

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


def compute_v64_observation_proposal(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v64_plan.md")
    train = _read_first_json(
        artifact_root / "latest_stage2_v64_observation_proposal_train.json",
        artifact_root / "latest_stage2_v63_write_policy_train.json",
    )
    proposal_eval = _read_json(artifact_root / "latest_stage2_v64_observation_proposal_eval.json")
    state = _read_first_json(
        artifact_root / "latest_stage2_v64_persistent_state.json",
        artifact_root / "latest_stage2_v63_persistent_state.json",
    )
    internal_eval = _read_first_json(
        artifact_root / "latest_stage2_v64_internal_eval.json",
        artifact_root / "latest_stage2_v63_internal_eval.json",
    )
    personamem = _read_first_json(
        artifact_root / "latest_stage2_v64_personamem_no_routing.json",
        artifact_root / "latest_stage2_v63_personamem_no_routing.json",
    )
    decision = _read_first_json(
        artifact_root / "latest_stage2_v64_decision.json",
        artifact_root / "latest_stage2_v63_decision.json",
    )
    error_attribution = _read_first_json(
        artifact_root / "latest_stage2_v64_error_attribution.json",
        artifact_root / "latest_stage2_v63_error_attribution.json",
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

    never_written = _as_int(error_attribution, "never_written_count")
    parser_only_never_written = _as_int(error_attribution, "parser_only_never_written_count")
    never_written_reduction = parser_only_never_written - never_written

    proposal_recall = _as_float(proposal_eval, "proposal_recall")
    parser_only_recall = _as_float(proposal_eval, "parser_only_recall")
    support_recovery = _as_float(proposal_eval, "support_recovery")
    parser_only_support_recovery = _as_float(proposal_eval, "parser_only_support_recovery")

    checks: dict[str, bool] = {
        "plan_exists": "Learned Observation Proposal / Extraction Coverage" in plan_text
        and "hybrid candidate pool" in plan_text.lower()
        and "never_written" in plan_text,
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
        "learned_observation_proposer_authoritative": (
            _as_bool(proposal_eval, "authoritative_observation_proposer_in_candidate_path")
            or _contains_any(trained_modules, ("observation_proposer", "u ie", "uie", "gliner", "extractor"))
            or _as_bool(train, "observation_proposer_trained")
        ),
        "hybrid_candidate_pool": _as_bool(proposal_eval, "hybrid_candidate_pool_used")
        and _as_int(proposal_eval, "rule_candidate_count") > 0
        and _as_int(proposal_eval, "learned_candidate_count") > 0
        and _as_int(proposal_eval, "merged_candidate_count") >= max(
            _as_int(proposal_eval, "rule_candidate_count"),
            _as_int(proposal_eval, "learned_candidate_count"),
        ),
        "proposal_recall_positive": proposal_recall > parser_only_recall or support_recovery > parser_only_support_recovery,
        "never_written_reduced": parser_only_never_written > 0 and never_written_reduction > 0,
        "dedup_normalization_validation": (
            _as_bool(proposal_eval, "candidate_dedup_enabled")
            or _as_bool(proposal_eval, "candidate_normalization_enabled")
        )
        and (
            _as_bool(proposal_eval, "candidate_validation_enabled")
            or _as_int(proposal_eval, "deduped_candidate_count") > 0
        ),
        "support_coverage_positive": _as_float(proposal_eval, "support_recovery") > _as_float(proposal_eval, "parser_only_support_recovery")
        or (
            _as_float(proposal_eval, "write_recall") > _as_float(proposal_eval, "parser_only_write_recall")
        ),
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
        "no_personamem_gold_for_proposer_or_substrate": train is not None
        and personamem is not None
        and _as_bool(train, "uses_personamem_gold_for_proposer") is False
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
        "learned_write_time_routing": 4,
        "authoritative_learned_reader": 5,
        "authoritative_learned_decision": 5,
        "learned_observation_proposer_authoritative": 14,
        "hybrid_candidate_pool": 10,
        "proposal_recall_positive": 10,
        "never_written_reduced": 10,
        "dedup_normalization_validation": 6,
        "support_coverage_positive": 5,
        "state_not_collapsed": 4,
        "raw_context_retrieval_disabled": 2,
        "answer_time_routing_disabled": 2,
        "no_personamem_gold_for_proposer_or_substrate": 3,
        "personamem_full589": 1,
        "personamem_beats_text_only": 2,
        "personamem_beats_option_only": 3,
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
    if not checks["learned_observation_proposer_authoritative"]:
        caps.append({"reason": "learned_observation_proposer_missing", "cap": 60})
    if not checks["hybrid_candidate_pool"]:
        caps.append({"reason": "hybrid_candidate_pool_missing", "cap": 65})
    if not checks["proposal_recall_positive"]:
        caps.append({"reason": "proposal_recall_not_positive", "cap": 70})
    if not checks["never_written_reduced"]:
        caps.append({"reason": "never_written_not_reduced", "cap": 75})
    if not checks["dedup_normalization_validation"]:
        caps.append({"reason": "dedup_normalization_validation_missing", "cap": 80})
    if not checks["personamem_beats_text_only"]:
        caps.append({"reason": "personamem_no_cal_not_above_text_only", "cap": 85})
    if not checks["personamem_beats_option_only"]:
        caps.append({"reason": "personamem_no_cal_not_above_option_only", "cap": 90})
    if not checks["personamem_significant_margin"]:
        caps.append({"reason": "personamem_margin_not_meaningful", "cap": 95})

    score = min(raw_score, min((cap["cap"] for cap in caps), default=raw_score))
    return {
        "metric_name": "stage2_v64_observation_proposal_score",
        "score": score,
        "raw_score": raw_score,
        "checks": checks,
        "caps": caps,
        "current_truth": {
            "personamem_no_calibration_correct": no_cal_correct,
            "text_only_correct": text_only_correct,
            "option_only_correct": option_only_correct,
            "margin_vs_text_only": margin_text,
            "margin_vs_option_only": margin_option,
            "proposal_recall": proposal_recall,
            "parser_only_recall": parser_only_recall,
            "support_recovery": support_recovery,
            "parser_only_support_recovery": parser_only_support_recovery,
            "never_written_count": never_written,
            "parser_only_never_written_count": parser_only_never_written,
            "never_written_reduction": never_written_reduction,
        },
        "decision": "stop_ready" if score >= 100 else "needs_more_work",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_v64_observation_proposal(args.root)
    if args.score_only:
        print(payload["score"])
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
