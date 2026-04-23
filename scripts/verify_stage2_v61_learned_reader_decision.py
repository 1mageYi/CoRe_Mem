"""Mechanical verifier for v6.1 learned reader/decision over persistent memory."""

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


def compute_v61_learned_reader_decision(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v61_plan.md")
    train = _read_first_json(
        artifact_root / "latest_stage2_v61_reader_decision_train.json",
        artifact_root / "latest_stage2_v6_persistent_memory_train.json",
    )
    state = _read_first_json(
        artifact_root / "latest_stage2_v61_persistent_state.json",
        artifact_root / "latest_stage2_v6_persistent_state.json",
    )
    internal_eval = _read_first_json(
        artifact_root / "latest_stage2_v61_internal_eval.json",
        artifact_root / "latest_stage2_v6_persistent_memory_eval.json",
    )
    personamem = _read_first_json(
        artifact_root / "latest_stage2_v61_personamem_no_routing.json",
        artifact_root / "latest_stage2_v6_personamem_no_routing.json",
    )
    ablation = _read_first_json(
        artifact_root / "latest_stage2_v61_arch_ablation.json",
        artifact_root / "latest_stage2_v6_arch_ablation.json",
    )
    decision = _read_first_json(
        artifact_root / "latest_stage2_v61_decision.json",
        artifact_root / "latest_stage2_v6_decision.json",
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
    hard_negative_source = str(internal_eval.get("hard_negative_source", "") if internal_eval else "")
    ablation_mode = str(ablation.get("ablation_mode", "") if ablation else "").lower()
    typed_residuals = _list(state, "typed_residual_relations")
    other_fact_share = _as_float(state, "other_fact_share")

    checks: dict[str, bool] = {
        "plan_exists": "Learned Reader/Decision over Persistent Memory" in plan_text
        and "Do not go back to answer-time routing" in plan_text,
        "persistent_core_bank": _as_bool(state, "core_bank_exists") and _as_int(state, "core_bank_size") > 0,
        "persistent_residual_bank": _as_bool(state, "residual_bank_exists") and _as_int(state, "residual_bank_size") > 0,
        "persistent_state_checkpoint": _exists_from_payload(root, state, "state_checkpoint_path", "persistent_state_path"),
        "stream_write_trace": _exists_from_payload(root, state, "write_trace_path") and _as_int(state, "stream_observations_written") > 0,
        "learned_write_time_routing": _as_bool(train, "write_time_router_trained") and _as_bool(internal_eval, "write_time_router_applied_to_state"),
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
        "authoritative_learned_reader": (
            _as_bool(train, "query_conditioned_reader_trained")
            or "query_conditioned_reader" in trained_modules
            or "latent_reader" in trained_modules
        )
        and (
            _as_bool(internal_eval, "reader_in_authoritative_path")
            or _contains_any(reader_path, ("learned", "reader", "cross_attention"))
        )
        and not _as_bool(internal_eval, "authoritative_reader_uses_dot_product_only"),
        "authoritative_learned_decision": (
            _as_bool(train, "decision_head_trained")
            or _as_bool(train, "answer_head_trained")
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
        and (
            _as_bool(state, "revision_chain_enabled")
            or _as_bool(state, "duplicate_relation_clusters_reduced")
            or _as_float(state, "dedup_ratio") > 0.0
        ),
        "typed_residuals": (len(typed_residuals) >= 3 or _as_bool(internal_eval, "typed_residuals_used"))
        and (other_fact_share == 0.0 or other_fact_share < 0.5),
        "hard_internal_eval": _as_bool(internal_eval, "natural_language_queries")
        and not _as_bool(internal_eval, "gold_value_echo_queries")
        and (bool(hard_negative_source) or _as_int(internal_eval, "hard_negative_count") > 0),
        "true_architecture_ablations": "true" in ablation_mode or "disabled_architecture" in ablation_mode,
        "ablation_drops": all(
            _as_bool(ablation, key)
            for key in (
                "disabled_writer_drops",
                "disabled_reader_drops",
                "disabled_decision_drops",
                "disabled_core_residual_split_drops",
            )
        ),
        "personamem_full589": _as_int(personamem, "sample_count") == 589,
        "personamem_beats_text_only": margin_text > 0,
        "personamem_beats_option_only": margin_option > 0,
        "personamem_significant_margin": significant_margin,
        "decision_exists": bool(decision and decision.get("result_type") in {"positive_gain", "negative_result", "blocked"}),
    }

    weights = {
        "plan_exists": 4,
        "persistent_core_bank": 5,
        "persistent_residual_bank": 5,
        "persistent_state_checkpoint": 5,
        "stream_write_trace": 5,
        "learned_write_time_routing": 8,
        "raw_context_retrieval_disabled": 6,
        "answer_time_routing_disabled": 6,
        "no_personamem_gold_for_substrate": 5,
        "authoritative_learned_reader": 12,
        "authoritative_learned_decision": 12,
        "semantic_slot_matching": 7,
        "typed_residuals": 6,
        "hard_internal_eval": 8,
        "true_architecture_ablations": 4,
        "ablation_drops": 4,
        "personamem_full589": 2,
        "personamem_beats_text_only": 3,
        "personamem_beats_option_only": 2,
        "personamem_significant_margin": 5,
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
    if not checks["raw_context_retrieval_disabled"]:
        caps.append({"reason": "raw_context_retrieval_still_enabled", "cap": 55})
    if not checks["answer_time_routing_disabled"]:
        caps.append({"reason": "answer_time_routing_still_used", "cap": 45})
    if not checks["hard_internal_eval"]:
        caps.append({"reason": "internal_eval_not_hard_or_aligned", "cap": 60})
    if not checks["semantic_slot_matching"]:
        caps.append({"reason": "semantic_slot_matching_not_proven", "cap": 70})
    if not checks["typed_residuals"]:
        caps.append({"reason": "typed_residuals_not_proven", "cap": 75})
    if not checks["personamem_beats_text_only"]:
        caps.append({"reason": "personamem_not_above_text_only", "cap": 80})
    if not checks["personamem_beats_option_only"]:
        caps.append({"reason": "personamem_not_above_option_only", "cap": 85})
    if not checks["personamem_significant_margin"]:
        caps.append({"reason": "personamem_margin_not_meaningful", "cap": 90})
    if not checks["no_personamem_gold_for_substrate"]:
        caps.append({"reason": "personamem_gold_isolation_not_proven", "cap": 30})

    applied_cap = min((item["cap"] for item in caps), default=100)
    score = min(raw_score, applied_cap)
    return {
        "metric": "stage2_v61_learned_reader_decision_score",
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
    payload = compute_v61_learned_reader_decision(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
