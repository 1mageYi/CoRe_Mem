"""Mechanical verifier for v6 persistent core-residual latent memory.

v6 intentionally raises the bar beyond v5.2. A retained result must show a
persistent learned memory state and must not close out through answer-time
text-vs-latent confidence routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "outputs_v2" / "artifacts"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _list(payload: dict[str, Any] | None, key: str) -> list[Any]:
    if not payload:
        return []
    value = payload.get(key)
    return value if isinstance(value, list) else []


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


def _as_bool(payload: dict[str, Any] | None, key: str) -> bool:
    return bool(payload and payload.get(key) is True)


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


def _has_action(payload: dict[str, Any] | None, *actions: str) -> bool:
    seen = {str(item) for item in _list(payload, "observed_update_actions")}
    seen.update(str(item) for item in _list(payload, "learned_update_actions"))
    return all(action in seen for action in actions)


def compute_v6_persistent_latent_memory(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v6_plan.md")
    train = _read_json(artifact_root / "latest_stage2_v6_persistent_memory_train.json")
    state = _read_json(artifact_root / "latest_stage2_v6_persistent_state.json")
    eval_payload = _read_json(artifact_root / "latest_stage2_v6_persistent_memory_eval.json")
    personamem = _read_json(artifact_root / "latest_stage2_v6_personamem_no_routing.json")
    ablation = _read_json(artifact_root / "latest_stage2_v6_arch_ablation.json")
    decision = _read_json(artifact_root / "latest_stage2_v6_decision.json")

    text_only_correct = _as_int(personamem, "text_only_correct")
    no_cal_correct = _as_int(personamem, "no_calibration_correct")
    margin_correct = no_cal_correct - text_only_correct
    significant_margin = bool(
        margin_correct >= 30
        or _as_bool(personamem, "bootstrap_significant_vs_text_only")
        or _as_bool(decision, "significant_margin_confirmed")
    )

    ablation_mode = str(ablation.get("ablation_mode", "") if ablation else "").lower()
    score_mode = str(personamem.get("score_mode", "") if personamem else "").lower()
    trained_modules = " ".join(str(item).lower() for item in _list(train, "trained_modules"))
    eval_consumes = str(eval_payload.get("authoritative_input", "") if eval_payload else "").lower()

    checks: dict[str, bool] = {
        "plan_exists": "Persistent Core-Residual Latent Memory" in plan_text
        and "answer-time confidence routing is disabled" in plan_text,
        "persistent_core_bank": _as_bool(state, "core_bank_exists")
        and _as_int(state, "core_bank_size") > 0,
        "persistent_residual_bank": _as_bool(state, "residual_bank_exists")
        and _as_int(state, "residual_bank_size") > 0,
        "persistent_state_checkpoint": _exists_from_payload(
            root,
            state,
            "state_checkpoint_path",
            "persistent_state_path",
        ),
        "stream_write_trace": _exists_from_payload(root, state, "write_trace_path")
        and _as_int(state, "stream_observations_written") > 0,
        "learned_write_time_routing": _as_bool(train, "write_time_router_trained")
        and _as_bool(eval_payload, "write_time_router_applied_to_state")
        and _has_action(train, "new_core", "new_residual", "overwrite", "promote_to_core"),
        "trainable_latent_reader": "latent_reader" in trained_modules
        or "query_conditioned_reader" in trained_modules,
        "trainable_belief_or_readout": "belief" in trained_modules
        and ("readout" in trained_modules or "answer" in trained_modules),
        "authoritative_uses_persistent_state": _as_bool(eval_payload, "evaluation_consumes_persistent_state")
        and "persistent" in eval_consumes,
        "raw_context_retrieval_disabled": _as_bool(eval_payload, "raw_context_retrieval_disabled")
        and _as_bool(personamem, "raw_context_retrieval_disabled"),
        "answer_time_routing_disabled": _as_bool(personamem, "answer_time_routing_used") is False
        and "confidence_routed" not in score_mode
        and "route" not in score_mode,
        "no_personamem_gold_for_substrate": _as_bool(train, "uses_personamem_gold_for_substrate") is False
        and _as_bool(personamem, "gold_used_for_memory_substrate") is False,
        "latent_beats_shuffled_bank": _as_float(eval_payload, "latent_only_accuracy")
        > _as_float(eval_payload, "shuffled_bank_accuracy"),
        "full_beats_text_only_internal": _as_float(eval_payload, "full_accuracy")
        > _as_float(eval_payload, "text_only_accuracy"),
        "core_residual_both_useful": _as_float(eval_payload, "core_only_accuracy") > 0.0
        and _as_float(eval_payload, "residual_only_accuracy") > 0.0,
        "true_architecture_ablations": "true" in ablation_mode
        or "disabled_architecture" in ablation_mode,
        "ablation_drops": all(
            _as_bool(ablation, key)
            for key in (
                "disabled_writer_drops",
                "disabled_belief_drops",
                "disabled_core_residual_split_drops",
                "shuffled_persona_drops",
            )
        ),
        "personamem_full589": _as_int(personamem, "sample_count") == 589,
        "personamem_no_cal_beats_text_only": margin_correct > 0,
        "personamem_significant_margin": significant_margin,
        "decision_exists": bool(decision and decision.get("result_type") in {"positive_gain", "negative_result", "blocked"}),
    }

    weights = {
        "plan_exists": 4,
        "persistent_core_bank": 7,
        "persistent_residual_bank": 7,
        "persistent_state_checkpoint": 7,
        "stream_write_trace": 5,
        "learned_write_time_routing": 10,
        "trainable_latent_reader": 5,
        "trainable_belief_or_readout": 6,
        "authoritative_uses_persistent_state": 7,
        "raw_context_retrieval_disabled": 7,
        "answer_time_routing_disabled": 7,
        "no_personamem_gold_for_substrate": 5,
        "latent_beats_shuffled_bank": 5,
        "full_beats_text_only_internal": 4,
        "core_residual_both_useful": 4,
        "true_architecture_ablations": 6,
        "ablation_drops": 6,
        "personamem_full589": 3,
        "personamem_no_cal_beats_text_only": 3,
        "personamem_significant_margin": 4,
        "decision_exists": 1,
    }
    raw_score = sum(weight for name, weight in weights.items() if checks[name])

    caps: list[dict[str, Any]] = []
    if not (checks["persistent_core_bank"] and checks["persistent_residual_bank"]):
        caps.append({"reason": "missing_persistent_core_or_residual_bank", "cap": 20})
    if not checks["persistent_state_checkpoint"]:
        caps.append({"reason": "missing_persistent_state_checkpoint", "cap": 30})
    if not checks["learned_write_time_routing"]:
        caps.append({"reason": "write_time_routing_not_connected_to_state", "cap": 45})
    if not checks["authoritative_uses_persistent_state"]:
        caps.append({"reason": "authoritative_eval_not_using_persistent_state", "cap": 55})
    if not checks["raw_context_retrieval_disabled"]:
        caps.append({"reason": "raw_context_retrieval_still_enabled", "cap": 60})
    if not checks["answer_time_routing_disabled"]:
        caps.append({"reason": "answer_time_confidence_routing_still_used", "cap": 40})
    if not checks["true_architecture_ablations"]:
        caps.append({"reason": "ablations_are_not_true_disabled_architecture_reruns", "cap": 70})
    if not checks["ablation_drops"]:
        caps.append({"reason": "architecture_ablations_do_not_drop", "cap": 75})
    if not checks["personamem_no_cal_beats_text_only"]:
        caps.append({"reason": "personamem_no_cal_not_above_text_only", "cap": 80})
    if not checks["personamem_significant_margin"]:
        caps.append({"reason": "personamem_margin_not_meaningful", "cap": 85})
    if not checks["no_personamem_gold_for_substrate"]:
        caps.append({"reason": "personamem_gold_isolation_not_proven", "cap": 30})

    applied_cap = min((cap["cap"] for cap in caps), default=100)
    score = min(raw_score, applied_cap)
    return {
        "metric": "stage2_v6_persistent_latent_memory_score",
        "score": score,
        "raw_score": raw_score,
        "total": 100,
        "checks": checks,
        "margin_correct_vs_text_only": margin_correct,
        "caps": caps,
        "applied_cap": applied_cap,
        "stop_ready": score >= 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v6_persistent_latent_memory(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
