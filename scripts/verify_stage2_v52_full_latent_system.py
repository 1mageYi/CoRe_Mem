"""Mechanical verifier for v5.2 full learned latent memory system.

v5.1 proved a real-training bootstrap. v5.2 requires the stronger claim:
multi-backbone, multi-task, trainable latent reader/controller/belief modules,
and PersonaMem no-calibration above the text-only baseline.
"""

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


def _list(payload: dict[str, Any] | None, key: str) -> list[Any]:
    if not payload:
        return []
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _checkpoint_exists(root: Path, payload: dict[str, Any] | None) -> bool:
    if not payload or not payload.get("checkpoint_path"):
        return False
    path = Path(str(payload["checkpoint_path"]))
    if not path.is_absolute():
        path = root / path
    return path.exists()


def _metric(payload: dict[str, Any] | None, *keys: str) -> float:
    if not payload:
        return 0.0
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_module_group(training: dict[str, Any] | None, names: tuple[str, ...]) -> bool:
    modules = " ".join(str(item).lower() for item in _list(training, "trained_modules"))
    return any(name in modules for name in names)


def compute_v52_full_latent_system(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v52_plan.md")
    backbone = _read_json(artifact_root / "latest_stage2_v52_backbone_compare.json")
    training = _read_json(artifact_root / "latest_stage2_v52_multitask_training.json")
    eval_payload = _read_json(artifact_root / "latest_stage2_v52_full_latent_eval.json")
    personamem = _read_json(artifact_root / "latest_stage2_v52_personamem_full589.json")
    ablation = _read_json(artifact_root / "latest_stage2_v52_ablation_summary.json")
    decision = _read_json(artifact_root / "latest_stage2_v52_decision.json")

    loaded_backbones = _list(backbone, "loaded_backbones")
    tasks = set(str(item) for item in _list(training, "trained_tasks"))
    required_tasks = {
        "retrieval_alignment",
        "slot_autoencoding",
        "composition_to_belief",
        "lifecycle_prediction",
    }
    text_only_correct = _as_int(personamem, "text_only_correct")
    no_cal_correct = _as_int(personamem, "no_calibration_correct")
    text_only_accuracy = _as_float(personamem, "text_only_accuracy")
    no_cal_accuracy = _as_float(personamem, "no_calibration_accuracy")

    backend_text = str(backbone.get("evaluation_backend", "") if backbone else "").lower()

    checks: dict[str, bool] = {
        "plan_exists": "Full Learned Latent Memory System" in plan_text and "text-only" in plan_text,
        "at_least_two_real_backbones": len(loaded_backbones) >= 2
        and _as_bool(backbone, "pretrained_weights_loaded"),
        "three_backbone_target_attempted": len(_list(backbone, "attempted_backbones")) >= 3,
        "real_backend": bool(backend_text) and "proxy" not in backend_text,
        "real_checkpoint_exists": _checkpoint_exists(root, training),
        "train_samples_20k": _as_int(training, "train_samples") >= 20000,
        "multi_task_training": required_tasks.issubset(tasks),
        "not_retrieval_only": not _as_bool(training, "retrieval_only") and len(tasks) >= 4,
        "encoder_or_projection_trainable": _has_module_group(
            training, ("encoder_adapter", "encoder adapter", "projection", "lora")
        ),
        "latent_reader_trainable": _has_module_group(
            training, ("latent_reader", "latent reader", "resampler")
        ),
        "write_controller_trainable": _has_module_group(
            training, ("write_controller", "write controller", "lifecycle", "controller")
        ),
        "belief_module_trainable": _has_module_group(
            training, ("belief_decoder", "belief decoder", "belief_graph", "belief graph")
        ),
        "uses_no_personamem_gold_for_training": bool(
            training and training.get("uses_personamem_gold") is False
        ),
        "trained_beats_frozen": _as_bool(eval_payload, "trained_beats_frozen")
        or _metric(eval_payload, "trained_metrics", "mrr") > _metric(eval_payload, "frozen_metrics", "mrr"),
        "multitask_beats_retrieval_only": _as_bool(eval_payload, "multitask_beats_retrieval_only")
        or _as_float(eval_payload, "multitask_latent") > _as_float(eval_payload, "retrieval_only_latent"),
        "latent_only_beats_shuffled": _as_float(eval_payload, "latent_only")
        > _as_float(eval_payload, "shuffled_latent"),
        "full_beats_text_only": _as_float(eval_payload, "full") > _as_float(eval_payload, "text_only"),
        "ablation_drops": all(
            _as_bool(ablation, key)
            for key in ("no_controller_drops", "no_belief_drops", "no_core_residual_drops")
        ),
        "personamem_full589": _as_int(personamem, "sample_count") == 589,
        "personamem_no_cal_beats_text_only": bool(
            no_cal_correct > max(text_only_correct, 214)
            or (no_cal_accuracy > text_only_accuracy and no_cal_accuracy > 214 / 589)
        ),
        "personamem_gold_not_used_for_substrate": bool(
            personamem and personamem.get("gold_used_for_memory_substrate") is False
        ),
        "decision_exists": bool(
            decision
            and decision.get("result_type") in {"positive_gain", "negative_result", "blocked"}
        ),
    }

    weights = {
        "plan_exists": 4,
        "at_least_two_real_backbones": 9,
        "three_backbone_target_attempted": 3,
        "real_backend": 4,
        "real_checkpoint_exists": 8,
        "train_samples_20k": 6,
        "multi_task_training": 10,
        "not_retrieval_only": 6,
        "encoder_or_projection_trainable": 5,
        "latent_reader_trainable": 6,
        "write_controller_trainable": 6,
        "belief_module_trainable": 7,
        "uses_no_personamem_gold_for_training": 4,
        "trained_beats_frozen": 5,
        "multitask_beats_retrieval_only": 5,
        "latent_only_beats_shuffled": 4,
        "full_beats_text_only": 4,
        "ablation_drops": 5,
        "personamem_full589": 3,
        "personamem_no_cal_beats_text_only": 4,
        "personamem_gold_not_used_for_substrate": 2,
        "decision_exists": 2,
    }
    raw_score = sum(weight for name, weight in weights.items() if checks[name])

    caps: list[dict[str, Any]] = []
    if not checks["at_least_two_real_backbones"]:
        caps.append({"reason": "less_than_two_real_backbones", "cap": 20})
    if not checks["real_checkpoint_exists"]:
        caps.append({"reason": "missing_real_checkpoint", "cap": 30})
    if not checks["multi_task_training"] or not checks["not_retrieval_only"]:
        caps.append({"reason": "not_full_multitask_training", "cap": 45})
    if not (
        checks["encoder_or_projection_trainable"]
        and checks["latent_reader_trainable"]
        and checks["write_controller_trainable"]
        and checks["belief_module_trainable"]
    ):
        caps.append({"reason": "missing_required_trainable_module_groups", "cap": 55})
    if not checks["multitask_beats_retrieval_only"]:
        caps.append({"reason": "multitask_not_above_retrieval_only", "cap": 65})
    if not checks["trained_beats_frozen"]:
        caps.append({"reason": "trained_not_above_frozen", "cap": 70})
    if not checks["full_beats_text_only"]:
        caps.append({"reason": "full_not_above_text_only", "cap": 75})
    if not checks["personamem_no_cal_beats_text_only"]:
        caps.append({"reason": "personamem_no_cal_not_above_text_only", "cap": 80})
    if not (
        checks["uses_no_personamem_gold_for_training"]
        and checks["personamem_gold_not_used_for_substrate"]
    ):
        caps.append({"reason": "gold_isolation_not_proven", "cap": 30})

    applied_cap = min((cap["cap"] for cap in caps), default=100)
    score = min(raw_score, applied_cap)
    return {
        "metric": "stage2_v52_full_latent_system_score",
        "score": score,
        "raw_score": raw_score,
        "total": 100,
        "checks": checks,
        "caps": caps,
        "applied_cap": applied_cap,
        "stop_ready": score >= 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_v52_full_latent_system(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
