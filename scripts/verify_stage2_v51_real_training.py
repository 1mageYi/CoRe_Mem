"""Mechanical verifier for v5.1 real pretrained training evidence.

This intentionally treats v5 proxy artifacts as scaffold evidence only. The
score can rise only when real pretrained weights, real checkpoints, held-out
evals, and anti-shortcut comparisons are published under v5.1 artifact names.
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


def _checkpoint_exists(root: Path, payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    checkpoint_path = payload.get("checkpoint_path")
    if not checkpoint_path:
        return False
    path = Path(str(checkpoint_path))
    if not path.is_absolute():
        path = root / path
    return path.exists()


def _backend_is_real(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    backend = str(payload.get("backend") or payload.get("evaluation_backend") or "").lower()
    if not backend:
        return False
    proxy_markers = ("proxy", "hash", "lexical", "deterministic_hashing")
    real_markers = ("sentence_transformers", "transformers", "huggingface", "hf")
    return any(marker in backend for marker in real_markers) and not any(
        marker in backend for marker in proxy_markers
    )


def compute_v51_real_training(root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = root / "outputs_v2" / "artifacts"
    plan_text = _read_text(root / "docs" / "v51_plan.md")
    v5_proxy = _read_json(artifact_root / "latest_stage2_v5_encoder_compare.json")
    backbone = _read_json(artifact_root / "latest_stage2_v51_real_backbone_compare.json")
    training = _read_json(artifact_root / "latest_stage2_v51_real_training.json")
    latent = _read_json(artifact_root / "latest_stage2_v51_latent_eval.json")
    personamem = _read_json(artifact_root / "latest_stage2_v51_personamem_full589.json")
    decision = _read_json(artifact_root / "latest_stage2_v51_real_training_decision.json")

    checks: dict[str, bool] = {
        "plan_exists": "Real Pretrained Training" in plan_text and "score cap" in plan_text,
        "v5_proxy_marked_not_loaded": bool(
            v5_proxy and v5_proxy.get("pretrained_weights_loaded") is False
        ),
        "real_pretrained_loaded": _as_bool(backbone, "pretrained_weights_loaded"),
        "real_backend": _backend_is_real(backbone),
        "uses_no_personamem_gold_for_backbone": bool(
            backbone and backbone.get("uses_personamem_gold") is False
        ),
        "real_checkpoint_exists": _checkpoint_exists(root, training),
        "train_samples_10k": _as_int(training, "train_samples") >= 10000,
        "uses_no_personamem_gold_for_training": bool(
            training and training.get("uses_personamem_gold") is False
        ),
        "trained_beats_frozen": _as_bool(latent, "trained_beats_frozen"),
        "latent_only_beats_shuffled": _as_float(latent, "latent_only") > _as_float(
            latent, "shuffled_latent"
        ),
        "full_beats_text_only": _as_float(latent, "full") > _as_float(latent, "text_only"),
        "personamem_full589": _as_int(personamem, "sample_count") == 589,
        "personamem_no_cal_beats_baselines": (
            _as_float(personamem, "no_calibration_accuracy")
            > max(
                _as_float(personamem, "option_only_accuracy"),
                _as_float(personamem, "random_baseline"),
            )
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
        "plan_exists": 5,
        "v5_proxy_marked_not_loaded": 5,
        "real_pretrained_loaded": 10,
        "real_backend": 10,
        "uses_no_personamem_gold_for_backbone": 5,
        "real_checkpoint_exists": 10,
        "train_samples_10k": 10,
        "uses_no_personamem_gold_for_training": 5,
        "trained_beats_frozen": 10,
        "latent_only_beats_shuffled": 7,
        "full_beats_text_only": 8,
        "personamem_full589": 5,
        "personamem_no_cal_beats_baselines": 5,
        "personamem_gold_not_used_for_substrate": 3,
        "decision_exists": 2,
    }
    score = sum(weight for name, weight in weights.items() if checks[name])

    caps: list[dict[str, Any]] = []
    if not checks["real_pretrained_loaded"]:
        caps.append({"reason": "pretrained_weights_loaded_not_true", "cap": 20})
    if not checks["real_backend"]:
        caps.append({"reason": "backend_not_real_pretrained_stack", "cap": 20})
    if not checks["real_checkpoint_exists"]:
        caps.append({"reason": "missing_real_checkpoint", "cap": 35})
    if not checks["train_samples_10k"]:
        caps.append({"reason": "train_samples_below_10000", "cap": 45})
    if not checks["trained_beats_frozen"]:
        caps.append({"reason": "trained_does_not_beat_frozen", "cap": 65})
    if not checks["latent_only_beats_shuffled"]:
        caps.append({"reason": "latent_only_does_not_beat_shuffled", "cap": 70})
    if not checks["full_beats_text_only"]:
        caps.append({"reason": "full_does_not_beat_text_only", "cap": 75})
    if checks["personamem_full589"] and not checks["personamem_no_cal_beats_baselines"]:
        caps.append({"reason": "personamem_no_calibration_not_above_baselines", "cap": 80})
    personamem_gold_ok = personamem is None or personamem.get("gold_used_for_memory_substrate") is False
    if training and not (
        checks["uses_no_personamem_gold_for_backbone"]
        and checks["uses_no_personamem_gold_for_training"]
        and personamem_gold_ok
    ):
        caps.append({"reason": "gold_isolation_not_fully_proven", "cap": 30})

    applied_cap = min((cap["cap"] for cap in caps), default=100)
    capped_score = min(score, applied_cap)

    return {
        "metric": "stage2_v51_real_training_score",
        "score": capped_score,
        "raw_score": score,
        "total": 100,
        "checks": checks,
        "caps": caps,
        "applied_cap": applied_cap,
        "stop_ready": capped_score >= 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_v51_real_training(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
