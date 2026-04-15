"""Mechanical verifier for the semantic-first learned-model stage-2 line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _contains_all(path: Path, patterns: list[str]) -> bool:
    text = _read_text(path)
    return bool(text) and all(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_trained_eval(root: Path) -> dict[str, float]:
    best_token_f1 = 0.0
    best_lifecycle = 0.0
    best_belief = 0.0
    best_retrieval = 0.0
    for path in sorted((root / "outputs_v2" / "evals_local").glob("*stage2_local_eval.json")):
        payload = _read_json(path)
        trained = payload.get("trained_eval") or {}
        metrics = trained.get("metrics") or {}
        per_task = trained.get("per_task") or {}
        best_token_f1 = max(best_token_f1, float(metrics.get("token_f1", 0.0)))
        best_lifecycle = max(
            best_lifecycle,
            float((per_task.get("lifecycle_prediction") or {}).get("token_f1", 0.0)),
        )
        best_belief = max(
            best_belief,
            float((per_task.get("composition_to_belief") or {}).get("token_f1", 0.0)),
        )
        best_retrieval = max(
            best_retrieval,
            float((per_task.get("retrieval_alignment") or {}).get("token_f1", 0.0)),
        )
    return {
        "token_f1": best_token_f1,
        "lifecycle_token_f1": best_lifecycle,
        "belief_token_f1": best_belief,
        "retrieval_token_f1": best_retrieval,
    }


def _contains_no_shortcuts(root: Path) -> bool:
    text = _read_text(root / "scripts" / "run_stage2_memory_canary.py")
    banned = [
        "Latent matcher candidate:",
        "_finalize_personamem_provider_prediction",
        "_best_personamem_option_label(",
        "blank-provider fallback",
    ]
    return bool(text) and not any(pattern in text for pattern in banned)


def _online_system_avoids_symbolic_fallback(root: Path) -> bool:
    text = _read_text(root / "src" / "core_mem" / "v2" / "system.py")
    banned = [
        "symbolic_fallback",
        'return fallback, "symbolic"',
        'return fallback, "symbolic_fallback"',
    ]
    return bool(text) and not any(pattern in text for pattern in banned)


def _code_contains_any(root: Path, paths: list[str], patterns: list[str]) -> bool:
    for rel in paths:
        text = _read_text(root / rel)
        if text and any(pattern in text for pattern in patterns):
            return True
    return False


def compute_semantic_model_score(root: Path) -> dict[str, Any]:
    current_status = root / "docs" / "current_status.md"
    implementation_plan = root / "docs" / "implementation_plan.md"
    project_index = root / ".agent-os" / "project-index.md"
    todo = root / ".agent-os" / "todo.md"
    longrun_plan = root / "docs" / "v21_longrun_plan.md"
    best = _best_trained_eval(root)

    checks = {
        "current_status_tracks_td031_semantic_shift": _contains_all(
            current_status,
            ["`TD-031`", "语义优先", "格式外部约束"],
        ),
        "implementation_plan_tracks_stage_p": _contains_all(
            implementation_plan,
            ["阶段 P：Semantic-First Learned Decoder", "语义字段", "格式约束外置"],
        ),
        "agentos_tracks_td031_ws017": _contains_all(
            project_index,
            ["Top next action: `TD-031`", "Active workstreams: `WS-017`"],
        ) and _contains_all(todo, ["`TD-031` `[doing]`", "semantic-first"]),
        "longrun_plan_mentions_semantic_first_constraints": _contains_all(
            longrun_plan,
            ["语义正确优先于 raw JSON 表面格式完全匹配", "阶段 F：语义优先训练重构"],
        ) and (
            "格式约束外置" in _read_text(longrun_plan)
            or "格式外部约束处理" in _read_text(longrun_plan)
        ),
        "benchmark_runner_avoids_shortcuts": _contains_no_shortcuts(root),
        "online_system_avoids_symbolic_fallback": _online_system_avoids_symbolic_fallback(root),
        "online_system_supports_learned_memory": _code_contains_any(
            root,
            ["src/core_mem/v2/system.py", "scripts/run_stage2_memory_canary.py"],
            ["memory_mode", "learned_memory", "use_learned_memory"],
        ),
        "training_supports_online_aligned": _code_contains_any(
            root,
            ["src/core_mem/v2/training.py", "scripts/train_stage2.py", "configs/stage2_train.yaml"],
            ["online_aligned", "composition_to_belief", "retrieval_alignment", "lifecycle_prediction"],
        ),
        "training_not_locked_to_raw_json_exactness": not _code_contains_any(
            root,
            ["src/core_mem/v2/training.py"],
            ["return only compact JSON that matches the target schema"],
        ),
        "semantic_eval_metrics_exist": _code_contains_any(
            root,
            ["src/core_mem/v2/training.py", "scripts/eval_stage2_local.py", "src/core_mem/v2/eval_local.py"],
            ["json_validity_rate", "semantic_validity_rate", "field_f1", "field_accuracy"],
        ),
        "best_trained_eval_token_f1_ge_0_20": best["token_f1"] >= 0.20,
        "best_trained_eval_token_f1_ge_0_45": best["token_f1"] >= 0.45,
        "best_lifecycle_token_f1_ge_0_55": best["lifecycle_token_f1"] >= 0.55,
        "best_belief_token_f1_ge_0_60": best["belief_token_f1"] >= 0.60,
        "best_retrieval_token_f1_ge_0_05": best["retrieval_token_f1"] >= 0.05,
        "current_head_personamem_learned_128_exists": (root / "outputs_v2" / "artifacts" / "latest_personamem_stage2_learned_canary.json").exists(),
        "current_head_longmemeval_learned_64_exists": (root / "outputs_v2" / "artifacts" / "latest_longmemeval_stage2_learned_canary.json").exists(),
    }
    score = sum(1 for value in checks.values() if value)
    return {
        "metric": "stage2_v21_semantic_model_score",
        "score": score,
        "total": len(checks),
        "checks": checks,
        "best_trained_eval_token_f1": best["token_f1"],
        "best_lifecycle_token_f1": best["lifecycle_token_f1"],
        "best_belief_token_f1": best["belief_token_f1"],
        "best_retrieval_token_f1": best["retrieval_token_f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    payload = compute_semantic_model_score(REPO_ROOT)
    if args.score_only:
        print(payload["score"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
